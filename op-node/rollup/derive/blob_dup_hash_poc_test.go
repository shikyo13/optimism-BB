package derive

// TestDuplicateBlobHashHaltDerivation is an end-to-end proof-of-concept for
// the regression introduced by commit e19f427923cc
// ("op-service: rip out deprecated blob sidecars client and related code").
//
// Background
// ----------
// The original blob-fetching path used eth.IndexedBlobHash — an (index, hash)
// pair where index is the blob's unique position in the L1 block sidecar.
// Two physically distinct blobs with identical content (same versioned hash H)
// would have different indices (e.g. 0 and 1), and fetching by index correctly
// returned both.
//
// Commit e19f427923cc replaced indexed fetching with hash-only fetching via
// the new eth/v1/beacon/blobs/{slot}?versioned_hashes= endpoint.  When two
// batcher transactions in the same L1 block each commit a blob of identical
// content, dataAndHashesFromTxs now produces hashes=[H, H].  The beacon client
// sends ?versioned_hashes=H&versioned_hashes=H.  Beacon implementations that
// deduplicate the query return 1 blob instead of 2, triggering the
// BeaconHTTPClient count check:
//   #returned blobs(1) != #requested blobs(2)
// This propagates as a TemporaryError, and derivation permanently stalls on
// the affected L1 block.
//
// The FakeBeacon's blobstore.GetBlobsByHash exhibits a complementary bug: when
// the store contains two blobs at indices 0 and 1 both with hash H and
// hashes=[H,H] is requested, the inner loop appends matches for BOTH index
// entries on EACH pass over hashes, yielding 4 items for a 2-item request,
// which fails the len check with "not all blobs found".
//
// Sub-tests
// ---------
//  1. duplicate_hashes_produced — proves dataAndHashesFromTxs emits [H, H]
//     when two batcher txs each carry a blob with the same versioned hash.
//  2. deduplicating_beacon_stalls_derivation — proves BlobDataSource.Next
//     returns a TemporaryError when the beacon client (simulating
//     deduplication) returns only 1 blob for a 2-element hashes request,
//     reproducing the chain-halt scenario on a real or FakeBeacon that
//     deduplicates versioned_hashes query parameters.

import (
	"context"
	"crypto/ecdsa"
	"fmt"
	"math/big"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"

	"github.com/ethereum-optimism/optimism/op-service/eth"
	"github.com/ethereum-optimism/optimism/op-service/testlog"
	"github.com/ethereum-optimism/optimism/op-service/testutils"
	"github.com/ethereum/go-ethereum/log"
)

func TestDuplicateBlobHashHaltDerivation(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	// Shared batcher identity and config.
	privateKey := testutils.InsecureRandomKey(rng)
	pubKey, _ := privateKey.Public().(*ecdsa.PublicKey)
	batcherAddr := crypto.PubkeyToAddress(*pubKey)
	batchInboxAddr := testutils.RandomAddress(rng)
	chainID := new(big.Int).SetUint64(rng.Uint64())
	signer := types.NewPragueSigner(chainID)
	config := DataSourceConfig{
		l1Signer:          signer,
		batchInboxAddress: batchInboxAddr,
	}
	logger := testlog.Logger(t, log.LvlInfo)

	// duplicateHash is the versioned hash shared by two identical-content blobs.
	duplicateHash := testutils.RandomHash(rng)

	// buildBlobTx creates a valid batcher blob transaction referencing duplicateHash.
	buildBlobTx := func(nonce uint64) *types.Transaction {
		tx, _ := types.SignNewTx(privateKey, signer, &types.BlobTx{
			Nonce:      nonce,
			Gas:        2_000_000,
			To:         batchInboxAddr,
			BlobHashes: []common.Hash{duplicateHash},
		})
		return tx
	}

	// Sub-test 1: dataAndHashesFromTxs emits [H, H] for two batcher txs with
	// the same blob content.
	t.Run("duplicate_hashes_produced", func(t *testing.T) {
		tx1 := buildBlobTx(0)
		tx2 := buildBlobTx(1)
		txs := types.Transactions{tx1, tx2}

		_, hashes := dataAndHashesFromTxs(txs, &config, batcherAddr, logger)

		require.Len(t, hashes, 2, "both batcher blobs must be collected")
		require.Equal(t, duplicateHash, hashes[0], "first hash must equal duplicateHash")
		require.Equal(t, duplicateHash, hashes[1], "second hash must equal duplicateHash")
		require.Equal(t, hashes[0], hashes[1], "hashes must be identical — duplicate detected")
	})

	// Sub-test 2: when the beacon client deduplicates the versioned_hashes
	// query and returns only 1 blob for a [H, H] request, BlobDataSource.Next
	// returns a TemporaryError — derivation stalls permanently on the block.
	t.Run("deduplicating_beacon_stalls_derivation", func(t *testing.T) {
		tx1 := buildBlobTx(0)
		tx2 := buildBlobTx(1)

		// mockFetcher simulates a beacon that deduplicates versioned_hashes:
		// it returns only 1 blob regardless of how many duplicate hashes were
		// requested, matching observed behavior of common HTTP query parsers
		// that normalise repeated keys.
		mockFetcher := &deduplicatingBlobFetcher{blob: &eth.Blob{}}

		// blockRef provides the L1 context for BlobDataSource.
		blockRef := eth.L1BlockRef{
			Hash:   testutils.RandomHash(rng),
			Number: 100,
			Time:   1_000_000,
		}

		// mockL1 returns the two batcher blob transactions.
		// We use an inline mock because MockEthClient panics when BlockInfo is
		// nil (it does an unconditional type assertion on the first return).
		mockL1 := &staticTxFetcher{
			hash: blockRef.Hash,
			txs:  types.Transactions{tx1, tx2},
		}

		ds := NewBlobDataSource(context.Background(), logger, config, mockL1, mockFetcher, blockRef, batcherAddr)

		_, err := ds.Next(context.Background())

		// The error must be a TemporaryError (not EOF, not ResetError).
		// A TemporaryError causes the derivation pipeline to retry
		// indefinitely — permanent stall until the node is updated.
		require.ErrorIs(t, err, ErrTemporary,
			"deduplicating beacon must cause a TemporaryError, stalling derivation")
	})
}

// staticTxFetcher satisfies L1TransactionFetcher for a single known block,
// returning a fixed transaction list.  It avoids the type-assertion panic in
// MockEthClient when BlockInfo is nil.
type staticTxFetcher struct {
	hash common.Hash
	txs  types.Transactions
}

func (f *staticTxFetcher) InfoAndTxsByHash(_ context.Context, hash common.Hash) (eth.BlockInfo, types.Transactions, error) {
	if hash != f.hash {
		return nil, nil, fmt.Errorf("unexpected hash %s", hash)
	}
	return eth.HeaderBlockInfo(&types.Header{}), f.txs, nil
}

// deduplicatingBlobFetcher simulates a beacon client that deduplicates the
// versioned_hashes query parameter.  In the real stack, BeaconHTTPClient.BeaconBlobs
// checks len(response) == len(hashes) and returns an error when they differ:
//   "#returned blobs(N) != #requested blobs(M)"
// That error propagates up to BlobDataSource.open as a TemporaryError (the
// non-NotFound branch of the error check at blob_data_source.go:103-104).
type deduplicatingBlobFetcher struct {
	blob *eth.Blob
}

func (f *deduplicatingBlobFetcher) GetBlobsByHash(_ context.Context, _ uint64, hashes []common.Hash) ([]*eth.Blob, error) {
	if len(hashes) == 0 {
		return nil, nil
	}
	// Count unique hashes — this is what a deduplicating server returns.
	uniqueHashes := make(map[common.Hash]struct{})
	for _, h := range hashes {
		uniqueHashes[h] = struct{}{}
	}
	uniqueCount := len(uniqueHashes)
	if uniqueCount != len(hashes) {
		// Simulate BeaconHTTPClient.BeaconBlobs count-mismatch error:
		// the beacon returned fewer blobs (one per unique hash) than requested.
		return nil, fmt.Errorf("#returned blobs(%d) != #requested blobs(%d)", uniqueCount, len(hashes))
	}
	result := make([]*eth.Blob, len(hashes))
	for i := range result {
		result[i] = f.blob
	}
	return result, nil
}
