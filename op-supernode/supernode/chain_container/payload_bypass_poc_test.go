package chain_container

// TestLegacyDenyListBypassOnPayloadProcess is an end-to-end proof-of-concept
// for the security regression introduced by upstream commit 6cdfde0837
// ("supernode: Record full Output bytes with denied heads").
//
// Background
// ----------
// The denylist was originally written using a raw 32-byte-per-entry binary
// format (concatenated payload hashes). Commit 6cdfde0837 migrated the DB
// to a JSON-encoded DenyRecord struct and deliberately removed the legacy
// fallback decoder, leaving nodes that accumulated entries under the old
// format unable to decode their own denylist.
//
// Impact
// ------
// DenyList.Contains propagates the decode error through
// simpleChainContainer.IsDenied to EngineController.onPayloadProcess.
// onPayloadProcess is fail-open: when IsDenied returns a non-nil error it
// logs the error and calls engine.NewPayload anyway, accepting the payload.
// A malicious sequencer can therefore replay a previously-invalidated block
// on an upgraded node that still has legacy-format entries on disk.
//
// This file contains two sub-tests:
//   1. legacy_format_bypasses_denylist — proves NewPayload is called for a
//      "denied" block whose DB entry is in the legacy 32-byte format.
//   2. json_format_correctly_denies   — proves NewPayload is NOT called when
//      the same block is stored in the current JSON format.

import (
	"context"
	"testing"

	bolt "go.etcd.io/bbolt"
	"github.com/stretchr/testify/require"

	nodeMetrics "github.com/ethereum-optimism/optimism/op-node/metrics"
	"github.com/ethereum-optimism/optimism/op-node/rollup"
	nodeEngine "github.com/ethereum-optimism/optimism/op-node/rollup/engine"
	nodeSync "github.com/ethereum-optimism/optimism/op-node/rollup/sync"
	"github.com/ethereum-optimism/optimism/op-service/eth"
	"github.com/ethereum-optimism/optimism/op-service/testlog"
	"github.com/ethereum-optimism/optimism/op-service/testutils"
	"github.com/ethereum/go-ethereum/common"
)

func TestLegacyDenyListBypassOnPayloadProcess(t *testing.T) {
	const blockNumber = uint64(100)
	deniedHash := common.HexToHash("0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef")

	payload := &eth.ExecutionPayloadEnvelope{
		ExecutionPayload: &eth.ExecutionPayload{
			BlockNumber: eth.Uint64Quantity(blockNumber),
			BlockHash:   deniedHash,
			ParentHash:  common.Hash{},
		},
	}

	// Case 1: legacy 32-byte raw-hash entry → IsDenied errors → NewPayload called (bypass).
	t.Run("legacy_format_bypasses_denylist", func(t *testing.T) {
		dir := t.TempDir()
		dl, err := OpenDenyList(dir)
		require.NoError(t, err)
		defer dl.Close()

		// Write the legacy format directly: 32 raw bytes per entry, no JSON wrapper.
		// This is the format written by op-supernode before commit 6cdfde0837.
		err = dl.db.Update(func(tx *bolt.Tx) error {
			b := tx.Bucket(denyListBucketName)
			return b.Put(heightToKey(blockNumber), deniedHash.Bytes())
		})
		require.NoError(t, err)

		// Step 1: confirm the legacy entry causes a decode error in IsDenied.
		sa := &denyListSuperAuthority{dl: dl}
		denied, isDeniedErr := sa.IsDenied(blockNumber, deniedHash)
		require.Error(t, isDeniedErr, "legacy entry must cause a decode error")
		require.False(t, denied)

		// Step 2: wire through EngineController and assert NewPayload IS called.
		// The fail-open branch in onPayloadProcess logs the IsDenied error and
		// falls through to engine.NewPayload, accepting the payload.
		mockEngine := &testutils.MockEngine{}
		emitter := &testutils.MockEmitter{}

		mockEngine.ExpectNewPayload(payload.ExecutionPayload, nil, &eth.PayloadStatusV1{Status: eth.ExecutionValid}, nil)
		emitter.ExpectOnceType("PayloadSuccessEvent")

		ec := nodeEngine.NewEngineController(
			context.Background(),
			mockEngine,
			testlog.Logger(t, 0),
			nodeMetrics.NoopMetrics,
			&rollup.Config{},
			&nodeSync.Config{},
			false,
			&testutils.MockL1Source{},
			emitter,
			sa,
		)

		ec.OnEvent(context.Background(), nodeEngine.PayloadProcessEvent{
			Envelope: payload,
			Ref:      eth.L2BlockRef{Hash: deniedHash, Number: blockNumber},
		})

		// Assertions pass only if engine.NewPayload WAS called — bypass confirmed.
		mockEngine.AssertExpectations(t)
		emitter.AssertExpectations(t)
	})

	// Case 2: same block stored in current JSON format → correctly denied, NewPayload NOT called.
	t.Run("json_format_correctly_denies", func(t *testing.T) {
		dir := t.TempDir()
		dl, err := OpenDenyList(dir)
		require.NoError(t, err)
		defer dl.Close()

		require.NoError(t, dl.Add(blockNumber, deniedHash, 0, eth.Bytes32{}, eth.Bytes32{}))

		sa := &denyListSuperAuthority{dl: dl}
		denied, err := sa.IsDenied(blockNumber, deniedHash)
		require.NoError(t, err)
		require.True(t, denied, "JSON-format entry must be correctly detected")

		mockEngine := &testutils.MockEngine{}
		emitter := &testutils.MockEmitter{}
		// No engine or emitter expectations: unsafe denied payload is silently dropped.

		ec := nodeEngine.NewEngineController(
			context.Background(),
			mockEngine,
			testlog.Logger(t, 0),
			nodeMetrics.NoopMetrics,
			&rollup.Config{},
			&nodeSync.Config{},
			false,
			&testutils.MockL1Source{},
			emitter,
			sa,
		)

		ec.OnEvent(context.Background(), nodeEngine.PayloadProcessEvent{
			Envelope: payload,
			Ref:      eth.L2BlockRef{Hash: deniedHash, Number: blockNumber},
		})

		// Assertions pass only if engine.NewPayload was NOT called — correct denial.
		mockEngine.AssertExpectations(t)
		emitter.AssertExpectations(t)
	})
}

// denyListSuperAuthority wraps DenyList as a minimal rollup.SuperAuthority.
// It isolates the denylist path from the rest of simpleChainContainer so the
// PoC tests can focus on the IsDenied → onPayloadProcess interaction.
type denyListSuperAuthority struct {
	dl *DenyList
}

func (a *denyListSuperAuthority) IsDenied(height uint64, hash common.Hash) (bool, error) {
	return a.dl.Contains(height, hash)
}

func (a *denyListSuperAuthority) FullyVerifiedL2Head() (eth.BlockID, bool) {
	return eth.BlockID{}, true
}

func (a *denyListSuperAuthority) FinalizedL2Head() (eth.BlockID, bool) {
	return eth.BlockID{}, true
}

var _ rollup.SuperAuthority = (*denyListSuperAuthority)(nil)
