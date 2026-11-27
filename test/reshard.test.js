import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  ReshardableIterator,
  alignToGlobalBatchStart,
  arraysEqual,
  globalOrder,
  resolveHalfBatchRecovery,
} from '../src/reshard-iterator.js';
import { makeDataset, runComparison } from '../src/training.js';

const samples = makeDataset();

test('global order is deterministic and epoch-dependent', () => {
  assert.deepEqual(globalOrder(32, 0, 2611914), globalOrder(32, 0, 2611914));
  assert.notDeepEqual(globalOrder(32, 0, 2611914), globalOrder(32, 1, 2611914));
  assert.equal(new Set(globalOrder(32, 0, 2611914)).size, 32);
});

test('four-worker and two-worker shards flatten to the same global batch', () => {
  const iterator = new ReshardableIterator({ samples, batchSize: 8, worldSize: 4 });
  const four = iterator.planBatch(4);
  const two = iterator.planBatch(2);

  assert.deepEqual(four.sampleIds, two.sampleIds);
  assert.deepEqual(four.shards.flatMap((shard) => shard.sampleIds), four.sampleIds);
  assert.deepEqual(two.shards.flatMap((shard) => shard.sampleIds), two.sampleIds);
  assert.deepEqual(four.shards.map((shard) => shard.size), [2, 2, 2, 2]);
  assert.deepEqual(two.shards.map((shard) => shard.size), [4, 4]);
});

test('checkpoint restores the same global position under a new world size', () => {
  const before = new ReshardableIterator({ samples, batchSize: 8, worldSize: 4 });
  let checkpoint = before.checkpoint();

  const first = before.planBatch(4);
  checkpoint = before.commitBatch(first);
  const secondFour = before.planBatch(4);

  const restored = ReshardableIterator.fromCheckpoint(
    { ...checkpoint, samples },
    { samples, worldSize: 2 },
  );
  const secondTwo = restored.planBatch(2);

  assert.equal(checkpoint.cursor, 8);
  assert.deepEqual(secondTwo.sampleIds, secondFour.sampleIds);
  assert.equal(secondTwo.worldSize, 2);
});

test('a half-finished batch rewinds to its boundary and is replayed', () => {
  assert.equal(alignToGlobalBatchStart(28, 8, 32), 24);

  const checkpoint = {
    version: 1,
    datasetSize: 32,
    batchSize: 8,
    seed: 2611914,
    dropLast: true,
    cursor: 24,
  };
  const decision = resolveHalfBatchRecovery({
    checkpoint,
    newWorldSize: 2,
    samples,
    incompleteBatch: {
      committed: false,
      baseCursor: 24,
      observedCursor: 28,
      completedRanks: [0, 1],
      completedSampleIds: ['a', 'b', 'c', 'd'],
    },
  });

  assert.equal(decision.resumeCursor, 24);
  assert.equal(decision.rawCursor, 28);
  assert.equal(decision.rewoundSamples, 4);
  assert.equal(decision.partialUpdateCommitted, false);
  assert.equal(decision.iterator.worldSize, 2);

  const replay = decision.iterator.planBatch(2);
  assert.equal(replay.baseCursor, 24);
  assert.equal(replay.sampleIds.length, 8);
});

test('out-of-order batch commit is rejected', () => {
  const iterator = new ReshardableIterator({ samples, batchSize: 8 });
  const batch = iterator.planBatch();
  iterator.cursor = 8;
  assert.throws(() => iterator.commitBatch(batch), /does not match/);
});

for (const failStep of [0, 1, 3, 7, 11]) {
  for (const partialWorkers of [1, 2, 3]) {
    test(`baseline matches 4→2 recovery at failStep=${failStep}, partial=${partialWorkers}`, () => {
      const result = runComparison({ failStep, partialWorkers });
      const { comparison } = result;

      assert.equal(comparison.totalSteps, 12);
      assert.equal(comparison.flattenedOrderExact, true);
      assert.equal(comparison.exact, true);
      assert.equal(comparison.maxGradientDiff, 0);
      assert.equal(comparison.maxWeightDiff, 0);
      assert.ok(arraysEqual(comparison.finalWeights.baseline, comparison.finalWeights.recovery));

      for (const step of comparison.steps) {
        assert.equal(step.sampleOrderExact, true);
        assert.equal(step.gradientExact, true);
        assert.equal(step.deltaExact, true);
        assert.equal(step.weightsExact, true);
      }
    });
  }
}

test('partial attempt does not create an optimizer step', () => {
  const result = runComparison({ failStep: 3, partialWorkers: 2 });
  assert.equal(result.baseline.steps.length, 12);
  assert.equal(result.recovery.committedSteps.length, 12);
  assert.equal(result.recovery.partialAttempts.length, 1);
  assert.equal(result.recovery.partialAttempts[0].status, 'discarded-uncommitted');
  assert.equal(result.recovery.timeline.filter((item) => item.kind === 'partial-attempt').length, 1);
});
