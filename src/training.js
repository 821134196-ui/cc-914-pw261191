// Deterministic tiny model and simulated data-parallel workers.
//
// The model is linear regression with a bias feature fixed at 1. SGD has no
// mutable optimizer-state object beyond weights. Gradients from simulated
// workers are reduced in canonical global sample/sequence order, which is what
// makes the 4-worker and 2-worker floating-point traces bit-for-bit identical.

import {
  ReshardableIterator,
  arraysEqual,
  bitwiseArraysEqual,
  hashSeed,
  maxNumericDifference,
  mulberry32,
  resolveHalfBatchRecovery,
} from './reshard-iterator.js';

export const DEFAULT_CONFIG = Object.freeze({
  datasetSize: 32,
  batchSize: 8,
  worldSizeBefore: 4,
  worldSizeAfter: 2,
  steps: 12,
  failStep: 3,
  partialWorkers: 2,
  learningRate: 0.08,
  seed: 2611914,
  initialWeights: Object.freeze([0.2, -0.15, 0.05]),
});

export function makeDataset(size = DEFAULT_CONFIG.datasetSize, seed = DEFAULT_CONFIG.seed) {
  const random = mulberry32(hashSeed(`tiny-model-dataset:${seed}`));
  const samples = [];

  for (let i = 0; i < size; i += 1) {
    const x1 = Math.round((random() * 2 - 1) * 200) / 100; // [-2, 2]
    const x2 = Math.round((random() * 2 - 1) * 100) / 100; // [-1, 1]
    const noise = (random() - 0.5) * 0.08;
    const y = Math.round((1.3 * x1 - 1.7 * x2 + 0.25 + noise) * 1000) / 1000;

    samples.push({
      id: `S${String(i + 1).padStart(2, '0')}`,
      x: [x1, x2, 1],
      y,
    });
  }

  return samples;
}

export function dot(weights, features) {
  let result = 0;
  for (let i = 0; i < weights.length; i += 1) {
    result += weights[i] * features[i];
  }
  return result;
}

export function meanSquaredError(samples, weights) {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (const sample of samples) {
    const error = dot(weights, sample.x) - sample.y;
    sum += error * error;
  }
  return sum / samples.length;
}

export function computeSampleGradient(sample, weights) {
  const prediction = dot(weights, sample.x);
  const error = prediction - sample.y;
  return sample.x.map((xValue) => 2 * error * xValue);
}

export function computeShardResult(shard, weights) {
  const localGradientSum = new Array(weights.length).fill(0);
  const sampleGradients = [];

  for (const sample of shard.samples) {
    const gradient = computeSampleGradient(sample, weights);
    sampleGradients.push({
      id: sample.id,
      sequencePosition: sample.sequencePosition,
      gradient,
    });
    for (let j = 0; j < weights.length; j += 1) {
      localGradientSum[j] += gradient[j];
    }
  }

  return {
    rank: shard.rank,
    start: shard.start,
    end: shard.end,
    size: shard.size,
    sampleIds: shard.sampleIds,
    localGradientSum,
    sampleGradients,
  };
}

/**
 * Deterministic global reduction. Worker boundaries do not define summation
 * order; sequence positions do. This avoids a 4-worker vs 2-worker difference
 * caused solely by different floating-point addition grouping.
 */
export function reduceGlobalGradient(workerResults, globalBatchSize) {
  const records = workerResults.flatMap((worker) => worker.sampleGradients);
  records.sort((a, b) => a.sequencePosition - b.sequencePosition);

  const dimensions = workerResults[0]?.localGradientSum.length ?? 0;
  const gradient = new Array(dimensions).fill(0);

  for (const record of records) {
    for (let j = 0; j < dimensions; j += 1) {
      gradient[j] += record.gradient[j];
    }
  }

  for (let j = 0; j < dimensions; j += 1) {
    gradient[j] /= globalBatchSize;
  }

  return gradient;
}

export function executePlannedBatch(batch, weights, learningRate) {
  const weightsBefore = [...weights];
  const lossBefore = meanSquaredError(batch.samples, weightsBefore);
  const workerResults = batch.shards.map((shard) => computeShardResult(shard, weightsBefore));
  const gradient = reduceGlobalGradient(workerResults, batch.batchSize);
  const delta = gradient.map((value) => -learningRate * value);
  const weightsAfter = weightsBefore.map((value, index) => value + delta[index]);
  const lossAfter = meanSquaredError(batch.samples, weightsAfter);

  return {
    step: batch.step,
    epoch: batch.epoch,
    baseCursor: batch.baseCursor,
    endCursor: batch.endCursor,
    worldSize: batch.worldSize,
    sampleIds: batch.sampleIds,
    sequencePositions: batch.sequencePositions,
    shards: batch.shards.map((shard) => ({
      rank: shard.rank,
      start: shard.start,
      end: shard.end,
      size: shard.size,
      sampleIds: shard.sampleIds,
    })),
    workerResults,
    weightsBefore,
    gradient,
    delta,
    weightsAfter,
    lossBefore,
    lossAfter,
  };
}

function serializableCheckpoint(iteratorCheckpoint, weights, committedStep) {
  // JSON round-trip intentionally demonstrates that the checkpoint is plain
  // data. JavaScript's number serialization round-trips IEEE-754 doubles.
  return JSON.parse(JSON.stringify({
    ...iteratorCheckpoint,
    committedStep,
    weights: [...weights],
    optimizer: { type: 'sgd', learningRate: DEFAULT_CONFIG.learningRate },
  }));
}

function createIterator(config, worldSize) {
  const samples = config.samples ?? makeDataset(config.datasetSize, config.seed);
  return new ReshardableIterator({
    samples,
    batchSize: config.batchSize,
    worldSize,
    seed: config.seed,
    cursor: 0,
  });
}

export function runUninterruptedTraining(overrides = {}) {
  const config = { ...DEFAULT_CONFIG, ...overrides };
  const samples = config.samples ?? makeDataset(config.datasetSize, config.seed);
  const iterator = createIterator({ ...config, samples }, config.worldSizeBefore);
  let weights = [...config.initialWeights];
  const steps = [];

  for (let index = 0; index < config.steps; index += 1) {
    const batch = iterator.planBatch(config.worldSizeBefore);
    const executed = executePlannedBatch(batch, weights, config.learningRate);
    weights = executed.weightsAfter;
    const iteratorCheckpoint = iterator.commitBatch(batch);
    const checkpoint = serializableCheckpoint(iteratorCheckpoint, weights, index + 1);
    const trace = {
      ...executed,
      runId: 'baseline-4-workers',
      phase: 'uninterrupted',
      status: 'committed',
      checkpoint,
    };
    steps.push(trace);
  }

  return {
    runId: 'baseline',
    worldSize: config.worldSizeBefore,
    steps,
    finalWeights: weights,
    finalCheckpoint: steps.at(-1)?.checkpoint ?? null,
  };
}

/**
 * Run 4 workers until failStep. At that step only partialWorkers shards
 * finish. Their update is discarded; restore from the last committed
 * checkpoint and replay the whole global batch with worldSizeAfter workers.
 */
export function runScaledRecoveryTraining(overrides = {}) {
  const config = { ...DEFAULT_CONFIG, ...overrides };
  const samples = config.samples ?? makeDataset(config.datasetSize, config.seed);
  let iterator = createIterator({ ...config, samples }, config.worldSizeBefore);
  let weights = [...config.initialWeights];
  let checkpoint = serializableCheckpoint(iterator.checkpoint(), weights, 0);
  const initialCheckpoint = checkpoint;

  const committedSteps = [];
  const timeline = [];
  const partialAttempts = [];
  let recoveryDecision = null;

  const commitTrace = (batch, executed, phase, status, runId) => {
    weights = executed.weightsAfter;
    const iteratorCheckpoint = iterator.commitBatch(batch);
    const nextCheckpoint = serializableCheckpoint(
      iteratorCheckpoint,
      weights,
      committedSteps.length + 1,
    );
    checkpoint = nextCheckpoint;
    const trace = {
      ...executed,
      runId,
      phase,
      status,
      checkpoint: nextCheckpoint,
    };
    committedSteps.push(trace);
    timeline.push(trace);
    return trace;
  };

  for (let index = 0; index < config.steps; index += 1) {
    if (index < config.failStep) {
      const batch = iterator.planBatch(config.worldSizeBefore);
      const executed = executePlannedBatch(batch, weights, config.learningRate);
      commitTrace(batch, executed, 'before-scale', 'committed', 'recovery-4-to-2');
      continue;
    }

    if (index === config.failStep) {
      const failedPlan = iterator.planBatch(config.worldSizeBefore);
      const completedShards = failedPlan.shards.slice(0, config.partialWorkers);
      const partialWorkerResults = completedShards.map((shard) =>
        computeShardResult(shard, weights)
      );
      const completedSampleIds = completedShards.flatMap((shard) => shard.sampleIds);
      const incompleteBatch = {
        committed: false,
        baseCursor: failedPlan.baseCursor,
        observedCursor: failedPlan.baseCursor + completedSampleIds.length,
        step: failedPlan.step,
        worldSize: config.worldSizeBefore,
        completedRanks: completedShards.map((shard) => shard.rank),
        completedSampleIds,
      };

      const partialEvent = {
        kind: 'partial-attempt',
        runId: 'recovery-4-to-2',
        step: failedPlan.step,
        globalStepIndex: index,
        phase: 'failed-half-batch',
        status: 'discarded-uncommitted',
        worldSize: config.worldSizeBefore,
        completedRanks: incompleteBatch.completedRanks,
        completedSampleIds,
        allSampleIds: failedPlan.sampleIds,
        shards: failedPlan.shards.map((shard) => ({
          rank: shard.rank,
          sampleIds: shard.sampleIds,
          completed: shard.rank < config.partialWorkers,
        })),
        workerResults: partialWorkerResults,
        note: 'Only local shard gradients exist. No global optimizer step or checkpoint was committed.',
      };
      partialAttempts.push(partialEvent);
      timeline.push(partialEvent);

      // The durable checkpoint remains at the batch boundary. Resolve the
      // half-finished attempt into a rollback/replay decision and a new
      // iterator sharded for worldSizeAfter.
      const recovery = resolveHalfBatchRecovery({
        checkpoint,
        incompleteBatch,
        newWorldSize: config.worldSizeAfter,
        samples,
      });
      const { iterator: recoveredIterator, ...decisionSummary } = recovery;
      recoveryDecision = decisionSummary;
      iterator = recoveredIterator;

      const replayBatch = iterator.planBatch(config.worldSizeAfter);
      const replayExecuted = executePlannedBatch(replayBatch, weights, config.learningRate);
      commitTrace(
        replayBatch,
        replayExecuted,
        'replay-after-scale',
        'committed-replayed-full-batch',
        'recovery-4-to-2',
      );
      continue;
    }

    const batch = iterator.planBatch(config.worldSizeAfter);
    const executed = executePlannedBatch(batch, weights, config.learningRate);
    commitTrace(batch, executed, 'after-scale', 'committed', 'recovery-4-to-2');
  }

  return {
    runId: 'recovery',
    worldSizeBefore: config.worldSizeBefore,
    worldSizeAfter: config.worldSizeAfter,
    steps: committedSteps,
    committedSteps,
    timeline,
    partialAttempts,
    recoveryDecision,
    initialCheckpoint,
    finalWeights: weights,
    finalCheckpoint: committedSteps.at(-1)?.checkpoint ?? checkpoint,
  };
}

export function compareTrainingRuns(baseline, recovery) {
  const stepComparisons = baseline.steps.map((baselineStep, index) => {
    const recoveryStep = recovery.committedSteps[index];
    const sampleOrderExact = arraysEqual(baselineStep.sampleIds, recoveryStep.sampleIds)
      && arraysEqual(baselineStep.sequencePositions, recoveryStep.sequencePositions);
    const gradientExact = bitwiseArraysEqual(baselineStep.gradient, recoveryStep.gradient);
    const deltaExact = bitwiseArraysEqual(baselineStep.delta, recoveryStep.delta);
    const weightsExact = bitwiseArraysEqual(
      baselineStep.weightsAfter,
      recoveryStep.weightsAfter,
    );

    return {
      index,
      step: baselineStep.step,
      epoch: baselineStep.epoch,
      baselineWorldSize: baselineStep.worldSize,
      recoveryWorldSize: recoveryStep.worldSize,
      baselineSampleIds: baselineStep.sampleIds,
      recoverySampleIds: recoveryStep.sampleIds,
      baselineShards: baselineStep.shards,
      recoveryShards: recoveryStep.shards,
      sampleOrderExact,
      gradientExact,
      deltaExact,
      weightsExact,
      gradientMaxDiff: maxNumericDifference(baselineStep.gradient, recoveryStep.gradient),
      weightsMaxDiff: maxNumericDifference(
        baselineStep.weightsAfter,
        recoveryStep.weightsAfter,
      ),
      baselineGradient: baselineStep.gradient,
      recoveryGradient: recoveryStep.gradient,
      baselineWeightsAfter: baselineStep.weightsAfter,
      recoveryWeightsAfter: recoveryStep.weightsAfter,
      baselineLoss: baselineStep.lossAfter,
      recoveryLoss: recoveryStep.lossAfter,
    };
  });

  const baselineOrder = baseline.steps.flatMap((step) => step.sampleIds);
  const recoveryOrder = recovery.committedSteps.flatMap((step) => step.sampleIds);
  const flattenedOrderExact = arraysEqual(baselineOrder, recoveryOrder);
  const finalWeightsExact = bitwiseArraysEqual(baseline.finalWeights, recovery.finalWeights);
  const maxGradientDiff = Math.max(...stepComparisons.map((item) => item.gradientMaxDiff));
  const maxWeightDiff = Math.max(...stepComparisons.map((item) => item.weightsMaxDiff));
  const exact = flattenedOrderExact
    && stepComparisons.every((item) =>
      item.sampleOrderExact && item.gradientExact && item.deltaExact && item.weightsExact)
    && finalWeightsExact;

  return {
    exact,
    totalSteps: stepComparisons.length,
    flattenedOrderExact,
    flattenedOrder: { baseline: baselineOrder, recovery: recoveryOrder },
    maxGradientDiff,
    maxWeightDiff,
    finalWeightsExact,
    finalWeights: {
      baseline: baseline.finalWeights,
      recovery: recovery.finalWeights,
    },
    steps: stepComparisons,
  };
}

export function runComparison(overrides = {}) {
  const config = { ...DEFAULT_CONFIG, ...overrides };
  const samples = config.samples ?? makeDataset(config.datasetSize, config.seed);
  const resolvedConfig = { ...config, samples };
  const baseline = runUninterruptedTraining(resolvedConfig);
  const recovery = runScaledRecoveryTraining(resolvedConfig);
  const comparison = compareTrainingRuns(baseline, recovery);

  return { config: resolvedConfig, baseline, recovery, comparison };
}
