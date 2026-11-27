(() => {
  // src/reshard-iterator.js
  var ITERATOR_VERSION = 1;
  var HALF_BATCH_POLICY = "rollback-and-replay-uncommitted-batch";
  function hashSeed(value) {
    const text = String(value);
    let hash = 2166136261;
    for (let i = 0; i < text.length; i += 1) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  }
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function nextRandom() {
      a += 1831565813;
      let t = a;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function globalOrder(datasetSize, epoch, seed) {
    const order = Array.from({ length: datasetSize }, (_, index) => index);
    const random = mulberry32(hashSeed(`reshard-iterator:${seed}:epoch:${epoch}`));
    for (let i = datasetSize - 1; i > 0; i -= 1) {
      const j = Math.floor(random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    return order;
  }
  function partitionSamples(samples, worldSize) {
    if (!Number.isInteger(worldSize) || worldSize <= 0) {
      throw new RangeError("worldSize must be a positive integer");
    }
    const total = samples.length;
    const baseSize = Math.floor(total / worldSize);
    const extra = total % worldSize;
    const shards = [];
    let offset = 0;
    for (let rank = 0; rank < worldSize; rank += 1) {
      const size = baseSize + (rank < extra ? 1 : 0);
      const shardSamples = samples.slice(offset, offset + size);
      shards.push({
        rank,
        start: offset,
        end: offset + size,
        size,
        sampleIds: shardSamples.map((sample) => sample.id),
        samples: shardSamples
      });
      offset += size;
    }
    return shards;
  }
  function alignToGlobalBatchStart(cursor, batchSize, datasetSize) {
    if (!Number.isInteger(cursor) || cursor < 0) {
      throw new RangeError("cursor must be a non-negative integer");
    }
    const epoch = Math.floor(cursor / datasetSize);
    const offset = cursor - epoch * datasetSize;
    return epoch * datasetSize + Math.floor(offset / batchSize) * batchSize;
  }
  var ReshardableIterator = class _ReshardableIterator {
    constructor(options = {}) {
      const {
        samples,
        batchSize = 8,
        worldSize = 4,
        seed = 2611914,
        cursor = 0,
        dropLast = true
      } = options;
      if (!Array.isArray(samples) || samples.length === 0) {
        throw new TypeError("samples must be a non-empty array");
      }
      if (!Number.isInteger(batchSize) || batchSize <= 0) {
        throw new RangeError("batchSize must be a positive integer");
      }
      if (!Number.isInteger(worldSize) || worldSize <= 0) {
        throw new RangeError("worldSize must be a positive integer");
      }
      if (!Number.isInteger(cursor) || cursor < 0) {
        throw new RangeError("cursor must be a non-negative integer");
      }
      if (dropLast && samples.length % batchSize !== 0) {
        throw new Error(
          `Demo iterator requires datasetSize (${samples.length}) divisible by batchSize (${batchSize}) when dropLast=true`
        );
      }
      this.samples = samples;
      this.datasetSize = samples.length;
      this.batchSize = batchSize;
      this.worldSize = worldSize;
      this.seed = seed;
      this.dropLast = dropLast;
      this.cursor = cursor;
      this.version = ITERATOR_VERSION;
    }
    orderForEpoch(epoch) {
      return globalOrder(this.datasetSize, epoch, this.seed);
    }
    /**
     * Build the global batch at the current committed cursor without mutating
     * that cursor. The caller invokes commitBatch() only after the optimizer
     * update for the whole global batch has completed.
     */
    planBatch(worldSize = this.worldSize) {
      if (!Number.isInteger(worldSize) || worldSize <= 0) {
        throw new RangeError("worldSize must be a positive integer");
      }
      let cursor = this.cursor;
      let epoch = Math.floor(cursor / this.datasetSize);
      let offset = cursor % this.datasetSize;
      if (this.dropLast && offset + this.batchSize > this.datasetSize) {
        cursor = (epoch + 1) * this.datasetSize;
        epoch += 1;
        offset = 0;
      }
      const order = this.orderForEpoch(epoch);
      const globalIndices = order.slice(offset, offset + this.batchSize);
      const orderedSamples = globalIndices.map((datasetIndex, orderIndex) => {
        const sample = this.samples[datasetIndex];
        return {
          ...sample,
          // Sequence position in the infinite deterministic global stream.
          sequencePosition: cursor + orderIndex,
          datasetIndex
        };
      });
      const epochBatch = Math.floor(offset / this.batchSize);
      const batchesPerEpoch = this.datasetSize / this.batchSize;
      const step = epoch * batchesPerEpoch + epochBatch;
      const shards = partitionSamples(orderedSamples, worldSize);
      return {
        kind: "global-batch-plan",
        policy: HALF_BATCH_POLICY,
        step,
        epoch,
        epochBatch,
        baseCursor: cursor,
        endCursor: cursor + orderedSamples.length,
        batchSize: orderedSamples.length,
        requestedBatchSize: this.batchSize,
        worldSize,
        sampleIds: orderedSamples.map((sample) => sample.id),
        sequencePositions: orderedSamples.map((sample) => sample.sequencePosition),
        samples: orderedSamples,
        shards
      };
    }
    commitBatch(batch) {
      if (!batch || batch.kind !== "global-batch-plan") {
        throw new TypeError("commitBatch expects a batch planned by planBatch()");
      }
      if (batch.baseCursor !== this.cursor) {
        throw new Error(
          `Batch base cursor ${batch.baseCursor} does not match iterator cursor ${this.cursor}; refusing out-of-order commit`
        );
      }
      this.cursor = batch.endCursor;
      return this.checkpoint();
    }
    checkpoint(extra = {}) {
      return {
        version: ITERATOR_VERSION,
        policy: HALF_BATCH_POLICY,
        datasetSize: this.datasetSize,
        batchSize: this.batchSize,
        seed: this.seed,
        dropLast: this.dropLast,
        // This is the last COMMITTED boundary, never a half-processed cursor.
        cursor: this.cursor,
        ...extra
      };
    }
    static fromCheckpoint(checkpoint, options = {}) {
      if (!checkpoint || checkpoint.version !== ITERATOR_VERSION) {
        throw new Error("Unsupported or missing iterator checkpoint version");
      }
      return new _ReshardableIterator({
        samples: options.samples ?? checkpoint.samples,
        batchSize: checkpoint.batchSize,
        seed: checkpoint.seed,
        dropLast: checkpoint.dropLast,
        cursor: options.cursor ?? checkpoint.cursor,
        worldSize: options.worldSize ?? checkpoint.worldSize ?? 4
      });
    }
  };
  function resolveHalfBatchRecovery({
    checkpoint,
    incompleteBatch = null,
    observedCursor = null,
    newWorldSize,
    samples
  }) {
    if (!checkpoint) throw new TypeError("checkpoint is required");
    let cursor = checkpoint.cursor;
    let batchBaseCursor = null;
    let recomputeSampleIds = [];
    let discardedCompletedRanks = [];
    let source = "durable-checkpoint";
    if (incompleteBatch && incompleteBatch.committed !== true) {
      batchBaseCursor = incompleteBatch.baseCursor ?? checkpoint.cursor;
      cursor = incompleteBatch.observedCursor ?? incompleteBatch.baseCursor ?? checkpoint.cursor;
      recomputeSampleIds = incompleteBatch.completedSampleIds ?? [];
      discardedCompletedRanks = incompleteBatch.completedRanks ?? [];
      source = "incomplete-batch-rollback";
    } else if (Number.isInteger(observedCursor)) {
      cursor = observedCursor;
      source = "observed-half-cursor";
    }
    const alignedCursor = batchBaseCursor ?? alignToGlobalBatchStart(
      cursor,
      checkpoint.batchSize,
      checkpoint.datasetSize
    );
    const rewoundSamples = Math.max(0, cursor - alignedCursor);
    const iterator = new ReshardableIterator({
      samples: samples ?? checkpoint.samples,
      batchSize: checkpoint.batchSize,
      seed: checkpoint.seed,
      dropLast: checkpoint.dropLast,
      cursor: alignedCursor,
      worldSize: newWorldSize
    });
    return {
      policy: HALF_BATCH_POLICY,
      source,
      rawCursor: cursor,
      resumeCursor: alignedCursor,
      rewoundSamples,
      recomputeSampleIds,
      discardedCompletedRanks,
      partialUpdateCommitted: false,
      iterator
    };
  }
  function arraysEqual(a, b) {
    return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((value, index) => value === b[index]);
  }
  function maxNumericDifference(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
      return Number.POSITIVE_INFINITY;
    }
    return a.reduce((max, value, index) => {
      const diff = Math.abs(value - b[index]);
      return Number.isFinite(diff) ? Math.max(max, diff) : Number.POSITIVE_INFINITY;
    }, 0);
  }
  function bitwiseArraysEqual(a, b) {
    return arraysEqual(a, b) && a.every((value, index) => Object.is(value, b[index]));
  }

  // src/training.js
  var DEFAULT_CONFIG = Object.freeze({
    datasetSize: 32,
    batchSize: 8,
    worldSizeBefore: 4,
    worldSizeAfter: 2,
    steps: 12,
    failStep: 3,
    partialWorkers: 2,
    learningRate: 0.08,
    seed: 2611914,
    initialWeights: Object.freeze([0.2, -0.15, 0.05])
  });
  function makeDataset(size = DEFAULT_CONFIG.datasetSize, seed = DEFAULT_CONFIG.seed) {
    const random = mulberry32(hashSeed(`tiny-model-dataset:${seed}`));
    const samples = [];
    for (let i = 0; i < size; i += 1) {
      const x1 = Math.round((random() * 2 - 1) * 200) / 100;
      const x2 = Math.round((random() * 2 - 1) * 100) / 100;
      const noise = (random() - 0.5) * 0.08;
      const y = Math.round((1.3 * x1 - 1.7 * x2 + 0.25 + noise) * 1e3) / 1e3;
      samples.push({
        id: `S${String(i + 1).padStart(2, "0")}`,
        x: [x1, x2, 1],
        y
      });
    }
    return samples;
  }
  function dot(weights, features) {
    let result = 0;
    for (let i = 0; i < weights.length; i += 1) {
      result += weights[i] * features[i];
    }
    return result;
  }
  function meanSquaredError(samples, weights) {
    if (samples.length === 0) return 0;
    let sum = 0;
    for (const sample of samples) {
      const error = dot(weights, sample.x) - sample.y;
      sum += error * error;
    }
    return sum / samples.length;
  }
  function computeSampleGradient(sample, weights) {
    const prediction = dot(weights, sample.x);
    const error = prediction - sample.y;
    return sample.x.map((xValue) => 2 * error * xValue);
  }
  function computeShardResult(shard, weights) {
    const localGradientSum = new Array(weights.length).fill(0);
    const sampleGradients = [];
    for (const sample of shard.samples) {
      const gradient = computeSampleGradient(sample, weights);
      sampleGradients.push({
        id: sample.id,
        sequencePosition: sample.sequencePosition,
        gradient
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
      sampleGradients
    };
  }
  function reduceGlobalGradient(workerResults, globalBatchSize) {
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
  function executePlannedBatch(batch, weights, learningRate) {
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
        sampleIds: shard.sampleIds
      })),
      workerResults,
      weightsBefore,
      gradient,
      delta,
      weightsAfter,
      lossBefore,
      lossAfter
    };
  }
  function serializableCheckpoint(iteratorCheckpoint, weights, committedStep) {
    return JSON.parse(JSON.stringify({
      ...iteratorCheckpoint,
      committedStep,
      weights: [...weights],
      optimizer: { type: "sgd", learningRate: DEFAULT_CONFIG.learningRate }
    }));
  }
  function createIterator(config, worldSize) {
    const samples = config.samples ?? makeDataset(config.datasetSize, config.seed);
    return new ReshardableIterator({
      samples,
      batchSize: config.batchSize,
      worldSize,
      seed: config.seed,
      cursor: 0
    });
  }
  function runUninterruptedTraining(overrides = {}) {
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
        runId: "baseline-4-workers",
        phase: "uninterrupted",
        status: "committed",
        checkpoint
      };
      steps.push(trace);
    }
    return {
      runId: "baseline",
      worldSize: config.worldSizeBefore,
      steps,
      finalWeights: weights,
      finalCheckpoint: steps.at(-1)?.checkpoint ?? null
    };
  }
  function runScaledRecoveryTraining(overrides = {}) {
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
        committedSteps.length + 1
      );
      checkpoint = nextCheckpoint;
      const trace = {
        ...executed,
        runId,
        phase,
        status,
        checkpoint: nextCheckpoint
      };
      committedSteps.push(trace);
      timeline.push(trace);
      return trace;
    };
    for (let index = 0; index < config.steps; index += 1) {
      if (index < config.failStep) {
        const batch2 = iterator.planBatch(config.worldSizeBefore);
        const executed2 = executePlannedBatch(batch2, weights, config.learningRate);
        commitTrace(batch2, executed2, "before-scale", "committed", "recovery-4-to-2");
        continue;
      }
      if (index === config.failStep) {
        const failedPlan = iterator.planBatch(config.worldSizeBefore);
        const completedShards = failedPlan.shards.slice(0, config.partialWorkers);
        const partialWorkerResults = completedShards.map(
          (shard) => computeShardResult(shard, weights)
        );
        const completedSampleIds = completedShards.flatMap((shard) => shard.sampleIds);
        const incompleteBatch = {
          committed: false,
          baseCursor: failedPlan.baseCursor,
          observedCursor: failedPlan.baseCursor + completedSampleIds.length,
          step: failedPlan.step,
          worldSize: config.worldSizeBefore,
          completedRanks: completedShards.map((shard) => shard.rank),
          completedSampleIds
        };
        const partialEvent = {
          kind: "partial-attempt",
          runId: "recovery-4-to-2",
          step: failedPlan.step,
          globalStepIndex: index,
          phase: "failed-half-batch",
          status: "discarded-uncommitted",
          worldSize: config.worldSizeBefore,
          completedRanks: incompleteBatch.completedRanks,
          completedSampleIds,
          allSampleIds: failedPlan.sampleIds,
          shards: failedPlan.shards.map((shard) => ({
            rank: shard.rank,
            sampleIds: shard.sampleIds,
            completed: shard.rank < config.partialWorkers
          })),
          workerResults: partialWorkerResults,
          note: "Only local shard gradients exist. No global optimizer step or checkpoint was committed."
        };
        partialAttempts.push(partialEvent);
        timeline.push(partialEvent);
        const recovery = resolveHalfBatchRecovery({
          checkpoint,
          incompleteBatch,
          newWorldSize: config.worldSizeAfter,
          samples
        });
        const { iterator: recoveredIterator, ...decisionSummary } = recovery;
        recoveryDecision = decisionSummary;
        iterator = recoveredIterator;
        const replayBatch = iterator.planBatch(config.worldSizeAfter);
        const replayExecuted = executePlannedBatch(replayBatch, weights, config.learningRate);
        commitTrace(
          replayBatch,
          replayExecuted,
          "replay-after-scale",
          "committed-replayed-full-batch",
          "recovery-4-to-2"
        );
        continue;
      }
      const batch = iterator.planBatch(config.worldSizeAfter);
      const executed = executePlannedBatch(batch, weights, config.learningRate);
      commitTrace(batch, executed, "after-scale", "committed", "recovery-4-to-2");
    }
    return {
      runId: "recovery",
      worldSizeBefore: config.worldSizeBefore,
      worldSizeAfter: config.worldSizeAfter,
      steps: committedSteps,
      committedSteps,
      timeline,
      partialAttempts,
      recoveryDecision,
      initialCheckpoint,
      finalWeights: weights,
      finalCheckpoint: committedSteps.at(-1)?.checkpoint ?? checkpoint
    };
  }
  function compareTrainingRuns(baseline, recovery) {
    const stepComparisons = baseline.steps.map((baselineStep, index) => {
      const recoveryStep = recovery.committedSteps[index];
      const sampleOrderExact = arraysEqual(baselineStep.sampleIds, recoveryStep.sampleIds) && arraysEqual(baselineStep.sequencePositions, recoveryStep.sequencePositions);
      const gradientExact = bitwiseArraysEqual(baselineStep.gradient, recoveryStep.gradient);
      const deltaExact = bitwiseArraysEqual(baselineStep.delta, recoveryStep.delta);
      const weightsExact = bitwiseArraysEqual(
        baselineStep.weightsAfter,
        recoveryStep.weightsAfter
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
          recoveryStep.weightsAfter
        ),
        baselineGradient: baselineStep.gradient,
        recoveryGradient: recoveryStep.gradient,
        baselineWeightsAfter: baselineStep.weightsAfter,
        recoveryWeightsAfter: recoveryStep.weightsAfter,
        baselineLoss: baselineStep.lossAfter,
        recoveryLoss: recoveryStep.lossAfter
      };
    });
    const baselineOrder = baseline.steps.flatMap((step) => step.sampleIds);
    const recoveryOrder = recovery.committedSteps.flatMap((step) => step.sampleIds);
    const flattenedOrderExact = arraysEqual(baselineOrder, recoveryOrder);
    const finalWeightsExact = bitwiseArraysEqual(baseline.finalWeights, recovery.finalWeights);
    const maxGradientDiff = Math.max(...stepComparisons.map((item) => item.gradientMaxDiff));
    const maxWeightDiff = Math.max(...stepComparisons.map((item) => item.weightsMaxDiff));
    const exact = flattenedOrderExact && stepComparisons.every((item) => item.sampleOrderExact && item.gradientExact && item.deltaExact && item.weightsExact) && finalWeightsExact;
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
        recovery: recovery.finalWeights
      },
      steps: stepComparisons
    };
  }
  function runComparison(overrides = {}) {
    const config = { ...DEFAULT_CONFIG, ...overrides };
    const samples = config.samples ?? makeDataset(config.datasetSize, config.seed);
    const resolvedConfig = { ...config, samples };
    const baseline = runUninterruptedTraining(resolvedConfig);
    const recovery = runScaledRecoveryTraining(resolvedConfig);
    const comparison = compareTrainingRuns(baseline, recovery);
    return { config: resolvedConfig, baseline, recovery, comparison };
  }

  // web/app.js
  var elements = {
    failStep: document.getElementById("fail-step"),
    partialWorkers: document.getElementById("partial-workers"),
    runButton: document.getElementById("run-comparison"),
    verdict: document.getElementById("verdict-banner"),
    kpiSteps: document.getElementById("kpi-steps"),
    kpiStepsSub: document.getElementById("kpi-steps-sub"),
    kpiSamples: document.getElementById("kpi-samples"),
    kpiSamplesSub: document.getElementById("kpi-samples-sub"),
    kpiGradient: document.getElementById("kpi-gradient"),
    kpiWeights: document.getElementById("kpi-weights"),
    kpiWeightsSub: document.getElementById("kpi-weights-sub"),
    finalBadge: document.getElementById("final-weights-badge"),
    finalWeightsBody: document.getElementById("final-weights-body"),
    observedCursor: document.getElementById("observed-cursor"),
    resumeCursor: document.getElementById("resume-cursor"),
    recoverySummary: document.getElementById("recovery-summary"),
    checkpointJson: document.getElementById("checkpoint-json"),
    eventLog: document.getElementById("event-log"),
    flattenedOrder: document.getElementById("flattened-order"),
    stepList: document.getElementById("step-list")
  };
  var parameterNames = ["w\u2081 \xB7 x\u2081", "w\u2082 \xB7 x\u2082", "b \xB7 bias"];
  function formatFixed(value, digits = 12) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "\u2014";
    return value.toFixed(digits);
  }
  function formatDiff(value) {
    if (value === 0) return "0";
    if (!Number.isFinite(value)) return "\u221E";
    if (value < 1e-12) return "< 1e-12";
    return value.toExponential(3);
  }
  function workerBadge(rank) {
    return `<span class="worker-badge w${rank}" title="Worker ${rank}">W${rank}</span>`;
  }
  function sampleChips(ids = []) {
    return `<div class="sample-chips">${ids.map((id) => `<span class="sample-chip">${id}</span>`).join("")}</div>`;
  }
  function renderShardRows(shards, partialEvent = null) {
    return shards.map((shard) => {
      const partialShard = partialEvent?.shards?.find((item) => item.rank === shard.rank);
      let rowClass = "worker-row";
      let status = "";
      if (partialEvent && partialShard?.completed) {
        rowClass += " partial";
        status = '<span class="badge warning">\u5C40\u90E8\u5B8C\u6210 \xB7 \u68AF\u5EA6\u4E22\u5F03</span>';
      } else if (partialEvent && partialShard && !partialShard.completed) {
        rowClass += " skipped";
        status = '<span class="badge">\u5D29\u6E83\u524D\u672A\u8FD0\u884C</span>';
      }
      return `
        <div class="${rowClass}">
          <div class="worker-tag">${workerBadge(shard.rank)}</div>
          <div>
            <div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-bottom:5px;">
              <span class="step-meta">${shard.sampleIds.length} samples</span>
              ${status}
            </div>
            ${sampleChips(shard.sampleIds)}
          </div>
        </div>
      `;
    }).join("");
  }
  function renderVerdict(result) {
    const { exact, totalSteps, maxGradientDiff, maxWeightDiff } = result.comparison;
    elements.verdict.classList.toggle("bad", !exact);
    if (exact) {
      elements.verdict.innerHTML = `
      <div class="verdict-icon" aria-hidden="true">\u2713</div>
      <div>
        <h2>\u6062\u590D\u8F68\u8FF9\u4E0E\u4E0D\u4E2D\u65AD\u8BAD\u7EC3\u5B8C\u5168\u7B49\u4EF7</h2>
        <p>
          ${totalSteps} \u4E2A\u5168\u5C40\u66F4\u65B0\u6B65\u7684\u6837\u672C ID/\u4F4D\u7F6E\u987A\u5E8F\u3001\u5F52\u7EA6\u68AF\u5EA6\u3001SGD \u589E\u91CF\u548C\u63D0\u4EA4\u540E\u6743\u91CD\u5168\u90E8\u9010\u4F4D\u4E00\u81F4\uFF1B
          \u6700\u5927\u68AF\u5EA6\u5DEE\u4E0E\u6700\u5927\u6743\u91CD\u5DEE\u5747\u4E3A ${formatDiff(maxGradientDiff)} / ${formatDiff(maxWeightDiff)}\u3002
          \u534A\u6279\u5C40\u90E8\u8BA1\u7B97\u88AB\u91CD\u653E\uFF0C\u4F46\u6CA1\u6709\u4EA7\u751F\u7B2C\u4E8C\u6B21\u4F18\u5316\u5668\u66F4\u65B0\u3002
        </p>
      </div>
    `;
    } else {
      elements.verdict.innerHTML = `
      <div class="verdict-icon" aria-hidden="true">!</div>
      <div>
        <h2>\u68C0\u6D4B\u5230\u4E0D\u4E00\u81F4</h2>
        <p>\u8BF7\u68C0\u67E5\u6837\u672C\u6E38\u6807\u3001\u5168\u5C40\u5F52\u7EA6\u987A\u5E8F\u6216\u68C0\u67E5\u70B9\u5185\u5BB9\u3002\u5F53\u524D\u6700\u5927\u68AF\u5EA6\u5DEE ${maxGradientDiff}\uFF0C\u6700\u5927\u6743\u91CD\u5DEE ${maxWeightDiff}\u3002</p>
      </div>
    `;
    }
  }
  function renderKpis(result) {
    const { comparison, config } = result;
    const sampleCount = comparison.flattenedOrder.baseline.length;
    elements.kpiSteps.textContent = String(comparison.totalSteps);
    elements.kpiStepsSub.textContent = `\u6BCF\u6B65\u5168\u5C40 batch=${config.batchSize}`;
    elements.kpiSamples.textContent = comparison.flattenedOrderExact ? `${sampleCount}/${sampleCount}` : "\u4E0D\u4E00\u81F4";
    elements.kpiSamples.className = `kpi-value ${comparison.flattenedOrderExact ? "good-text" : "warning-text"}`;
    elements.kpiSamplesSub.textContent = comparison.flattenedOrderExact ? "\u9010\u6B65\u4E0E\u6241\u5E73\u5316\u5168\u5C40\u987A\u5E8F\u76F8\u540C" : "\u6241\u5E73\u5316\u987A\u5E8F\u5B58\u5728\u5DEE\u5F02";
    elements.kpiGradient.textContent = formatDiff(comparison.maxGradientDiff);
    elements.kpiGradient.className = `kpi-value mono ${comparison.maxGradientDiff === 0 ? "good-text" : "warning-text"}`;
    elements.kpiWeights.textContent = formatDiff(comparison.maxWeightDiff);
    elements.kpiWeights.className = `kpi-value mono ${comparison.maxWeightDiff === 0 ? "good-text" : "warning-text"}`;
    elements.kpiWeightsSub.textContent = comparison.finalWeightsExact ? "\u6700\u7EC8\u6743\u91CD\u9010\u4F4D\u76F8\u540C" : "\u6700\u7EC8\u6743\u91CD\u5B58\u5728\u5DEE\u5F02";
  }
  function renderFinalWeights(result) {
    const { finalWeights } = result.comparison;
    elements.finalBadge.textContent = result.comparison.finalWeightsExact ? "\u2713 \u6700\u7EC8\u6743\u91CD\u9010\u4F4D\u4E00\u81F4" : "\u4E0D\u4E00\u81F4";
    elements.finalBadge.className = `badge ${result.comparison.finalWeightsExact ? "good" : "warning"}`;
    elements.finalWeightsBody.innerHTML = parameterNames.map((name, index) => {
      const baseline = finalWeights.baseline[index];
      const recovery = finalWeights.recovery[index];
      const diff = Math.abs(baseline - recovery);
      return `
        <tr>
          <td><strong>${name}</strong></td>
          <td class="num" title="${String(baseline)}">${formatFixed(baseline)}</td>
          <td class="num" title="${String(recovery)}">${formatFixed(recovery)}</td>
          <td class="num ${diff === 0 ? "diff-zero" : "warning-text"}">${formatDiff(diff)}</td>
        </tr>
      `;
    }).join("");
  }
  function renderRecovery(result) {
    const { recovery, config } = result;
    const decision = recovery.recoveryDecision;
    const crashCheckpoint = config.failStep === 0 ? recovery.initialCheckpoint : recovery.committedSteps[config.failStep - 1].checkpoint;
    elements.observedCursor.textContent = String(decision.rawCursor);
    elements.resumeCursor.textContent = String(decision.resumeCursor);
    elements.recoverySummary.innerHTML = `
    <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px;">
      <span class="badge warning">\u7B56\u7565\uFF1A\u56DE\u6EDA\u672A\u63D0\u4EA4\u6279\u6B21\u5E76\u91CD\u653E</span>
      <span class="badge">\u89C2\u5BDF\u6E38\u6807 ${decision.rawCursor}</span>
      <span class="badge good">\u6062\u590D\u6E38\u6807 ${decision.resumeCursor}</span>
      <span class="badge warning">\u56DE\u6EDA ${decision.rewoundSamples} \u4E2A\u5DF2\u5904\u7406\u6837\u672C\u4F4D\u7F6E</span>
      <span class="badge">\u65B0\u62D3\u6251 ${config.worldSizeAfter} workers \xD7 ${config.batchSize / config.worldSizeAfter} samples</span>
    </div>
    <p style="margin:0 0 8px; color:var(--text-secondary); font-size:14px;">
      \u9700\u91CD\u7B97\u7684\u5DF2\u5B8C\u6210 worker \u6837\u672C\uFF1A
      <code>${decision.recomputeSampleIds.join(", ") || "\uFF08\u65E0\uFF09"}</code>
    </p>
    <p style="margin:0; color:var(--text-secondary); font-size:14px;">
      \u8FD9\u4E9B\u6837\u672C\u5728\u5D29\u6E83\u524D\u5B8C\u6210\u4E86\u5C40\u90E8\u524D\u5411/\u53CD\u5411\u8BA1\u7B97\uFF0C\u4F46\u5BF9\u5E94\u5168\u5C40\u68AF\u5EA6\u548C\u4F18\u5316\u5668\u63D0\u4EA4\u4E0D\u5B58\u5728\uFF1B\u56E0\u6B64\u91CD\u653E\u4E0D\u4F1A\u9020\u6210\u7B2C\u4E8C\u6B21\u6743\u91CD\u66F4\u65B0\u3002
    </p>
  `;
    elements.checkpointJson.textContent = JSON.stringify(crashCheckpoint, null, 2);
  }
  function renderEventLog(result) {
    const { config, recovery, comparison } = result;
    const partial = recovery.partialAttempts[0];
    const beforeCount = config.failStep;
    const afterCount = config.steps - config.failStep - 1;
    const flattenedLength = comparison.flattenedOrder.baseline.length;
    const events = [
      {
        tone: "",
        html: `<strong>\u521D\u59CB\u5316\uFF1A</strong>\u56FA\u5B9A\u79CD\u5B50 ${config.seed} \u751F\u6210 ${config.datasetSize} \u4E2A\u6837\u672C\u7684\u786E\u5B9A\u6027 epoch \u6392\u5217\uFF1B\u5168\u5C40 batch=${config.batchSize}\uFF0C\u521D\u59CB\u62D3\u6251 ${config.worldSizeBefore} workers\uFF0C\u6BCF worker ${config.batchSize / config.worldSizeBefore} \u4E2A\u6837\u672C\u3002`
      },
      {
        tone: "good",
        html: `<strong>\u5DF2\u63D0\u4EA4\u524D\u5E8F\u6B65\uFF1A</strong>\u7B2C 0\u2013${config.failStep - 1} \u6B65\u5171 ${beforeCount} \u4E2A\u5168\u5C40\u6279\u6B21\u5B8C\u6210\u5F52\u7EA6\u3001SGD \u66F4\u65B0\u548C\u68C0\u67E5\u70B9\u63D0\u4EA4\u3002\u68C0\u67E5\u70B9\u53EA\u63A8\u8FDB\u5230\u6E38\u6807 ${config.failStep * config.batchSize}\u3002`
      },
      {
        tone: "warn",
        html: `<strong>\u6A21\u62DF\u7F29\u7EC4/\u6545\u969C\uFF1A</strong>\u7B2C ${config.failStep} \u6B65\u5728 ranks ${partial.completedRanks.join(", ")} \u5B8C\u6210\u540E\u4E2D\u65AD\uFF1B\u5DF2\u5904\u7406\u6837\u672C\u4E3A <code>${partial.completedSampleIds.join(", ")}</code>\u3002\u5C40\u90E8\u68AF\u5EA6\u88AB\u9694\u79BB\uFF0C\u672A\u5F62\u6210\u5168\u5C40\u66F4\u65B0\u3002`
      },
      {
        tone: "warn",
        html: `<strong>\u5E94\u7528\u534A\u6279\u7B56\u7565\uFF1A</strong>\u89C2\u5BDF\u5230\u7684\u4E34\u65F6\u4F4D\u7F6E\u4E3A ${recovery.recoveryDecision.rawCursor}\uFF0C\u4F46\u5B83\u4E0D\u662F\u63D0\u4EA4\u4F4D\u7F6E\uFF1B\u8FED\u4EE3\u5668\u56DE\u6EDA\u5230\u5168\u5C40\u6279\u8FB9\u754C ${recovery.recoveryDecision.resumeCursor}\uFF0C\u5E76\u6309 worldSize=${config.worldSizeAfter} \u91CD\u65B0\u5206\u7247\u3002`
      },
      {
        tone: "good",
        html: `<strong>\u91CD\u653E\u7B2C ${config.failStep} \u6B65\uFF1A</strong>\u540C\u4E00\u6279\u6837\u672C ${comparison.steps[config.failStep].baselineSampleIds.join(", ")} \u88AB\u91CD\u65B0\u8BA1\u7B97\uFF0C\u89C4\u8303\u987A\u5E8F\u5F52\u7EA6\u5F97\u5230\u4E0E\u4E0D\u4E2D\u65AD\u8F68\u8FF9\u76F8\u540C\u7684\u68AF\u5EA6\u548C\u6743\u91CD\u3002`
      },
      {
        tone: "",
        html: `<strong>\u7F29\u7EC4\u540E\u7EE7\u7EED\uFF1A</strong>\u5269\u4F59 ${afterCount} \u6B65\u5747\u4F7F\u7528 ${config.worldSizeAfter} workers\u3001\u6BCF worker ${config.batchSize / config.worldSizeAfter} samples\uFF1B\u5168\u5C40 batch \u4ECD\u4E3A ${config.batchSize}\u3002`
      },
      {
        tone: "good",
        html: `<strong>\u6700\u7EC8\u6821\u9A8C\uFF1A</strong>${flattenedLength} \u4E2A\u6241\u5E73\u5316\u6837\u672C\u4F4D\u7F6E\u4E00\u81F4\uFF0C${comparison.totalSteps} \u4E2A\u68AF\u5EA6\u66F4\u65B0\u4E00\u81F4\uFF0C\u6700\u7EC8\u6743\u91CD\u9010\u4F4D${comparison.finalWeightsExact ? "\u76F8\u540C" : "\u4E0D\u540C"}\u3002`
      }
    ];
    elements.eventLog.innerHTML = events.map((event) => `<li class="${event.tone}">${event.html}</li>`).join("");
  }
  function renderFlattenedOrder(result) {
    elements.flattenedOrder.textContent = result.comparison.flattenedOrder.baseline.join("  \u2192  ");
  }
  function numericLines(step) {
    return parameterNames.map((name, index) => `
      <div class="line">
        <span>${name}</span>
        <span title="baseline / recovery">
          g ${formatFixed(step.baselineGradient[index], 9)} / ${formatFixed(step.recoveryGradient[index], 9)}
        </span>
      </div>
      <div class="line">
        <span>w after</span>
        <span>${formatFixed(step.baselineWeightsAfter[index], 12)} / ${formatFixed(step.recoveryWeightsAfter[index], 12)}</span>
      </div>
    `).join("");
  }
  function renderSteps(result) {
    const partialByIndex = new Map(
      result.recovery.partialAttempts.map((event) => [event.globalStepIndex, event])
    );
    elements.stepList.innerHTML = result.comparison.steps.map((step) => {
      const partialEvent = partialByIndex.get(step.index) ?? null;
      const baselineStatus = partialEvent ? '<span class="badge warning">\u5728\u6B64\u6CE8\u5165\u534A\u6279\u6545\u969C</span>' : '<span class="badge good">\u5DF2\u63D0\u4EA4</span>';
      return `
        <article class="panel step-card" data-testid="step-card" data-step="${step.index}">
          <header class="step-head">
            <h3 class="step-title">
              \u5168\u5C40\u6B65 ${step.index + 1}
              <span class="step-meta">zero-based step=${step.step} \xB7 epoch=${step.epoch} \xB7 cursor ${step.index * result.config.batchSize}</span>
            </h3>
            <div class="step-checks">
              <span class="badge ${step.sampleOrderExact ? "good" : "warning"}">
                ${step.sampleOrderExact ? "\u2713" : "!"} \u6837\u672C\u987A\u5E8F\u4E00\u81F4
              </span>
              <span class="badge ${step.gradientExact ? "good" : "warning"}" title="\u6700\u5927\u5DEE ${step.gradientMaxDiff}">
                ${step.gradientExact ? "\u2713 \u68AF\u5EA6\u9010\u4F4D\u4E00\u81F4" : `\u68AF\u5EA6\u5DEE ${formatDiff(step.gradientMaxDiff)}`}
              </span>
              <span class="badge ${step.weightsExact ? "good" : "warning"}" title="\u6700\u5927\u5DEE ${step.weightsMaxDiff}">
                ${step.weightsExact ? "\u2713 \u6743\u91CD\u9010\u4F4D\u4E00\u81F4" : `\u6743\u91CD\u5DEE ${formatDiff(step.weightsMaxDiff)}`}
              </span>
            </div>
          </header>

          <div class="lanes">
            <section class="lane baseline">
              <div class="lane-title">
                <span>\u4E0D\u4E2D\u65AD\u8BAD\u7EC3</span>
                <span>${baselineStatus}</span>
              </div>
              ${partialEvent ? `
                <p class="partial-callout">
                  \u6A21\u62DF ${partialEvent.worldSize} workers \u4E2D ${partialEvent.completedRanks.length} \u4E2A\u5206\u7247\u5B8C\u6210\u540E\u5D29\u6E83\uFF1B
                  \u5DF2\u5B8C\u6210\u5C40\u90E8\u68AF\u5EA6\u4E22\u5F03\uFF0C\u4E0D\u63A8\u8FDB\u68C0\u67E5\u70B9\u3002
                </p>
              ` : ""}
              ${renderShardRows(step.baselineShards, partialEvent)}
            </section>

            <section class="lane recovery">
              <div class="lane-title">
                <span>\u7F29\u7EC4\u6062\u590D\u8F68\u8FF9</span>
                <span class="badge good">
                  ${step.index < result.config.failStep ? `\u6545\u969C\u524D \xB7 ${step.recoveryWorldSize} workers` : step.index === result.config.failStep ? `\u56DE\u6EDA\u540E\u6574\u6279\u91CD\u653E \xB7 ${step.recoveryWorldSize} workers` : `\u6062\u590D\u540E \xB7 ${step.recoveryWorldSize} workers`}
                </span>
              </div>
              ${renderShardRows(step.recoveryShards)}
            </section>
          </div>

          <details class="numeric-details">
            <summary>\u67E5\u770B\u68AF\u5EA6\u3001SGD \u589E\u91CF\u4E0E\u6743\u91CD\uFF08\u5DE6/\u53F3\u5206\u522B\u4E3A 4-worker \u57FA\u7EBF\u4E0E 2-worker \u6062\u590D\uFF09</summary>
            <div class="numeric-grid">
              <div class="numeric-box">
                <h4>\u6570\u503C\u5BF9\u6BD4</h4>
                ${numericLines(step)}
              </div>
              <div class="numeric-box">
                <h4>\u672C\u6B65\u6821\u9A8C</h4>
                <div class="line"><span>gradient max abs diff</span><span class="${step.gradientMaxDiff === 0 ? "diff-zero" : ""}">${step.gradientMaxDiff}</span></div>
                <div class="line"><span>weight max abs diff</span><span class="${step.weightsMaxDiff === 0 ? "diff-zero" : ""}">${step.weightsMaxDiff}</span></div>
                <div class="line"><span>baseline loss after</span><span>${formatFixed(step.baselineLoss, 9)}</span></div>
                <div class="line"><span>recovery loss after</span><span>${formatFixed(step.recoveryLoss, 9)}</span></div>
                <div class="line"><span>\u0394w exact</span><span class="${step.deltaExact ? "diff-zero" : "warning-text"}">${step.deltaExact}</span></div>
              </div>
            </div>
          </details>
        </article>
      `;
    }).join("");
  }
  function render() {
    const failStep = Number(elements.failStep.value);
    const partialWorkers = Number(elements.partialWorkers.value);
    const result = runComparison({ failStep, partialWorkers });
    renderVerdict(result);
    renderKpis(result);
    renderFinalWeights(result);
    renderRecovery(result);
    renderEventLog(result);
    renderFlattenedOrder(result);
    renderSteps(result);
    window.__LAST_COMPARISON__ = result;
  }
  elements.runButton.addEventListener("click", render);
  elements.failStep.addEventListener("change", render);
  elements.partialWorkers.addEventListener("change", render);
  render();
})();
