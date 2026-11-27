// Reshardable global-batch iterator.
//
// The iterator checkpoints one number that matters for data order: the flat
// count of globally committed samples. Worker ranks are deliberately NOT part
// of the checkpoint. After restoring with a new world size, the same global
// batch is repartitioned into contiguous worker shards. This keeps the
// flattened sample order independent of the number of simulated workers.

export const ITERATOR_VERSION = 1;
export const HALF_BATCH_POLICY = 'rollback-and-replay-uncommitted-batch';

export function hashSeed(value) {
  const text = String(value);
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function nextRandom() {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Deterministic Fisher--Yates permutation for an epoch. Epoch is mixed into
 * the PRNG seed, so epoch 0 and epoch 1 have different-but-stable orders.
 */
export function globalOrder(datasetSize, epoch, seed) {
  const order = Array.from({ length: datasetSize }, (_, index) => index);
  const random = mulberry32(hashSeed(`reshard-iterator:${seed}:epoch:${epoch}`));
  for (let i = datasetSize - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

export function partitionSamples(samples, worldSize) {
  if (!Number.isInteger(worldSize) || worldSize <= 0) {
    throw new RangeError('worldSize must be a positive integer');
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
      samples: shardSamples,
    });
    offset += size;
  }

  return shards;
}

/**
 * Align an observed/unsafe cursor back to the start of its global batch.
 * Alignment is performed inside its epoch because cursor is a flat sample
 * position and dataset boundaries need not coincide with batch boundaries in
 * general. This demo requires divisibility, but the helper remains explicit.
 */
export function alignToGlobalBatchStart(cursor, batchSize, datasetSize) {
  if (!Number.isInteger(cursor) || cursor < 0) {
    throw new RangeError('cursor must be a non-negative integer');
  }
  const epoch = Math.floor(cursor / datasetSize);
  const offset = cursor - epoch * datasetSize;
  return epoch * datasetSize + Math.floor(offset / batchSize) * batchSize;
}

export class ReshardableIterator {
  constructor(options = {}) {
    const {
      samples,
      batchSize = 8,
      worldSize = 4,
      seed = 2611914,
      cursor = 0,
      dropLast = true,
    } = options;

    if (!Array.isArray(samples) || samples.length === 0) {
      throw new TypeError('samples must be a non-empty array');
    }
    if (!Number.isInteger(batchSize) || batchSize <= 0) {
      throw new RangeError('batchSize must be a positive integer');
    }
    if (!Number.isInteger(worldSize) || worldSize <= 0) {
      throw new RangeError('worldSize must be a positive integer');
    }
    if (!Number.isInteger(cursor) || cursor < 0) {
      throw new RangeError('cursor must be a non-negative integer');
    }
    if (dropLast && samples.length % batchSize !== 0) {
      throw new Error(
        `Demo iterator requires datasetSize (${samples.length}) divisible by batchSize (${batchSize}) when dropLast=true`,
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
      throw new RangeError('worldSize must be a positive integer');
    }

    let cursor = this.cursor;
    let epoch = Math.floor(cursor / this.datasetSize);
    let offset = cursor % this.datasetSize;

    // With dropLast, a cursor left at a non-full tail skips to next epoch.
    // The demo's divisibility validation means this branch is defensive.
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
        datasetIndex,
      };
    });

    const epochBatch = Math.floor(offset / this.batchSize);
    const batchesPerEpoch = this.datasetSize / this.batchSize;
    const step = epoch * batchesPerEpoch + epochBatch;
    const shards = partitionSamples(orderedSamples, worldSize);

    return {
      kind: 'global-batch-plan',
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
      shards,
    };
  }

  commitBatch(batch) {
    if (!batch || batch.kind !== 'global-batch-plan') {
      throw new TypeError('commitBatch expects a batch planned by planBatch()');
    }
    if (batch.baseCursor !== this.cursor) {
      throw new Error(
        `Batch base cursor ${batch.baseCursor} does not match iterator cursor ${this.cursor}; refusing out-of-order commit`,
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
      ...extra,
    };
  }

  static fromCheckpoint(checkpoint, options = {}) {
    if (!checkpoint || checkpoint.version !== ITERATOR_VERSION) {
      throw new Error('Unsupported or missing iterator checkpoint version');
    }
    return new ReshardableIterator({
      samples: options.samples ?? checkpoint.samples,
      batchSize: checkpoint.batchSize,
      seed: checkpoint.seed,
      dropLast: checkpoint.dropLast,
      cursor: options.cursor ?? checkpoint.cursor,
      worldSize: options.worldSize ?? checkpoint.worldSize ?? 4,
    });
  }
}

/**
 * Decide where to resume after a failure.
 *
 * A global optimizer step is atomic. If some workers finished their shards but
 * the global gradient was not committed, their partial gradients are discarded
 * and every sample in that global batch is replayed under the new sharding.
 * Their numeric work may be repeated, but their effect on model state is
 * exactly once because the partial update never entered the weights.
 */
export function resolveHalfBatchRecovery({
  checkpoint,
  incompleteBatch = null,
  observedCursor = null,
  newWorldSize,
  samples,
}) {
  if (!checkpoint) throw new TypeError('checkpoint is required');

  let cursor = checkpoint.cursor;
  let batchBaseCursor = null;
  let recomputeSampleIds = [];
  let discardedCompletedRanks = [];
  let source = 'durable-checkpoint';

  if (incompleteBatch && incompleteBatch.committed !== true) {
    batchBaseCursor = incompleteBatch.baseCursor ?? checkpoint.cursor;
    // observedCursor may record how many samples had been processed before the
    // crash. It is evidence for the UI/audit trail, never a commit position.
    cursor = incompleteBatch.observedCursor
      ?? incompleteBatch.baseCursor
      ?? checkpoint.cursor;
    recomputeSampleIds = incompleteBatch.completedSampleIds ?? [];
    discardedCompletedRanks = incompleteBatch.completedRanks ?? [];
    source = 'incomplete-batch-rollback';
  } else if (Number.isInteger(observedCursor)) {
    cursor = observedCursor;
    source = 'observed-half-cursor';
  }

  const alignedCursor = batchBaseCursor ?? alignToGlobalBatchStart(
    cursor,
    checkpoint.batchSize,
    checkpoint.datasetSize,
  );

  const rewoundSamples = Math.max(0, cursor - alignedCursor);
  const iterator = new ReshardableIterator({
    samples: samples ?? checkpoint.samples,
    batchSize: checkpoint.batchSize,
    seed: checkpoint.seed,
    dropLast: checkpoint.dropLast,
    cursor: alignedCursor,
    worldSize: newWorldSize,
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
    iterator,
  };
}

export function arraysEqual(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a.length === b.length
    && a.every((value, index) => value === b[index]);
}

export function maxNumericDifference(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
    return Number.POSITIVE_INFINITY;
  }
  return a.reduce((max, value, index) => {
    const diff = Math.abs(value - b[index]);
    return Number.isFinite(diff) ? Math.max(max, diff) : Number.POSITIVE_INFINITY;
  }, 0);
}

export function bitwiseArraysEqual(a, b) {
  return arraysEqual(a, b) && a.every((value, index) => Object.is(value, b[index]));
}
