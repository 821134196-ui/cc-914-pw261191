import { runComparison } from '../src/training.js';

const elements = {
  failStep: document.getElementById('fail-step'),
  partialWorkers: document.getElementById('partial-workers'),
  runButton: document.getElementById('run-comparison'),
  verdict: document.getElementById('verdict-banner'),
  kpiSteps: document.getElementById('kpi-steps'),
  kpiStepsSub: document.getElementById('kpi-steps-sub'),
  kpiSamples: document.getElementById('kpi-samples'),
  kpiSamplesSub: document.getElementById('kpi-samples-sub'),
  kpiGradient: document.getElementById('kpi-gradient'),
  kpiWeights: document.getElementById('kpi-weights'),
  kpiWeightsSub: document.getElementById('kpi-weights-sub'),
  finalBadge: document.getElementById('final-weights-badge'),
  finalWeightsBody: document.getElementById('final-weights-body'),
  observedCursor: document.getElementById('observed-cursor'),
  resumeCursor: document.getElementById('resume-cursor'),
  recoverySummary: document.getElementById('recovery-summary'),
  checkpointJson: document.getElementById('checkpoint-json'),
  eventLog: document.getElementById('event-log'),
  flattenedOrder: document.getElementById('flattened-order'),
  stepList: document.getElementById('step-list'),
};

const parameterNames = ['w₁ · x₁', 'w₂ · x₂', 'b · bias'];

function formatFixed(value, digits = 12) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

function formatDiff(value) {
  if (value === 0) return '0';
  if (!Number.isFinite(value)) return '∞';
  if (value < 1e-12) return '< 1e-12';
  return value.toExponential(3);
}

function workerBadge(rank) {
  return `<span class="worker-badge w${rank}" title="Worker ${rank}">W${rank}</span>`;
}

function sampleChips(ids = []) {
  return `<div class="sample-chips">${ids
    .map((id) => `<span class="sample-chip">${id}</span>`)
    .join('')}</div>`;
}

function renderShardRows(shards, partialEvent = null) {
  return shards
    .map((shard) => {
      const partialShard = partialEvent?.shards?.find((item) => item.rank === shard.rank);
      let rowClass = 'worker-row';
      let status = '';

      if (partialEvent && partialShard?.completed) {
        rowClass += ' partial';
        status = '<span class="badge warning">局部完成 · 梯度丢弃</span>';
      } else if (partialEvent && partialShard && !partialShard.completed) {
        rowClass += ' skipped';
        status = '<span class="badge">崩溃前未运行</span>';
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
    })
    .join('');
}

function renderVerdict(result) {
  const { exact, totalSteps, maxGradientDiff, maxWeightDiff } = result.comparison;
  elements.verdict.classList.toggle('bad', !exact);

  if (exact) {
    elements.verdict.innerHTML = `
      <div class="verdict-icon" aria-hidden="true">✓</div>
      <div>
        <h2>恢复轨迹与不中断训练完全等价</h2>
        <p>
          ${totalSteps} 个全局更新步的样本 ID/位置顺序、归约梯度、SGD 增量和提交后权重全部逐位一致；
          最大梯度差与最大权重差均为 ${formatDiff(maxGradientDiff)} / ${formatDiff(maxWeightDiff)}。
          半批局部计算被重放，但没有产生第二次优化器更新。
        </p>
      </div>
    `;
  } else {
    elements.verdict.innerHTML = `
      <div class="verdict-icon" aria-hidden="true">!</div>
      <div>
        <h2>检测到不一致</h2>
        <p>请检查样本游标、全局归约顺序或检查点内容。当前最大梯度差 ${maxGradientDiff}，最大权重差 ${maxWeightDiff}。</p>
      </div>
    `;
  }
}

function renderKpis(result) {
  const { comparison, config } = result;
  const sampleCount = comparison.flattenedOrder.baseline.length;
  elements.kpiSteps.textContent = String(comparison.totalSteps);
  elements.kpiStepsSub.textContent = `每步全局 batch=${config.batchSize}`;

  elements.kpiSamples.textContent = comparison.flattenedOrderExact ? `${sampleCount}/${sampleCount}` : '不一致';
  elements.kpiSamples.className = `kpi-value ${comparison.flattenedOrderExact ? 'good-text' : 'warning-text'}`;
  elements.kpiSamplesSub.textContent = comparison.flattenedOrderExact
    ? '逐步与扁平化全局顺序相同'
    : '扁平化顺序存在差异';

  elements.kpiGradient.textContent = formatDiff(comparison.maxGradientDiff);
  elements.kpiGradient.className = `kpi-value mono ${comparison.maxGradientDiff === 0 ? 'good-text' : 'warning-text'}`;

  elements.kpiWeights.textContent = formatDiff(comparison.maxWeightDiff);
  elements.kpiWeights.className = `kpi-value mono ${comparison.maxWeightDiff === 0 ? 'good-text' : 'warning-text'}`;
  elements.kpiWeightsSub.textContent = comparison.finalWeightsExact ? '最终权重逐位相同' : '最终权重存在差异';
}

function renderFinalWeights(result) {
  const { finalWeights } = result.comparison;
  elements.finalBadge.textContent = result.comparison.finalWeightsExact ? '✓ 最终权重逐位一致' : '不一致';
  elements.finalBadge.className = `badge ${result.comparison.finalWeightsExact ? 'good' : 'warning'}`;

  elements.finalWeightsBody.innerHTML = parameterNames
    .map((name, index) => {
      const baseline = finalWeights.baseline[index];
      const recovery = finalWeights.recovery[index];
      const diff = Math.abs(baseline - recovery);
      return `
        <tr>
          <td><strong>${name}</strong></td>
          <td class="num" title="${String(baseline)}">${formatFixed(baseline)}</td>
          <td class="num" title="${String(recovery)}">${formatFixed(recovery)}</td>
          <td class="num ${diff === 0 ? 'diff-zero' : 'warning-text'}">${formatDiff(diff)}</td>
        </tr>
      `;
    })
    .join('');
}

function renderRecovery(result) {
  const { recovery, config } = result;
  const decision = recovery.recoveryDecision;
  const crashCheckpoint = config.failStep === 0
    ? recovery.initialCheckpoint
    : recovery.committedSteps[config.failStep - 1].checkpoint;

  elements.observedCursor.textContent = String(decision.rawCursor);
  elements.resumeCursor.textContent = String(decision.resumeCursor);

  elements.recoverySummary.innerHTML = `
    <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px;">
      <span class="badge warning">策略：回滚未提交批次并重放</span>
      <span class="badge">观察游标 ${decision.rawCursor}</span>
      <span class="badge good">恢复游标 ${decision.resumeCursor}</span>
      <span class="badge warning">回滚 ${decision.rewoundSamples} 个已处理样本位置</span>
      <span class="badge">新拓扑 ${config.worldSizeAfter} workers × ${config.batchSize / config.worldSizeAfter} samples</span>
    </div>
    <p style="margin:0 0 8px; color:var(--text-secondary); font-size:14px;">
      需重算的已完成 worker 样本：
      <code>${decision.recomputeSampleIds.join(', ') || '（无）'}</code>
    </p>
    <p style="margin:0; color:var(--text-secondary); font-size:14px;">
      这些样本在崩溃前完成了局部前向/反向计算，但对应全局梯度和优化器提交不存在；因此重放不会造成第二次权重更新。
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
      tone: '',
      html: `<strong>初始化：</strong>固定种子 ${config.seed} 生成 ${config.datasetSize} 个样本的确定性 epoch 排列；全局 batch=${config.batchSize}，初始拓扑 ${config.worldSizeBefore} workers，每 worker ${config.batchSize / config.worldSizeBefore} 个样本。`,
    },
    {
      tone: 'good',
      html: `<strong>已提交前序步：</strong>第 0–${config.failStep - 1} 步共 ${beforeCount} 个全局批次完成归约、SGD 更新和检查点提交。检查点只推进到游标 ${config.failStep * config.batchSize}。`,
    },
    {
      tone: 'warn',
      html: `<strong>模拟缩组/故障：</strong>第 ${config.failStep} 步在 ranks ${partial.completedRanks.join(', ')} 完成后中断；已处理样本为 <code>${partial.completedSampleIds.join(', ')}</code>。局部梯度被隔离，未形成全局更新。`,
    },
    {
      tone: 'warn',
      html: `<strong>应用半批策略：</strong>观察到的临时位置为 ${recovery.recoveryDecision.rawCursor}，但它不是提交位置；迭代器回滚到全局批边界 ${recovery.recoveryDecision.resumeCursor}，并按 worldSize=${config.worldSizeAfter} 重新分片。`,
    },
    {
      tone: 'good',
      html: `<strong>重放第 ${config.failStep} 步：</strong>同一批样本 ${comparison.steps[config.failStep].baselineSampleIds.join(', ')} 被重新计算，规范顺序归约得到与不中断轨迹相同的梯度和权重。`,
    },
    {
      tone: '',
      html: `<strong>缩组后继续：</strong>剩余 ${afterCount} 步均使用 ${config.worldSizeAfter} workers、每 worker ${config.batchSize / config.worldSizeAfter} samples；全局 batch 仍为 ${config.batchSize}。`,
    },
    {
      tone: 'good',
      html: `<strong>最终校验：</strong>${flattenedLength} 个扁平化样本位置一致，${comparison.totalSteps} 个梯度更新一致，最终权重逐位${comparison.finalWeightsExact ? '相同' : '不同'}。`,
    },
  ];

  elements.eventLog.innerHTML = events
    .map((event) => `<li class="${event.tone}">${event.html}</li>`)
    .join('');
}

function renderFlattenedOrder(result) {
  elements.flattenedOrder.textContent = result.comparison.flattenedOrder.baseline.join('  →  ');
}

function numericLines(step) {
  return parameterNames
    .map((name, index) => `
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
    `)
    .join('');
}

function renderSteps(result) {
  const partialByIndex = new Map(
    result.recovery.partialAttempts.map((event) => [event.globalStepIndex, event]),
  );

  elements.stepList.innerHTML = result.comparison.steps
    .map((step) => {
      const partialEvent = partialByIndex.get(step.index) ?? null;
      const baselineStatus = partialEvent
        ? '<span class="badge warning">在此注入半批故障</span>'
        : '<span class="badge good">已提交</span>';

      return `
        <article class="panel step-card" data-testid="step-card" data-step="${step.index}">
          <header class="step-head">
            <h3 class="step-title">
              全局步 ${step.index + 1}
              <span class="step-meta">zero-based step=${step.step} · epoch=${step.epoch} · cursor ${step.index * result.config.batchSize}</span>
            </h3>
            <div class="step-checks">
              <span class="badge ${step.sampleOrderExact ? 'good' : 'warning'}">
                ${step.sampleOrderExact ? '✓' : '!'} 样本顺序一致
              </span>
              <span class="badge ${step.gradientExact ? 'good' : 'warning'}" title="最大差 ${step.gradientMaxDiff}">
                ${step.gradientExact ? '✓ 梯度逐位一致' : `梯度差 ${formatDiff(step.gradientMaxDiff)}`}
              </span>
              <span class="badge ${step.weightsExact ? 'good' : 'warning'}" title="最大差 ${step.weightsMaxDiff}">
                ${step.weightsExact ? '✓ 权重逐位一致' : `权重差 ${formatDiff(step.weightsMaxDiff)}`}
              </span>
            </div>
          </header>

          <div class="lanes">
            <section class="lane baseline">
              <div class="lane-title">
                <span>不中断训练</span>
                <span>${baselineStatus}</span>
              </div>
              ${partialEvent ? `
                <p class="partial-callout">
                  模拟 ${partialEvent.worldSize} workers 中 ${partialEvent.completedRanks.length} 个分片完成后崩溃；
                  已完成局部梯度丢弃，不推进检查点。
                </p>
              ` : ''}
              ${renderShardRows(step.baselineShards, partialEvent)}
            </section>

            <section class="lane recovery">
              <div class="lane-title">
                <span>缩组恢复轨迹</span>
                <span class="badge good">
                  ${step.index < result.config.failStep
                    ? `故障前 · ${step.recoveryWorldSize} workers`
                    : step.index === result.config.failStep
                      ? `回滚后整批重放 · ${step.recoveryWorldSize} workers`
                      : `恢复后 · ${step.recoveryWorldSize} workers`}
                </span>
              </div>
              ${renderShardRows(step.recoveryShards)}
            </section>
          </div>

          <details class="numeric-details">
            <summary>查看梯度、SGD 增量与权重（左/右分别为 4-worker 基线与 2-worker 恢复）</summary>
            <div class="numeric-grid">
              <div class="numeric-box">
                <h4>数值对比</h4>
                ${numericLines(step)}
              </div>
              <div class="numeric-box">
                <h4>本步校验</h4>
                <div class="line"><span>gradient max abs diff</span><span class="${step.gradientMaxDiff === 0 ? 'diff-zero' : ''}">${step.gradientMaxDiff}</span></div>
                <div class="line"><span>weight max abs diff</span><span class="${step.weightsMaxDiff === 0 ? 'diff-zero' : ''}">${step.weightsMaxDiff}</span></div>
                <div class="line"><span>baseline loss after</span><span>${formatFixed(step.baselineLoss, 9)}</span></div>
                <div class="line"><span>recovery loss after</span><span>${formatFixed(step.recoveryLoss, 9)}</span></div>
                <div class="line"><span>Δw exact</span><span class="${step.deltaExact ? 'diff-zero' : 'warning-text'}">${step.deltaExact}</span></div>
              </div>
            </div>
          </details>
        </article>
      `;
    })
    .join('');
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

  // Useful for automated browser assertions and manual console inspection.
  window.__LAST_COMPARISON__ = result;
}

elements.runButton.addEventListener('click', render);
elements.failStep.addEventListener('change', render);
elements.partialWorkers.addEventListener('change', render);

render();
