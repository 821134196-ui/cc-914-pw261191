# 可重分片训练迭代器：4 worker → 2 worker 恢复对比

一个零后端依赖的浏览器演示：同一个确定性小线性模型先由 4 个模拟 worker 训练，
在某个全局批次只有部分 worker 完成时缩组/故障，再从检查点恢复为 2 个 worker。
页面会逐步对比两条轨迹的全局样本顺序、worker 分片、梯度、SGD 增量和最终权重。

## 直接打开

已经构建了可直接通过 `file://` 加载的 `web/app.classic.js`，因此可以：

```bash
# 方式一：直接用浏览器打开
open index.html        # macOS
xdg-open index.html    # Linux

# 方式二：启动零依赖静态服务器
npm start
# 然后访问 http://127.0.0.1:4173
```

也可以使用常见静态服务器从仓库根目录提供 `index.html`。

## 运行测试

```bash
npm test
```

测试使用 Node 内置 test runner，覆盖：

- epoch/seed 全局排列的确定性；
- 4 分片与 2 分片展平后为同一全局批次；
- 检查点在新拓扑下恢复同一游标；
- 半批游标 `28 -> 24` 的回滚与整批重放；
- 在多个故障步、完成 1/4、2/4、3/4 分片时，12 个更新步的样本、梯度、增量、权重全部一致。

## 半批的处理决定

**不尝试提交或修补一个只完成一半的全局批次，而是回滚到该批次起点并重放完整批次。**

原因是数据并行训练中可比较的状态边界是“整个全局 batch 的优化器步已提交”，而不是某个
worker 完成了自己的本地 microbatch。演示中的检查点只包含：

```js
{
  version: 1,
  cursor: 24,          // 已提交的扁平全局样本位置，恰好是全局 batch 边界
  batchSize: 8,
  datasetSize: 32,
  seed: 2611914,
  epoch: /* 由 cursor/datasetSize 推导 */,
  weights: [/* ... */],
  optimizer: { type: 'sgd', learningRate: 0.08 }
}
```

各 rank 的私有偏移**不进入检查点**。恢复时：

1. 找到最后一个已提交检查点，例如 `cursor = 24`；
2. 即使观察到半批临时位置 `observedCursor = 28`，也只把它用于审计；
3. 对齐/回滚到全局批边界 `resumeCursor = 24`；
4. 用同一个 seed/epoch 重新生成全局样本顺序；
5. 按新的 `worldSize = 2` 把同一个 8 样本批次切成 `4 + 4`；
6. 重放完整全局批次并提交一次优化器更新。

这会对已完成的局部样本产生 at-least-once 的重复计算，但由于半批局部梯度从未进入权重，
对模型状态和梯度更新仍是 exactly-once 效果。

## 为什么缩组后样本和权重仍相同

全局 batch size 固定为 8：

- 4 workers：每个 worker 2 个样本，分片为 `2 + 2 + 2 + 2`；
- 2 workers：每个 worker 4 个样本，分片为 `4 + 4`。

worker 分组只决定“在哪里计算”，不决定全局顺序。迭代器先产生确定性的全局排列，再做连续
分片；梯度归约时按 `sequencePosition` 规范排序并累加，而不是按 worker rank 或完成顺序累加。
因此本演示中两条轨迹连 IEEE-754 double 结果都逐位相同，页面显示最大梯度差和最大权重差均为 `0`。

> 真实分布式 AllReduce 的不同加法树可能产生微小浮点差异。若要求跨拓扑严格逐位一致，需要
> 确定性归约顺序、固定累加器/补偿求和或定点梯度；若只要求训练等价，通常使用数值容差即可。

## 代码结构

- `src/reshard-iterator.js`：全局排列、连续重分片、检查点、半批回滚策略。
- `src/training.js`：确定性数据集、小线性模型、模拟 worker 梯度、基线/恢复轨迹和对比。
- `web/app.js`：浏览器渲染与交互（ESM 源文件）。
- `web/app.classic.js`：由 `npm run build:browser` 生成的独立 IIFE，支持直接打开 `index.html`。
- `index.html` / `styles.css`：浏览器对比界面。
- `test/reshard.test.js`：核心不变量测试。

## 重新构建浏览器包

修改 `web/app.js` 或 `src/*.js` 后：

```bash
npm run build:browser
```

然后刷新浏览器；不需要框架或打包配置。
