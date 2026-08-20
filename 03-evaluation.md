# LLM 缓存命中机制学习笔记（下篇）：评估与验证

> 上篇讲了缓存原理和 7 条优化思路，中篇拆解了 DSH 在架构层面做了哪些设计来保证命中率。这篇回答一个自然的问题：**这些优化和设计，效果到底怎么样？怎么验证的？**
>
> 系列：上篇是缓存原理和优化思路；中篇是 DSH 的架构设计；下篇（本文）是评估验证方法、对比分析和实验结果。
>
> ⚠️ **实验数据待补充：** 第 5 节的 7 个实验场景的方法论和预期已就绪，实际数据正在采集中，完成后会更新本文。

---

## 1. 评估目标：我们到底在验证什么

上篇总结了 7 条优化思路，中篇对应了 DSH 的 8 项架构设计决策。把它们放到一起，评估就有了明确的对象：

| 优化思路（上篇 §5） | 对应 DSH 设计（中篇 §6） | 待验证的假设 |
|---|---|---|
| ① 保持系统提示词稳定 | PromptSection 按 order 固定排序 | 前缀稳定性 → 多轮对话缓存命中率 |
| ② 利用子任务隔离 | 子 Agent fork 继承 | fork 后子 Agent 能否复用父 Agent 缓存 |
| ③ 避免不必要的变量变化 | 严格变量插值 + PromptContext 分离 | 变量变化对命中率的影响幅度 |
| ④ 固定工具列表顺序 | 确定性工具 Schema 排序 | 排序 vs 不排序的命中率差异 |
| ⑤ 同一会话内完成多轮 | 事件溯源日志 + 会话投影缓存 | 冷启动 vs 热恢复的命中率差异 |
| ⑥ 动态信息放在请求末尾 | PromptContext 注入 user 消息 | 动态信息位置对前缀缓存的影响 |
| ⑦ 规划上下文压缩节奏 | 低频检查点式压缩 | 逐轮滑动窗口 vs 低频压缩的命中率对比 |

**评估要回答三个问题：**

1. **效果问题**：这些设计在真实场景下能把缓存命中率拉到多高？比不做这些设计的基线高多少？
2. **代价问题**：缓存优化是否引入了行为退化？Agent 在优化前后能力是否一致？
3. **迁移问题**：哪些设计是 DSH 特有的，哪些是任何 Agent 框架都可以借鉴的？

---

## 2. 业界是怎么做缓存评估的

在动手设计实验之前，先看一下业界已有的实践，避免闭门造车。

### 2.1 DigitalOcean：从 7% 到 74% 的命中率优化

[DigitalOcean 的实践文章](https://www.digitalocean.com/community/conceptual-articles/prompt-caching-in-practice-hit-rate) 是目前公开资料中最完整的缓存评估案例之一。他们的做法：

- **场景**：750 个并发请求，每个请求约 6,000 token 前缀（Claude Haiku 4.5），不同请求之间的前缀部分重叠但非完全相同
- **初始状态**：命中率只有 7%，因为请求随机分发到不同推理实例，前缀缓存分散在各实例上
- **优化手段**：前缀感知路由——把共享相同前缀的请求发到同一推理实例
- **最终结果**：命中率从 7% 提升到 74%，成本降低 59%

**可以借鉴的方法论：**

1. 在优化前后各跑一批请求，直接从 API 响应的 `usage` 字段提取 `cached_tokens` 和 `prompt_tokens`
2. 计算 `cacheReadTokens / totalInputTokens` 作为命中率
3. 乘以各 token 的单价，得到实际成本节省

### 2.2 Focused Labs：Agent 缓存评估的边界

[Focused Labs 的文章](https://focused.io/lab/agent-prompt-caches-are-a-runtime-boundary) 提出了一个关键观点：**Agent 的 Prompt Cache 是一个运行时边界（runtime boundary）**。不同 Agent 框架对这个边界的处理方式直接决定了缓存效率。

他们衡量缓存效果的核心指标：

- **缓存命中率**：`cached_tokens / total_prompt_tokens`
- **缓存 ROI**：`(全价 token 成本 - 实际成本) / 全价 token 成本`
- **TTFT 改善**：命中 vs 未命中的首 token 延迟对比

### 2.3 学术界的做法

学术论文（如 [Cache-Aware Prompt Compression](https://arxiv.org/abs/2607.15516)）通常采用更严格的方法：

- 使用 ShareGPT 等公开对话数据集作为测试集
- 在多轮对话中测量累积命中率曲线（N=5 轮 vs N=50 轮）
- 对比不同 prompt 压缩策略对缓存命中的影响
- 报告 cost-quality tradeoff（成本降低 vs 质量退化）

但这对学习文档来说太重了。我们采用更轻量的方法。

---

## 3. 实验设计

### 3.1 核心指标

从 DeepSeek API 每次响应中直接提取，不需要额外基础设施：

```
缓存命中率 = cacheReadTokens / (inputTokens + cacheReadTokens + cacheWriteTokens)

成本节省率 = 1 - (实际成本 / 全价成本)
           = 1 - (inputTokens × 1.0 + cacheReadTokens × 0.1 + cacheWriteTokens × 0.01)
                 / (inputTokens + cacheReadTokens + cacheWriteTokens) × 1.0
```

> 价格系数参考 DeepSeek API 定价：缓存命中 token 约原价 10%，缓存写入 token 约 0.01 元/百万 token。

### 3.2 场景矩阵

覆盖上篇 7 条优化思路对应的场景。每条优化思路一个实验场景，有明确的对比基线：

| # | 场景 | 对比什么 | 预期 |
|---|------|---------|------|
| S1 | **多轮对话基线** | 固定 system prompt + 固定工具集，5 轮纯工具调用对话 | 第 1 轮冷启动后，第 2-5 轮 system prompt + 工具 schema 持续命中 |
| S2 | **工具顺序稳定性** | 相同工具集，一次按字典序固定排序，一次随机打乱 | 固定排序的命中率显著高于随机打乱 |
| S3 | **动态信息位置** | 动态上下文（cwd、时间）放在 system prompt 开头 vs 放在 user 消息末尾 | 放在末尾时 system prompt 前缀不受影响，命中率更高 |
| S4 | **子 Agent fork** | 父 Agent 执行 3 轮后 fork 子 Agent 执行独立任务 | 子 Agent 首个请求的 system prompt + 父 Agent 历史前缀可命中 |
| S5 | **冷启动 vs 热恢复** | 同一会话：首次启动 vs 从会话投影缓存恢复后再次请求 | 热恢复后首个请求命中率接近上次会话结束时 |
| S6 | **修改 persona 的影响** | 修改 system prompt 中一个段落后重新发起相同对话 | 全量失效，命中率归零 |
| S7 | **上下文压缩策略** | 逐轮滑动窗口截断 vs 低频检查点压缩，测量 10 轮内的平均命中率 | 低频压缩的累积命中率显著高于逐轮截断 |

### 3.3 实验环境

| 项目 | 选择 |
|------|------|
| 模型 | deepseek-chat（支持 Context Caching） |
| 框架 | DSH（DeepSeek Harness） |
| System prompt | 使用 DSH 默认 persona（约 2,000 token） |
| 工具集 | 标准 toolset（read / write / bash / glob / grep / web_search，约 3,000 token schema） |
| 每轮对话 | 用户提问 → Agent 调用 1-3 个工具 → 返回结果 |
| 重复次数 | 每个场景跑 3 次取平均值，排除网络抖动 |

### 3.4 数据采集方法

最小可用方案——直接从 `session.jsonl` 中提取，不需要写额外的度量代码：

```bash
# 从会话日志中提取每次模型调用的 token 用量
cat session.jsonl | grep '"type":"assistant/chunk"' | \
  jq '{inputTokens: .data.chunk.usage.inputTokens, cacheRead: .data.chunk.usage.cacheReadTokens, cacheWrite: .data.chunk.usage.cacheWriteTokens}'
```

进阶方案——使用 DSH 的 `token-meter` 服务，在 benchmark composition 中直接计算：

```ts
// 在 benchmark composition 中
const before = ctx.tokenMeter.measure(session)

// 执行一轮对话
await agent.turn(userMessage)

const after = ctx.tokenMeter.measure(session)
const delta = after.totalTokens - before.totalTokens

// 从本轮的事件中提取 cacheReadTokens
const usageEvents = session.events.filter(e =>
  e.type === 'assistant/chunk' && e.data?.chunk?.type === 'usage'
)
const cacheRead = usageEvents.reduce((s, e) =>
  s + (e.data.chunk.usage.cacheReadTokens ?? 0), 0
)
const totalInput = usageEvents.reduce((s, e) =>
  s + (e.data.chunk.usage.inputTokens ?? 0) + (e.data.chunk.usage.cacheReadTokens ?? 0), 0
)

console.log(`本轮命中率: ${(cacheRead / totalInput * 100).toFixed(1)}%`)
```

---

## 4. 对比框架分析：DSH vs Pi vs OpenCode

DSH 不是唯一的 Agent Harness。Pi（earendil-works/pi）和 OpenCode（anomalyco/opencode）是当前最活跃的两个同类 Harness。三者都解决同一个问题——如何组织一个 Agent 的提示词、工具、上下文和会话——但架构选择截然不同，对缓存的影响也截然不同。

> 选它们做对比，不是因为 LangChain 或 AutoGen 不重要，而是因为 Pi 和 OpenCode 与 DSH 处于**同一个赛道**：都是终端优先的 Coding Agent Harness，都面临「多轮对话 + 工具调用 + 长上下文」场景下的缓存挑战。对比同类更能看出架构决策的差异。

### 4.1 对比维度

对每个框架，从 4 个维度分析其对缓存的影响：

| 维度 | 为什么重要 |
|------|-----------|
| 系统提示词组织 | 决定了前缀是否稳定——这是缓存命中的第一道关口 |
| 工具 schema 注入 | 工具 schema 通常数千 token，它的位置和顺序直接影响前缀长度和稳定性 |
| 多轮对话历史管理 | 历史消息的追加方式决定后续轮次的缓存命中率 |
| 子 Agent / 多 Agent 上下文 | 跨 Agent 的上下文继承决定缓存能否跨 Agent 复用 |

### 4.2 DSH

| 维度 | 做法 | 对缓存的影响 |
|------|------|-------------|
| 系统提示词组织 | `PromptSection` 按 `order` 固定排序，组装结果确定性 | ✅ 前缀始终稳定，多轮对话中 system prompt 持续命中 |
| 工具 schema 注入 | 按 `toolOrder` 排序后拼入 system prompt 前缀 | ✅ 工具 schema 成为前缀的一部分，多轮不变则持续命中 |
| 多轮对话历史 | 事件溯源日志 → 确定性派生消息序列，追加式 | ✅ 历史前缀不重排，每轮只需为新消息计算 KV |
| 子 Agent 上下文 | fork 继承父 Agent 已完成轮次的事件日志 | ✅ 子 Agent 首个请求的前缀大量命中 |

**缓存友好度：高。** 几乎所有可变因素都被框架层消化了——开发者不需要手动管理缓存。系统提示词约 2,000 token，工具 schema 按固定顺序拼入前缀，整个前缀在会话生命周期内保持不变。

### 4.3 Pi（π）

Pi 是一个极简的 Coding Agent Harness，核心设计哲学是「最小化系统提示词」。Claude Code、OpenCode 等框架的系统提示词动辄 7,000–10,000 token，而 Pi 的系统提示词只有约 500 token。它用一个「Skills」系统按需加载额外的指令，而不是把它们全部塞进系统提示词。

| 维度 | 做法 | 对缓存的影响 |
|------|------|-------------|
| 系统提示词组织 | 极简 base prompt（~500 token）+ Skills 按需加载为 user 消息 | ✅ 最小的系统提示词意味着缓存失效率最低——即使完全失效，损失也只有 500 token |
| 工具 schema 注入 | 工具定义在运行时动态加载，注入到 user 消息而非 system prompt | ⚠️ 工具 schema 不在 system prompt 前缀中，但 Pi 的 philosophy 是「工具定义本身也尽量精简」。工具 schema 变化时不影响前缀 |
| 多轮对话历史 | 追加式消息列表，context compaction 由扩展负责 | ⚠️ 取决于 compaction 策略。Pi 的轻量级设计意味着开发者需要自己配置 compaction |
| 子 Agent 上下文 | 不支持内置的 fork 机制；子 Agent 通过 `Agent` tool 创建，上下文独立 | ❌ 子 Agent 的上下文独立构造，无法复用父 Agent 缓存 |

**缓存友好度：中高。** Pi 的缓存策略和 DSH 走了完全不同的路线：DSH 通过架构保证「前缀绝对稳定」，Pi 通过极简设计让「前缀即使失效也损失极小」。Pi 的系统提示词只有 500 token，即使全量失效，成本影响也远小于 DSH 的 2,000+ token 或 OpenCode 的 7,000–10,000 token。

**Pi 的独特优势：** 因为系统提示词极短，Pi 对 system prompt 变化的容忍度极高。修改 persona、增删工具、调整指令——这些操作在 DSH 和 OpenCode 中会导致数千 token 的缓存失效，在 Pi 中只损失 500 token。

**Pi 的代价：** Skills 按需加载意味着部分指令在首次需要时才会出现在上下文中，可能增加首轮对话的 token 消耗。工具 schema 的简洁性依赖于开发者遵循 Pi 的工具设计规范，不规范的冗长工具定义会削弱这个优势。

> 参考：[Pi Coding Agent: The Minimal Harness That Rewrites Itself](https://byteiota.com/pi-coding-agent-minimal-harness/)、[Pi 架构解析](https://deepwiki.com/earendil-works/pi/2.1-agent-loop-and-agentharness-(pi-agent-core))

### 4.4 OpenCode

OpenCode 是目前最流行的开源 Coding Agent Harness 之一，有着最丰富的插件生态。它的系统提示词非常庞大（7,000–10,000 token），这意味着缓存命中与否对成本的影响极大——要么全部命中（省大量 token），要么全部失效（贵很多）。

OpenCode 的缓存优化历程是一个典型的「先发现问题，再到处修」的故事：

| 维度 | 做法 | 对缓存的影响 |
|------|------|-------------|
| 系统提示词组织 | 庞大的系统提示词，由多个 instruction 段和 skill guidance 组成 | ❌ 早期版本有严重的缓存稳定问题。PR #14743 修复了 Anthropic 缓存命中率从 3% 到 97.6% 的问题；PR #33246 让系统提示词在会话创建后不可变 |
| 工具 schema 注入 | 工具定义在系统提示词中，工具顺序依赖插件加载顺序 | ⚠️ PR #14743 通过「system split + tool stability」修复了工具 schema 的稳定性问题，但需要插件显式配合 |
| 多轮对话历史 | 追加式消息列表，但早期版本存在缓存断点（breakpoint）被意外打断的问题 | ⚠️ PR #43510 修复了追加消息破坏缓存断点的问题。经历了多轮修复后趋于稳定 |
| 子 Agent 上下文 | 支持 subagent，但上下文继承方式取决于实现 | ⚠️ 子 Agent 的上下文需要手动管理，没有内置的 fork 语义 |

**缓存友好度：中（修复后）。** OpenCode 的缓存问题不是设计缺陷，而是**演进过程中的技术债务**——它没有在架构层面预先设计缓存保证，而是在社区反馈后逐一修复。目前经过多轮 PR 修复后，缓存稳定性已经大幅改善，但前提是开发者使用正确的配置和插件。

**OpenCode 的教训：** 它的缓存修复历程非常值得关注，因为它展示了「没有在设计阶段考虑缓存」会带来什么问题：

1. **[PR #14743](https://github.com/anomalyco/opencode/pull/14743)**：Anthropic 缓存命中率从 3% 提升到 97.6%。修复手段是「system split」——把系统提示词拆成不变的 system 部分和可变的 user 部分，让 system 部分可以持续命中缓存。这和 DSH 的 `PromptContext` 分离是同一个思路，但 OpenCode 是在上线后才发现并修复的。
2. **[PR #33246](https://github.com/anomalyco/opencode/pull/33246)**：让系统提示词在会话创建后不可变（immutable）。这本质上是在事后补上 DSH 的 `PromptSection` 的确定性保证。
3. **[PR #43510](https://github.com/anomalyco/opencode/pull/43510)**：修复追加消息破坏缓存断点的问题。这对应 DSH 中「事件溯源日志只追加、不重排」的设计。

**OpenCode 的系统提示词为什么这么大？** 因为它把大量指令（tool use guidance、skill guidance、safety rules、formatting rules）全部塞进了系统提示词。DSH 把它们拆分到 `PromptSection`（系统提示词）和 `PromptContext`（user 消息）两个通道，Pi 把它们拆到 Skills（按需加载）。OpenCode 的「全部塞进 system prompt」策略在功能上没有问题，但对缓存不友好——任何一个指令段的变化都会导致整个系统提示词的缓存失效。

> 参考：[OpenCode 系统提示词架构](https://github.com/0xtresser/OpenCode-Book/blob/main/EN/Chapter_06_Agent_System/6.4_System_Prompt_Architecture.md)、[OpenCode 缓存优化修复路径](http://www.daiyunguke.com/news/259251)

### 4.5 对比总结

| | DSH | Pi | OpenCode |
|---|---|---|---|
| 系统提示词规模 | ~2,000 token | ~500 token | 7,000–10,000 token |
| 前缀稳定性 | ✅ 架构保证 | ✅ 极简→失效损失小 | ⚠️ 多轮修复后趋于稳定 |
| 工具 schema 确定性 | ✅ 固定排序进前缀 | ✅ 不进前缀，按需加载 | ⚠️ 修复后稳定，需插件配合 |
| 多轮对话缓存 | ✅ 高 | ✅ 中高 | ⚠️ 修复后中高 |
| 跨 Agent 缓存复用 | ✅ fork 继承 | ❌ 独立 context | ⚠️ 手动管理 |
| 缓存设计哲学 | 架构保证稳定性 | 极简化降低损失 | 事后修复补齐 |

**三种路径，三种取舍：**

```
DSH 的路径：  「把前缀锁死，让它永远不变」
            → 前缀长（~5,000 token system+tool），但一旦缓存就持续命中
            → 适合：高频多轮对话、fork 子 Agent 场景

Pi 的路径：   「把前缀缩到最小，让它失效了也不心疼」
            → 前缀极短（~500 token），即使全量失效也可接受
            → 适合：频繁修改 prompt 的实验性场景、单轮轻量任务

OpenCode 的路径：「先把功能做全，缓存问题后面再修」
            → 前缀最长（~10,000 token），早期缓存极差，修复后大幅改善
            → 适合：插件生态丰富、需要大量内置指令的场景
```

**核心发现：** 三个框架对缓存问题的处理方式反映了一个更深层的架构选择——**「复杂度放在哪里」**。DSH 把复杂度放在框架层（插件化 prompt 组装、fork 语义、事件溯源），换取开发者无感的高缓存命中率；Pi 把复杂度放在「做减法」上（极简 prompt、按需加载 Skills），换取失效时的低成本；OpenCode 把复杂度放在功能丰富性上，缓存是后来才补上的。没有哪个路径绝对正确，但如果你在意缓存命中率，DSH 的路径是最省心的。

---

## 5. TODO：实验数据

以下实验需要实际运行，当前数据待填充。每个场景都给出了具体的执行步骤，可以直接照做。

### 5.1 实验前置准备

```bash
# 1. 确保 DEEPSEEK_API_KEY 已设置
export DEEPSEEK_API_KEY=sk-xxx

# 2. 创建一个 benchmark composition（或使用已有的 agent preset）
# 3. 确保使用的 agent preset 包含标准工具集
```

### 5.2 S1：多轮对话基线

**目的**：测量同一个会话中 5 轮对话的缓存命中率曲线。

**步骤**：

1. 启动一个 agent 会话
2. 发送 5 轮包含工具调用的请求（如「读取 package.json 并分析依赖」「搜索最近的 git log」「列出所有 TypeScript 文件」等）
3. 从 `session.jsonl` 中提取每轮的 `cacheReadTokens` 和 `inputTokens`
4. 计算每轮的命中率

**待填充数据**：

| 轮次 | inputTokens | cacheReadTokens | 命中率 | 备注 |
|------|-------------|-----------------|--------|------|
| 第 1 轮 | TODO | 0 | 0% | 冷启动，无可命中缓存 |
| 第 2 轮 | TODO | TODO | TODO% | system prompt + 工具 schema 应命中 |
| 第 3 轮 | TODO | TODO | TODO% | 历史前缀持续命中 |
| 第 4 轮 | TODO | TODO | TODO% | |
| 第 5 轮 | TODO | TODO | TODO% | |

**预期**：第 1 轮命中率为 0%，第 2 轮起 system prompt（~2,000 token）+ 工具 schema（~3,000 token）应持续命中，命中率应在 50-70% 之间（取决于历史消息的占比）。

### 5.3 S2：工具顺序稳定性

**目的**：验证固定排序 vs 随机排序的命中率差异。

**步骤**：

1. 运行一次固定排序的 5 轮对话（对照组）
2. 修改代码，让工具 schema 在每次请求时随机打乱
3. 再运行一次随机排序的 5 轮对话（实验组）
4. 对比两组每轮的命中率

**待填充数据**：

| 轮次 | 固定排序命中率 | 随机排序命中率 | 差额 |
|------|--------------|--------------|------|
| 第 1 轮 | 0% | 0% | — |
| 第 2 轮 | TODO% | TODO% | TODO% |
| 第 3 轮 | TODO% | TODO% | TODO% |
| 第 4 轮 | TODO% | TODO% | TODO% |
| 第 5 轮 | TODO% | TODO% | TODO% |

**预期**：随机排序在工具 schema 部分（~3,000 token）无法命中，每轮命中率应该比固定排序低 10-20 个百分点。

### 5.4 S3：动态信息位置

**目的**：验证动态信息放在 system prompt 开头 vs user 消息末尾的命中率差异。

**步骤**：

1. 对照组：在 system prompt 开头插入 `{{cwd}}` 变量，cwd 在每轮对话中变化
2. 实验组：将 cwd 信息通过 `PromptContext` 注入到 user 消息末尾
3. 各跑 5 轮对话，对比命中率

**待填充数据**：

| 轮次 | 开头注入命中率 | 末尾注入命中率 | 差额 |
|------|--------------|--------------|------|
| 第 1 轮 | 0% | 0% | — |
| 第 2 轮 | TODO% | TODO% | TODO% |
| 第 3 轮 | TODO% | TODO% | TODO% |
| 第 4 轮 | TODO% | TODO% | TODO% |
| 第 5 轮 | TODO% | TODO% | TODO% |

**预期**：cwd 放在开头时，system prompt 的第一个 token 就变了，全部缓存失效。放在末尾时，cwd 之前的 system prompt + 工具 schema 仍可命中。

### 5.5 S4：子 Agent fork

**目的**：验证 fork 后子 Agent 能否复用父 Agent 缓存。

**步骤**：

1. 父 Agent 执行 3 轮对话（建立缓存）
2. 父 Agent fork 一个子 Agent 来处理子任务
3. 记录子 Agent 首个请求的 `cacheReadTokens`
4. 对比：如果子 Agent 是全新会话（不 fork），首个请求的命中率

**待填充数据**：

| 场景 | inputTokens | cacheReadTokens | 命中率 |
|------|-------------|-----------------|--------|
| Fork 子 Agent（首个请求） | TODO | TODO | TODO% |
| 全新子 Agent（首个请求） | TODO | 0 | 0% |

**预期**：fork 后的子 Agent 应该能命中父 Agent 的 system prompt + 工具 schema + 3 轮历史前缀，命中率可能超过 80%。全新子 Agent 命中率为 0%。

### 5.6 S5：冷启动 vs 热恢复

**目的**：验证会话投影缓存对恢复后命中率的影响。

**步骤**：

1. 执行一个 5 轮对话，记录第 5 轮的命中率
2. 关闭会话（模拟进程重启）
3. 从会话投影缓存恢复会话，发送第 6 轮请求
4. 记录第 6 轮的命中率

**待填充数据**：

| 状态 | 轮次 | inputTokens | cacheReadTokens | 命中率 |
|------|------|-------------|-----------------|--------|
| 冷启动 | 第 1 轮 | TODO | 0 | 0% |
| 热运行 | 第 5 轮 | TODO | TODO | TODO% |
| 热恢复 | 第 6 轮（恢复后） | TODO | TODO | TODO% |

**预期**：热恢复后第 6 轮的命中率应接近第 5 轮，因为会话投影缓存保证了恢复后的前缀与上次会话结束时一致。

### 5.7 S6：修改 persona 的影响

**目的**：量化修改 system prompt 对缓存命中率的破坏程度。

**步骤**：

1. 用 persona A 执行 3 轮对话（建立缓存）
2. 将 persona 中的一个段落改为 persona B
3. 重新发起相同对话，记录第 1 轮的命中率

**待填充数据**：

| 场景 | 轮次 | inputTokens | cacheReadTokens | 命中率 |
|------|------|-------------|-----------------|--------|
| Persona A（第 3 轮） | 第 3 轮 | TODO | TODO | TODO% |
| Persona B（第 1 轮，新会话） | 第 1 轮 | TODO | 0 | 0% |

**预期**：修改 persona 后，system prompt 的第一个 token 就变了，全部缓存失效。除非 persona 修改只影响末尾段落，且缓存粒度足够细（但 DeepSeek 的 Context Caching 是前缀匹配，第一个不同就全失效）。

### 5.8 S7：上下文压缩策略

**目的**：对比逐轮滑动窗口 vs 低频检查点压缩的累积命中率。

**步骤**：

1. 对照组：每轮对话后截断最早的消息（滑动窗口）
2. 实验组：只在第 5 轮和第 10 轮执行压缩（检查点式）
3. 各跑 15 轮对话，记录每轮的命中率

**待填充数据**：

| 轮次 | 滑动窗口命中率 | 检查点压缩命中率 | 差额 |
|------|--------------|----------------|------|
| 第 1-4 轮 | TODO% | TODO% | — |
| 第 5 轮（首次压缩） | TODO% | TODO% | TODO% |
| 第 6-9 轮 | TODO% | TODO% | TODO% |
| 第 10 轮（二次压缩） | TODO% | TODO% | TODO% |
| 第 11-15 轮 | TODO% | TODO% | TODO% |

**预期**：滑动窗口在第 1-4 轮命中率与检查点式相同，但从第 5 轮开始（窗口开始滑动），每轮都因前缀移位而失效。检查点式只在第 5 和第 10 轮各失效一次，中间 4 轮享受高命中率。

---

## 6. 验证优化不破坏能力

缓存优化的目标是「花更少的 token 完成相同的事」。如果命中率提高了但 Agent 能力退化了，这个优化没有意义。

### 6.1 验证方法

DSH 提供了两套验证设施：

**快照测试（`pnpm run test:snapshot`）**：在 keyless 环境下回放录制的模型响应，比对优化前后的 transcript 是否一致。如果优化修改了提示词结构（如调整 PromptSection 的 order），快照测试会精确指出变化的位置。

**Real-API e2e（`pnpm run test:e2e`）**：在真实模型下验证 Agent 行为。对缓存优化来说，关注点：

1. 相同的任务仍然能完成——Agent 正确使用工具，输出预期结果
2. 工具调用链没有变化（优化不应该改变 Agent 的决策逻辑）
3. 多轮对话的行为一致（优化不应该影响上下文理解）

### 6.2 验证清单

| 检查项 | 怎么做 | 通过标准 |
|--------|--------|---------|
| 快照测试全绿 | `pnpm run test:snapshot` | 无意外 diff（预期内的 prompt 结构变化需人工审查） |
| Real-API e2e 全绿 | `DEEPSEEK_API_KEY=$KEY pnpm run test:e2e` | 所有测试用例通过 |
| 工具调用链一致 | 对比优化前后 session.jsonl 中的 tool 事件 | 相同的用户输入应触发相同的工具调用序列 |
| 输出质量不退化 | 人工审查关键对话的最终输出 | Agent 完成了任务且输出质量与优化前一致 |

---

## 7. 对其他 Harness 架构的指导意义

从 DSH 的实践和对比分析中，可以抽象出几条**框架无关的缓存友好设计原则**：

### 原则 1：把不变前缀的稳定性交给框架，而非开发者

DSH 的 `PromptSection` + `order` 排序机制，让系统提示词的前缀稳定性由框架保证。开发者不需要关心「我的 system prompt 是不是每次拼出来都一样」。其他框架如果让开发者手动拼接 system prompt 字符串，缓存命中率就完全取决于开发者的细心程度。

**可迁移做法**：框架提供结构化的 prompt 组装 API，保证相同输入产生相同输出，且组装顺序不受插件加载顺序影响。

### 原则 2：工具 schema 是前缀的一部分，不是每次请求的「附加物」

DSH 把工具 schema 拼入系统提示词前缀，而 OpenCode 的早期版本中工具 schema 的顺序依赖于插件加载顺序。前者让工具 schema 成为稳定前缀的一部分，后者可能导致工具 schema 的 token 序列在不同请求间漂移，破坏缓存。

**可迁移做法**：如果工具集在会话生命周期内不变，把工具 schema 放在 system prompt 的固定位置，让它成为前缀的一部分。OpenCode 的 PR #14743 通过「tool stability」修复了这个问题，说明这个原则是跨框架通用的。

### 原则 3：动态信息放在前缀末尾

DSH 的 `PromptContext` 注入到 user 消息中而非 system prompt 中，本质上是把变化的部分往后推。变化的东西越靠后，前面能命中的缓存越多。

**可迁移做法**：所有动态上下文（当前时间、工作目录、用户信息）不要放在 system prompt 开头，而是放在 user 消息中或 system prompt 的最末尾。

### 原则 4：子 Agent 创建时继承父 Agent 的完整前缀

DSH 的 fork 机制让子 Agent 从父 Agent 的已完成轮次日志中派生初始上下文。这在多 Agent 场景中是一个巨大的缓存优势——子 Agent 的首次请求不需要从头计算所有前缀。

**可迁移做法**：如果框架支持子 Agent，子 Agent 初始化时应该继承父 Agent 的完整上下文（至少是已完成轮次），而不是给一个空的 system prompt。

### 原则 5：上下文压缩应该低频、批量化

DSH 选用的检查点式压缩（低频、一次性压缩）而非逐轮滑动窗口，是缓存视角下的最优选择。每次压缩都是一次「缓存重置」，压缩频率越低，缓存摊销的时间越长。

**可迁移做法**：框架的上下文管理策略应该把压缩作为低频事件，两次压缩之间让缓存充分摊销 Prefill 成本。逐轮滑动窗口是缓存视角下最差的设计。

### 原则 6：为缓存命中率提供一等公民的度量

DSH 的 `token-meter` 和 `session projection` 让开发者可以随时查询会话级的缓存命中率，不需要手动从 API 响应中提取。这是「如果无法度量，就无法优化」的体现。

**可迁移做法**：框架应该在日志或度量 API 中暴露每次请求的缓存命中信息，让开发者能感知到缓存的实际效果。

---

## 8. 评估流程总览

```
┌─────────────────────────────────────────────────────────────┐
│                    评估流程                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 运行 S1-S7 实验场景                                      │
│     ├── 对照组：优化前的基线                                  │
│     └── 实验组：应用优化后                                    │
│                                                             │
│  2. 收集数据                                                 │
│     ├── 每轮 inputTokens / cacheReadTokens / cacheWriteTokens │
│     ├── 计算命中率 = cacheRead / totalInput                  │
│     └── 计算成本节省率                                        │
│                                                             │
│  3. 能力验证                                                 │
│     ├── pnpm run test:snapshot → 快照 diff 审查              │
│     └── pnpm run test:e2e → 真实模型行为验证                  │
│                                                             │
│  4. 输出结论                                                 │
│     ├── 哪些优化效果最显著                                    │
│     ├── 哪些优化有副作用（能力退化）                           │
│     └── 哪些原则可迁移到其他框架                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 参考链接

- [DSH TokenMeter 源码](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/llm/token-meter/src/index.ts)
- [DSH SessionProjectionCache 源码](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/session/session-projection-cache/src/index.ts)
- [DSH 测试策略文档](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/testing.md)
- [DeepSeek API 上下文硬盘缓存文档](https://api-docs.deepseek.com/zh-cn/guides/kv_cache/)
- [vLLM 自动前缀缓存设计](https://docs.vllm.ai/en/latest/design/automatic_prefix_caching.html)
- [DigitalOcean: Prompt Caching in Practice (7% → 74% hit rate)](https://www.digitalocean.com/community/conceptual-articles/prompt-caching-in-practice-hit-rate)
- [Focused Labs: Agent Prompt Caches Are a Runtime Boundary](https://focused.io/lab/agent-prompt-caches-are-a-runtime-boundary)
- [Pi Coding Agent: The Minimal Harness That Rewrites Itself](https://byteiota.com/pi-coding-agent-minimal-harness/)
- [Pi Agent Loop & AgentHarness 架构](https://deepwiki.com/earendil-works/pi/2.1-agent-loop-and-agentharness-(pi-agent-core))
- [OpenCode System Prompt 架构](https://github.com/0xtresser/OpenCode-Book/blob/main/EN/Chapter_06_Agent_System/6.4_System_Prompt_Architecture.md)
- [OpenCode 缓存优化修复路径（3% → 97.6%）](http://www.daiyunguke.com/news/259251)
- [OpenCode PR #14743: system split + tool stability](https://github.com/anomalyco/opencode/pull/14743)
- [OpenCode PR #33246: immutable system prompt](https://github.com/anomalyco/opencode/pull/33246)
- [Pi、OpenCode、DSH 架构对比：Harness 把复杂度放在哪里](https://www.53ai.com/news/LargeLanguageModel/2026081956327.html)
- [Cache-Aware Prompt Compression (arXiv)](https://arxiv.org/abs/2607.15516)