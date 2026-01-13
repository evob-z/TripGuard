# LLM Agent 设计模式与架构综述

本文档旨在梳理当前主流的 LLM Agent 设计范式。结合 **TripGuard** 项目的演进（从线性脚本到工具驱动），我们可以看到 Agent 是如何从简单的“执行者”进化为具备规划、反思和协作能力的“智能体”的。

---

## 1. ReAct: 协同推理与行动 (Reasoning + Acting)

这是现代 Agent 的鼻祖模式。它解决了 LLM "懂推理但没手" 和 "有工具但不懂什么时候用" 的问题。

### 核心逻辑
Agent 在执行任务时，采用 **"Thought (观察/思考) -> Action (行动) -> Observation (结果)"** 的循环。它要求模型在行动前先生成推理轨迹（Reasoning Trace）。

* **Thought**: 用户想去哈尔滨，我需要先查天气。
* **Action**: 调用 `get_weather("Harbin")`。
* **Observation**: API 返回 -20°C。
* **Thought**: 天气很冷，根据政策需要检查是否允许...

### 适用场景
* 需要外部信息的单步或多步简单任务。
* **TripGuard V1/V2** 其实是 ReAct 的一种变体（线性或简单循环）。

> **出处**: Yao, S., et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR.

---

## 2. Tool-Driven / Function Calling (工具驱动/路由模式)

这是 **TripGuard V3 (当前架构)** 采用的模式，也是目前工业界落地的首选。

### 核心逻辑
不再强制要求模型输出 "Thought" 文本，而是利用 LLM 原生经过微调的 **Function Calling (Tool Calling)** 能力。
* **Router**: 模型直接输出结构化的 JSON 指令（如 `TripSubmission` 或 `lookup_policy`）。
* **Execution**: 确定的代码执行工具。
* **Response**: 结果回传给 LLM 进行自然语言生成。

### 相比 ReAct 的优势
* **稳定性高**: 减少了模型在 "Thought" 阶段胡言乱语导致格式错误的概率。
* **Token 节省**: 跳过了冗长的推理文本，直接进入参数提取。

> **出处**: Schick, T., et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS. (概念相关)

---

## 3. Plan-and-Solve / Planner-Executor (规划与执行分离)

当任务变得极其复杂（例如：“帮我规划一个从上海出发，途经东京和巴黎，最后去纽约的15天差旅，要符合所有当地政策”），单步的 ReAct 或 Router 往往会迷失方向。

### 核心逻辑
将 Agent 拆分为两个角色（或两个阶段）：
1.  **Planner (规划者)**: 不执行具体操作，只负责把大目标拆解为子任务列表 (Sub-goals)。
    * *Plan*: 1. 查上海到东京机票; 2. 查东京差旅政策; 3. 查巴黎酒店...
2.  **Executor (执行者)**: 领任务，挨个执行，并把结果汇报给 Planner。



### 适用场景
* 长程任务 (Long-horizon tasks)。
* 复杂行程规划（TripGuard 未来可能的升级方向）。

> **出处**: Wang, L., et al. (2023). *Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models*. ACL.

---

## 4. Reflexion: 自我反思与进化 (Self-Correction)

Agent 在执行失败时，不是直接报错，而是“自我反省”并尝试修正。

### 核心逻辑
引入一个 **Reflector (反思者)** 角色。
1.  **Try**: 尝试执行任务。
2.  **Evaluate**: 评判结果是否成功（例如由 Unit Test 或 LLM 裁判）。
3.  **Reflect**: 如果失败，生成一段“自我批评”存入短时记忆（“我刚才查天气用的城市名拼错了”）。
4.  **Retry**: 带着“自我批评”的记忆重试。

### 适用场景
* 代码生成（自动修复 Bug）。
* 严谨的合规审查（如果 TripGuard 第一次生成的审批意见被驳回，可以自动根据驳回理由重写）。

> **出处**: Shinn, N., et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS.

---

## 5. Multi-Agent Collaboration (多智能体协作)

当单一 LLM 的 Context Window 或能力不足以覆盖所有领域时，使用多个专门的 Agent 协作。

### 常见模式：Supervisor (监督者/管理者) 模式
* **Supervisor Agent**: 也就是包工头。它不干活，只负责分发任务。
* **Worker Agents**:
    * *WeatherAgent*: 专精天气。
    * *PolicyAgent*: 专精法律文档 RAG。
    * *BookingAgent*: 专精订票系统 API。

### TripGuard 的应用潜力
如果 TripGuard 未来接入了真实的飞书/钉钉审批流、携程订票接口、财务ERP系统，那么将这些功能拆分为独立的 Agent 统一管理会更清晰。

> **出处**: Park, J. S., et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. UIST. (虽然偏向模拟，但奠定了多 Agent 交互基础)

---

## 总结：TripGuard 的定位

| 模式 | 复杂度 | 适用性 | TripGuard 现状 |
| :--- | :--- | :--- | :--- |
| **ReAct** | ⭐⭐ | 简单问答 | V1/V2 阶段 |
| **Tool-Driven** | ⭐⭐⭐ | 明确意图的任务 | **V3 (当前阶段)** - 最佳工程实践 |
| **Plan-and-Solve** | ⭐⭐⭐⭐ | 复杂行程规划 | 未来 V4 可选 |
| **Multi-Agent** | ⭐⭐⭐⭐⭐ | 企业级全流程集成 | 未来 V5 可选 |