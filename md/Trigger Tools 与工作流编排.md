# Agent 架构面试指南：Trigger Tools 与工作流编排

## 1. 核心概念定义

### 什么是 Trigger Tool (触发器工具)？
在 Agent 系统中，工具（Tool）通常分为两类：
1.  **原子工具 (Atomic Tools)**：执行单一任务并返回结果，如“查询天气”、“搜索文档”。执行后通常返回到同一个 Agent 继续对话。
2.  **触发器工具 (Trigger/Handoff Tools)**：不仅传递数据，还充当**状态转换的开关**。当 LLM 调用此类工具时，意味着用户的意图发生了实质性变化，系统需要进入通过特定的**子工作流 (Sub-graph)** 或**移交 (Handoff)** 给另一个专用 Agent。

> **面试话术**：
> "项目中我设计了 `TripSubmission` 工具作为 Trigger Tool。它不仅仅是收集参数，它的调用会触发路由层将控制权从‘通用对话 Agent’转移到‘差旅审批工作流’，开启一系列严格的合规检查和数据库操作。"

---

## 2. 项目中的关键挑战与解决方案 (STAR法则)

### 🔴 难点一：OpenAI API 的严格消息序列限制 (Error 400)
**情境 (Situation)**：
在使用 OpenAI 协议的接口时，遇到 `Error 400: insufficient tool messages`。
**任务 (Task)**：
需要确保在 LLM 发出 `tool_calls` 后，消息历史中必须紧跟对应 `tool_call_id` 的执行结果，不能中断或跳过。
**行动 (Action)**：
1.  **放弃手动循环**：早期的手动 `for` 循环处理容易漏掉 ID 或顺序错误。
2.  **引入 LangGraph ToolNode**：利用 `langgraph.prebuilt.ToolNode`。它内置了自动消息对齐逻辑，确保即使工具报错，也会生成一个包含错误信息的 `ToolMessage` 回传给 LLM，保证对话链路闭环。
**结果 (Result)**：
彻底解决了 API 报错问题，系统健壮性提升，能够稳定处理多工具连续调用。

### 🔴 难点二：多意图识别与并行执行 (Mixed Intents)
**情境**：
用户在一个请求中同时包含不同类型的意图，例如：“帮我查北京天气(通用请求)，然后提交去上海的申请(业务流请求)”。
**行动**：
1.  **设计多路路由 (Multi-path Routing)**：自定义 `router_function`，解析 `tool_calls` 列表。
2.  **并行分支**：利用 LangGraph 的并行执行能力。如果检测到多个工具，路由函数返回节点列表 `['tool_executor', 'approval_workflow']`，让通用工具和业务流同时运行。
**结果**：
显著降低了用户等待时间，提升了交互体验，实现了“查天气”和“跑流程”互不干扰。

### 🔴 难点三：并行执行中的竞态条件 (Race Condition)
**情境**：
在并行执行“查天气”和“合规检查”时，系统报错 `KeyError: 'weather'`。原因是一条分支依赖另一条分支尚未写入的数据。
**行动**：
1.  **解耦依赖**：重构代码逻辑。认识到“查询静态政策”不应强依赖“实时天气数据”。
2.  **防御性编程**：在 `compliance_check` 节点中使用 `.get()` 方法处理缺失数据，并允许参数为空（如查询通用的“上海差旅规定”而非“上海雨天规定”）。
3.  **最终一致性**：将强依赖逻辑后置到汇聚节点（Decision Maker），在所有并行任务结束后再进行综合判断。
**结果**：
修复了崩溃 bug，同时保留了并行执行的高效性。

### 🔴 难点四：模型 API 能力差异 (Structured Output)
**情境**：
从 OpenAI 迁移到 DeepSeek V3/R1 时，代码报错 `This response_format type is unavailable`。因为 DeepSeek 暂不支持 OpenAI 的 `json_schema` 强制结构化输出模式。
**行动**：
1.  **降级策略 (Client-side Parsing)**：放弃依赖服务端的 `.with_structured_output()`。
2.  **使用 PydanticOutputParser**：改用 LangChain 的解析器，将 Schema 转换为 Prompt 中的 Format Instructions。
3.  **Prompt 强化**：在 System Prompt 中明确注入 JSON 格式要求，利用 DeepSeek 强大的指令遵循能力实现结构化输出。
**结果**：
实现了跨模型兼容，无需修改业务逻辑即可在不同 LLM 之间切换。

---

## 3. 架构设计模式总结

### 模式 A：路由模式 (The Router)
最常见的模式。LLM 作为一个分类器，决定下一步走哪条路。
* **代码体现**：`router_function` 根据 `tool_name` 决定返回 `"run_tool"` 还是 `"start_approval"`。
* **适用场景**：意图明确、互斥的任务分发。

### 模式 B：并行工作流 (Parallel Workflow)
* **代码体现**：
    ```
  python
    workflow.add_edge("data_sync", "check_weather")
    workflow.add_edge("data_sync", "compliance_check")
    # 两个节点都会执行，最后汇聚
    ```
* **关键点**：状态隔离（State Isolation）。确保并行的分支修改的是 State 中不同的字段，避免数据覆盖。

### 模式 C：人机回环 (Human-in-the-loop)
*(可作为扩展知识点提到)*
在触发 Trigger Tool 提交申请后，系统可以暂停（`interrupt_before=["make_decision"]`），等待人类经理在外部系统审批后，再恢复执行。LangGraph 的 `checkpointer` 机制天然支持这一点。

---

## 4. 面试常见 Q&A 模拟

**Q: 为什么不直接用一个大 Prompt 让 LLM 做完所有事，而要拆分成 Agent 和 Tools？**
**A:**
1.  **幻觉控制**：LLM 不擅长精确计算和实时查询，工具可以提供确定性的事实（Grounding）。
2.  **流程合规**：企业级应用（如差旅审批）必须遵循固定的 SOP（标准作业程序），Trigger Tool 强制将非结构化对话导入结构化代码流程，保证合规。
3.  **可维护性**：将“查天气”、“查政策”、“写数据库”解耦成独立节点，便于调试和测试。

**Q: 你的 Agent 是怎么保证输出结果符合 JSON 格式给前端用的？**
**A:**
我使用了 Pydantic 进行数据定义。对于支持 Function Calling 的模型，优先使用工具调用来约束输出；对于不支持的模型（如 DeepSeek），我使用 `PydanticOutputParser` 将 Schema 注入 Prompt，并在客户端进行校验和重试（Retry），确保最终结果一定是合法的 JSON 对象。