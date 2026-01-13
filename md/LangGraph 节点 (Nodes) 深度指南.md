# LangGraph 节点 (Nodes) 深度指南

在 LangGraph 中，**节点 (Node)** 是工作流执行的基本单元。每个节点通常是一个 Python 函数，负责接收当前的 **状态 (State)**，执行具体的业务逻辑（如调用 LLM、查询数据库、执行代码），并返回一个 **状态更新 (State Update)**。

---

## 1. 节点的类型

根据您的项目代码，节点主要分为以下几类：

### A. 智能代理节点 (Agent Node)
* **功能**：负责与 LLM 交互，进行意图识别、工具选择或生成回复。
* **代码示例**：`agent_node`
* **特征**：
    * 绑定工具 (`llm.bind_tools`)。
    * 包含 System Prompt。
    * 返回包含 `AIMessage` 的字典。

### B. 逻辑处理节点 (Logic/Processing Node)
* **功能**：不调用 LLM，仅处理数据格式、提取参数或进行条件判断。
* **代码示例**：`data_sync_node`
* **特征**：
    * 解析 `messages` 历史。
    * 提取 `tool_calls` 中的参数。
    * 构造 `ToolMessage` 回填历史（防止 400 错误）。
    * 返回具体的字段更新（如 `destination`, `budget`）。

### C. 外部交互节点 (IO/Side-effect Node)
* **功能**：与外部系统交互，如 API 调用、数据库读写。
* **代码示例**：
    * `check_weather_node` (API 查询)
    * `save_db_node` (数据库写入)
    * `compliance_check_node` (RAG 检索)
* **特征**：
    * 通常包含 `try/except` 块处理外部错误。
    * 将外部结果写入 State。

### D. 决策与反思节点 (Decision & Reflexion Node)
* **功能**：利用 LLM 进行推理、评判或结构化输出。
* **代码示例**：
    * `make_decision_node` (生成决策)
    * `critique_decision_node` (自我反思/审计)
* **特征**：
    * 配合 `PydanticOutputParser` 或 `with_structured_output` 强制输出 JSON。
    * 包含复杂的 Prompt 逻辑（如接收 Feedback 进行修正）。

### E. 内置节点 (Built-in Node)
* **功能**：LangGraph 预封装的通用功能节点。
* **代码示例**：`ToolNode` (在 `base.py` 中使用)

---

## 2. 编写节点要考虑的内容 (Checklist)

编写一个高质量的节点函数时，需要考虑以下 5 个核心要素：

### 1. 输入与输出 (Input/Output Schema)
* **输入**：必须接收 `state` 参数（通常是定义的 `TypedDict`，如 `TripState`）。
* **输出**：必须返回一个字典（`dict`），其键值对必须符合 `State` 的定义。
    * **注意**：LangGraph 会自动将返回的字典与现有 State 进行 **合并 (Merge)** 或 **追加 (Append)**（取决于 `State` 定义中的 `Annotated[..., add_messages]`）。

### 2. 状态读取 (State Access)
* 需要从 `state` 中安全地读取数据。
* **示例**：`destination = state.get("destination", "未知")`。使用 `.get()` 防止 `KeyError`。

### 3. 错误处理 (Error Handling)
* 节点内部应包含 `try/except` 机制，防止单个节点崩溃导致整个图停止。
* **示例**：`agent_node` 和 `make_decision_node` 中都包含了异常捕获，并返回兜底的错误信息或默认状态。

### 4. 幂等性与副作用 (Side Effects)
* 如果是 `save_db_node` 这种有副作用的节点，要确保逻辑正确（例如：是否会重复写入？）。
* 如果是纯逻辑节点，应保证相同的输入产生相同的输出。

### 5. 返回值构造 (Return Construction)
* **Message 更新**：如果是更新消息历史，确保返回的是列表 `{"messages": [new_message]}`。
* **ToolMessage 回填**：如果手动处理了工具调用（如 `data_sync_node`），必须手动构造并返回 `ToolMessage`，包含正确的 `tool_call_id`。

---

## 3. 节点的具体结构示例

以您的代码为例，一个典型的节点结构如下：

```python
def example_node(state: TripState):
    # 1. 解包/获取所需状态
    data = state.get("some_key")
    
    # 2. 执行核心逻辑 (LLM, API, DB, Logic)
    try:
        result = do_something(data)
        output = {"status": "SUCCESS", "data": result}
    except Exception as e:
        # 3. 错误处理
        output = {"status": "ERROR", "error": str(e)}
        
    # 4. 返回状态更新 (Partial Update)
    return output