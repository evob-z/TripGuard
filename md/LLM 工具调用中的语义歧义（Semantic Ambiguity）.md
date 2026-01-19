# LLM 开发实战：工具调用中的语义歧义 (Semantic Ambiguity) 问题报告

**日期**: 2026-01-17
**项目**: TripGuard 差旅合规助手
**关键词**: Prompt Engineering, Tool Calling, Hallucination, Semantic Ambiguity

---

## 1. 问题定义 (Problem Definition)

在基于 LLM 的 Agent 开发中，**语义歧义 (Semantic Ambiguity)** 是指：当工具（Tool/Function）的参数名称或描述存在多义性时，大模型基于预训练数据的统计概率，错误地“脑补”了开发者意图之外的含义，导致参数提取错误或产生幻觉性追问。

**在本案例中的具体表现：**
开发者定义了一个名为 `TripSubmission` 的工具，其中包含一个字段 `title`，意图是获取用户的 **“职级/职称”**（如“学生”、“教授”）。然而，Agent 在执行时却询问用户：*“您希望这个差旅申请的标题是什么？”*，将 `title` 误解为 **“文章标题”** 或 **“表单名称”**。

---

## 2. 根本原因 (Root Cause Analysis)

为什么模型会犯这种“低级”错误？我们需要从大模型的运行机制来理解：

### 2.1 训练数据的概率分布 (Prior Probability)
LLM 是基于海量文本训练的。在通用的语料库（Github代码、网页表单、文档）中，单词 **"Title"** 的语义分布极度不均衡：
* **90% 场景**：指代 `Article Title`（文章标题）、`Page Title`（页面标题）、`Project Title`（项目名称）。
* **10% 场景**：指代 `Job Title`（职位头衔）或 `Person Title`（尊称，如 Mr./Mrs.）。

当 `TripSubmission`（差旅提交）这个上下文出现时，模型联想到了“填写申请单”。在大多数申请单场景下，"Title" 通常指“申请单的主题/标题”。因此，模型优先激活了高概率的语义路径。

### 2.2 “变量名即 Prompt” 原则
在 Function Calling 中，**变量名 (Parameter Name)** 的权重往往高于 **描述 (Description)**。
即便你在 `description` 中写了 *"出差人的职级"*，但如果变量名本身叫 `title`，模型在快速推理（尤其是 Attention 机制分配权重）时，可能会忽略描述，直接被变量名的字面意思带偏。这种现象被称为 **"语义覆盖" (Semantic Overriding)**。

### 2.3 缺乏强约束 (Lack of Grounding)
如果 Prompt 中没有提供 Few-shot（少样本）示例，模型就只能依靠“猜”。模糊的命名（Ambiguous Naming）如 `data`, `info`, `title`, `type` 是导致幻觉的重灾区。

---

## 3. 解决方案 (Solutions)

针对此类问题，业界通用的最佳实践分为三个层级：

### ✅ 方案一：语义消歧重命名 (Explicit Renaming) —— **最推荐**
直接修改变量名，使其在语义上**唯一且排他**。

* **修改前**: `title` (歧义：标题 vs 职级)
* **修改后**: `job_rank` 或 `user_role_level` (无歧义：只能是职级)

> **实施代码**:
> ```python
> # core/tools.py
> class TripSubmission(BaseModel):
>     # ❌ Bad
>     # title: str = Field(..., description="职级")
>
>     # ✅ Good
>     job_rank: str = Field(..., description="申请人的身份职级，如：正高级、学生")
> ```

### 方案二：增强类型约束 (Type Constrained)
如果该字段的取值范围是有限的，使用 `Enum`（枚举）来强制模型做选择题，而不是填空题。这能彻底消除语义理解的偏差。

> **实施代码**:
> ```python
> from enum import Enum
>
> class JobRankEnum(str, Enum):
>     STUDENT = "学生"
>     PROFESSOR = "正高级"
>     STAFF = "普通教职工"
>
> class TripSubmission(BaseModel):
>     job_rank: JobRankEnum = Field(..., description="申请人职级")
> ```

### 方案三：Prompt 强化 (Description Boosting)
如果无法修改变量名（例如调用的是第三方 API），则必须在 Description 中使用**否定句**和**强调句**来进行“对齐修复”。

> **实施代码**:
> ```python
> title: str = Field(..., description="【注意】此处特指'Job Title'(职位头衔)，绝对不是表单标题！例如：'教授'、'经理'。")
> ```

---

## 4. 结论与启示 (Conclusion)

1.  **代码即自然语言**: 在 AI 开发中，变量命名不再只是为了给程序员看，更是**直接写给 AI 看的 Prompt**。
2.  **防御性命名**: 永远假设 AI 是“望文生义”的。避免使用 `content`, `type`, `name`, `title` 这种万能词，尽量使用 `file_content`, `ticket_type`, `user_full_name`, `job_rank` 等组合词。
3.  **测试边界**: 在测试 Agent 时，不仅要测试“它能不能跑通”，还要测试“它会不会理解歪”。

---

*参考资料:*
* Martin Fowler - Function calling using LLMs
* OpenAI Developer Community - Best practices for the Prompt
* Medium - The Perils of Ambiguous Naming