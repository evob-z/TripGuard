# TripGuard - 智能差旅合规助手

<div align="center">

基于 LangChain 和 LangGraph 构建的智能差旅审批系统，集成 RAG 检索增强、工具调用和自动化审批流程。

</div>

## 📋 项目简介

TripGuard 是一个智能差旅合规助手，通过 AI Agent 技术自动化处理差旅申请审批流程。系统集成了：

- 🤖 **智能对话**: 基于 LangGraph 的多节点工作流编排
- 📚 **知识检索**: RAG (检索增强生成) 技术查询差旅政策
- 🌤️ **工具调用**: 自动查询目的地天气信息
- ✅ **自动审批**: 多维度合规性检查与智能决策
- 💾 **数据持久化**: 完整的对话日志和审批记录存储

## 🏗️ 项目架构

```
agent-verification/
├── RAG/                          # 检索增强生成模块
│   ├── data/                     # 知识库数据
│   │   └── policy.txt           # 差旅政策文档
│   ├── build.py                 # 向量数据库构建脚本
│   └── retriever.py             # 检索器实现（含 Reranker）
├── core/                         # 核心模块
│   ├── base.py                  # 工作流定义与编排
│   ├── llm.py                   # LLM 封装
│   ├── nodes.py                 # LangGraph 节点实现
│   ├── state.py                 # 状态定义
│   └── tools.py                 # 工具定义
├── tools/                        # 外部工具集成
│   └── weather.py               # 天气查询工具
├── md/                           # 文档目录
│   ├── AGENT_DESIGN_PATTERNS.md
│   ├── PATH_ISSUE_SOLUTION.md
│   └── TOOL_CALL_ERROR_SOLUTION_REPORT.md
├── config.py                     # 配置管理
├── database.py                   # 数据库操作
├── embeddings.py                 # 向量处理
├── main.py                       # 应用入口
├── requirements.txt              # 依赖清单
├── setup.py                      # 打包配置
└── .env                          # 环境变量配置（需自行创建）
```

## ✨ 核心功能

### 1. 智能对话交互
- 自然语言对话理解用户需求
- 支持多轮对话和上下文记忆
- 自动识别差旅申请意图

### 2. RAG 知识检索
- 基于 Chroma 向量数据库存储政策文档
- 使用 BGE-M3 模型进行向量化
- 集成 BGE-Reranker 提升检索精度（粗召回 20 条 → 精排序前 3 条）

### 3. 工具调用能力
- **政策查询**: 自动检索相关差旅政策
- **天气查询**: 实时获取目的地天气信息
- **申请提交**: 结构化提取差旅申请信息

### 4. 自动化审批流程
```
用户输入 → Agent决策 → 工具执行 → 数据同步 → 天气检查 
→ 合规性检查 → 审批决策 → 决策审查 → 数据库保存 → 结果格式化
```

### 5. 数据持久化
- **审批记录表**: 存储完整的差旅申请和审批结果
- **对话日志表**: 记录用户与 AI 的完整对话历史
- 支持 MySQL、PostgreSQL、SQLite 多种数据库

## 🛠️ 技术栈

| 类别        | 技术                                  |
|-----------|-------------------------------------|
| **AI 框架** | LangChain, LangGraph, LangSmith     |
| **大模型**   | OpenAI GPT                          |
| **向量数据库** | Chroma                              |
| **嵌入模型**  | BAAI/bge-m3, BAAI/bge-reranker-base |
| **前端**    | Streamlit (可选)                      |
| **数据库**   | SQLAlchemy, MySQL/PostgreSQL/SQLite |
| **其他**    | Pydantic, HTTPX, Python-dotenv      |

## 🚀 快速开始

### 1. 环境准备

**系统要求**:
- Python 3.10
- pip 包管理工具

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/evob-z/TripGuard.git
cd agent-verification

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件，配置以下 API 密钥：

```env
# OpenAI API
OPENAI_API_KEY=sk-xxxxxxxxxxxx

# LangChain 追踪（可选）
LANGCHAIN_API_KEY=lsv2_xxxxxxxxxxxx

# MCP(还没做)
AMAP_MAPS_API_KEY=xxxxxxxxxxxx

# ChatGPT API（还没做）
CHATGPT_API_KEY=xxxxxxxxxxxx

# 数据库配置（建议mysql）
# DATABASE_URL=mysql+pymysql://user:password@localhost:3306/tripguard
# DATABASE_URL=postgresql://user:password@localhost:5432/tripguard
# DATABASE_URL=sqlite:///agent_verification.db
```

### 4. 构建知识库

在使用 RAG 功能前，需要先构建向量数据库：

```bash
# 方式 1: 作为模块运行
python -m RAG.build

# 方式 2: 直接运行脚本
cd RAG
python build.py
```

构建过程会：
- 读取 `RAG/data/policy.txt` 中的差旅政策文档
- 使用 BGE-M3 模型进行向量化
- 存储到 `RAG/md/chroma_db` 目录

### 5. 运行应用

```bash
python main.py
```

### 6. 使用示例

```
=== TripGuard 智能差旅合规助手 ===
我可以帮您：
  1. 查询差旅政策和规定
  2. 提交差旅申请并进行审批
  3. 回答差旅相关问题

输入 'quit' 或 'exit' 退出

👤 您: 我想申请去北京出差3天

🤖 TripGuard: 好的，我来帮您处理差旅申请...
[系统自动查询天气、检查政策、进行审批]
```

## 📊 工作流说明

系统采用 LangGraph 构建的状态图工作流：

1. **Agent 节点**: 处理对话，判断是否需要调用工具或启动审批
2. **Tool Executor**: 执行政策查询、天气查询等工具
3. **Data Sync**: 从对话中提取结构化的差旅申请信息
4. **Check Weather**: 获取目的地天气情况
5. **Compliance Check**: 根据政策进行合规性检查
6. **Make Decision**: 综合判断是否批准申请
7. **Critique Decision**: 审查决策的合理性
8. **Save DB**: 将审批结果保存到数据库
9. **Format Result**: 格式化审批结果返回用户

![Architecture](workflow.png)

## 🔧 配置说明

### LangChain 追踪

在 `core/base.py` 中可配置：

```
python
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # 开启追踪
os.environ["LANGCHAIN_PROJECT"] = "agent-verification"
```

### 数据库配置

支持三种数据库类型，在 `config.py` 中配置：

- **SQLite** : 适合测试和小规模使用
- **MySQL**: 适合生产环境
- **PostgreSQL**: 适合高并发场景

### RAG 检索配置

在 `RAG/retriever.py` 中可调整：

```python
search_kwargs={"k": 20}  # 粗召回数量
top_n=3                  # Rerank 后保留数量
```

## 📝 开发指南

### 添加新工具

1. 在 `tools/` 目录创建新的工具模块
2. 在 `core/tools.py` 中定义工具描述
3. 在 `core/base.py` 中注册工具

### 添加新节点

1. 在 `core/nodes.py` 中实现节点逻辑
2. 在 `core/base.py` 的工作流中添加节点
3. 配置节点之间的边和条件

### 扩展数据模型

在 `database.py` 中修改 SQLAlchemy 模型定义，然后运行数据库迁移。

## 🐛 故障排查

### 常见问题

1. **向量数据库未找到**
   ```
   FileNotFoundError: 请先构建数据库
   ```
   解决：运行 `python -m RAG.build` 构建向量数据库

2. **API Key 未配置**
   ```
   ValueError: × 请在.env中设置OPENAI_API_KEY
   ```
   解决：在项目根目录创建 `.env` 文件并配置相应的 API Key

3. **模型下载缓慢**
   
   首次运行会下载 BGE 模型，可能需要较长时间。建议配置 HuggingFace 镜像：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

## 📚 文档

- [Agent 设计模式](md/AGENT_DESIGN_PATTERNS.md)
- [路径问题解决方案](md/PATH_ISSUE_SOLUTION.md)
- [工具调用错误解决报告](md/TOOL_CALL_ERROR_SOLUTION_REPORT.md)

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，欢迎通过 Issue 反馈。

---

**注意**: 本项目使用的模型和 API 可能需要付费，请注意使用成本。
