# RAG 模块路径问题解决方案

## 问题描述

### 错误现象
```
FileNotFoundError: 请先构建数据库
```

在不同位置运行程序时，会因为相对路径问题导致无法找到向量数据库，具体表现为：

1. 直接运行 `python core/base.py` 会报错
2. 在项目根目录运行 `python -m core.base` 会报错
3. 在不同目录下调用 RAG 模块时，数据库路径不一致

### 根本原因

原代码存在两个核心问题：

#### 1. **相对路径依赖运行位置**
```python
# 原代码
if __name__ == "__main__":
    PERSIST_DIRECTORY = "./md/chroma_db"
else:
    PERSIST_DIRECTORY = "RAG/md/chroma_db"
```

这种写法的问题：
- `./data/chroma_db` 相对于**当前工作目录（cwd）**，而不是脚本文件所在目录
- 如果在项目根目录运行 `python RAG/build.py`，路径为 `./data/chroma_db`（错误）
- 如果在 RAG 目录运行 `python build.py`，路径为 `./data/chroma_db`（正确）
- 如果在其他位置导入模块，路径为 `RAG/data/chroma_db`（可能错误）

#### 2. **模块导入时立即执行**
```python
# 原代码（第 88 行）
retriever = get_advanced_retriever()  # 导入模块时立即执行
```

这导致：
- 只要导入 `retriever.py` 模块，就会立即尝试加载数据库
- 即使只是想使用其他函数，也会因为数据库不存在而报错
- 无法延迟到真正需要时再初始化

## 解决方案

### 核心思路

1. **使用绝对路径**：基于脚本文件自身位置计算路径，而不依赖运行位置
2. **懒加载（Lazy Loading）**：延迟初始化，在真正需要时才加载检索器

### 具体修改

#### 1. 修改路径计算方式（retriever.py & build.py）

**修改前：**
```python
if __name__ == "__main__":
    PERSIST_DIRECTORY = "./md/chroma_db"
else:
    PERSIST_DIRECTORY = "RAG/md/chroma_db"
```

**修改后：**
```python
from pathlib import Path

# 使用绝对路径，基于当前文件所在目录定位数据库路径
# 无论从哪里调用，都能正确找到数据库
CURRENT_FILE_DIR = Path(__file__).parent.resolve()
PERSIST_DIRECTORY = CURRENT_FILE_DIR / "md" / "chroma_db"
```

**优势：**
- `Path(__file__)` 获取当前脚本文件的绝对路径
- `.parent.resolve()` 获取脚本所在目录的绝对路径
- 无论在哪个目录运行，路径都指向 `RAG/data/chroma_db`

#### 2. 实现懒加载机制（retriever.py）

**修改前：**
```python
# 全局单例（导入时立即执行）
retriever = get_advanced_retriever()

def query_policy(query: str) -> str:
    docs = retriever.invoke(query)  # 直接使用全局变量
    # ...
```

**修改后：**
```python
# 延迟初始化全局单例，避免导入时立即执行
_retriever_instance = None

def get_retriever():
    """获取检索器单例（懒加载）"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = get_advanced_retriever()
    return _retriever_instance

def query_policy(query: str) -> str:
    retriever = get_retriever()  # 使用懒加载方式获取检索器
    docs = retriever.invoke(query)
    # ...
```

**优势：**
- 只有在第一次调用 `query_policy()` 时才会初始化检索器
- 如果数据库不存在，只在真正使用时报错，而不是导入时就报错
- 支持单例模式，避免重复加载

#### 3. 改进错误提示

**修改后：**
```python
if not PERSIST_DIRECTORY.exists():
    raise FileNotFoundError(
        f"请先构建数据库。\n"
        f"期望的数据库路径: {PERSIST_DIRECTORY}\n"
        f"请运行: python -m RAG.build 或 cd RAG && python build.py"
    )
```

**优势：**
- 明确显示期望的数据库路径
- 提供清晰的解决方案

## 技术要点

### 1. `Path(__file__)` 的作用
- `__file__` 是 Python 内置变量，表示当前脚本的文件路径
- `Path(__file__).parent` 获取脚本所在目录
- `.resolve()` 将相对路径转换为绝对路径

### 2. 懒加载（Lazy Loading）设计模式
- **定义**：延迟对象的创建或初始化，直到真正需要时才执行
- **优势**：
  - 减少启动时间
  - 避免不必要的资源占用
  - 提供更好的错误处理时机

### 3. 单例模式（Singleton Pattern）
- 确保一个类只有一个实例
- 使用全局变量缓存检索器实例，避免重复加载大模型

## 验证方法

### 1. 构建数据库
```bash
# 方法1：从任意位置运行
python -m RAG.build

# 方法2：进入 RAG 目录运行
cd RAG
python build.py
```

### 2. 测试不同位置调用
```bash
# 在项目根目录
python -m core.base

# 直接运行
python core/base.py

# 在其他目录导入
cd tools
python -c "from RAG.retriever import query_policy; print(query_policy('测试'))"
```

## 总结

| 问题 | 原因 | 解决方案 | 效果 |
|------|------|----------|------|
| 路径不一致 | 相对路径依赖运行位置 | 使用 `Path(__file__)` 基于脚本位置计算绝对路径 | 无论在哪里运行都能找到正确路径 |
| 导入即报错 | 模块导入时立即执行初始化 | 使用懒加载，延迟到真正使用时初始化 | 只在需要时才加载，错误提示更友好 |
| 重复加载 | 每次调用都创建新实例 | 单例模式缓存实例 | 提升性能，避免重复加载模型 |

## 最佳实践建议

1. **路径处理**：项目中涉及文件路径的地方，优先使用 `pathlib.Path` 和 `__file__` 构建绝对路径
2. **模块初始化**：避免在模块顶层执行耗时操作，使用懒加载延迟初始化
3. **错误提示**：提供清晰的错误信息和解决方案，包括期望的路径和操作步骤
4. **跨平台兼容**：使用 `pathlib.Path` 自动处理不同操作系统的路径分隔符
