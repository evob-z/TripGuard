import time
from pathlib import Path

import torch
from dotenv import load_dotenv
# LangChain 组件
from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager, index
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载环境变量
load_dotenv()

# ================= 配置区域 =================

# 1. 路径配置
CURRENT_FILE_DIR = Path(__file__).parent.resolve()
DATA_DIR = CURRENT_FILE_DIR / "data"
PERSIST_DIRECTORY = DATA_DIR / "chroma_db"

# 2. 独立的 SQLite 记录数据库 (只服务于 build.py)
# 这样完全避开了 MySQL 的兼容性问题，也不影响主程序连接 MySQL
RECORD_DB_PATH = CURRENT_FILE_DIR / "record_manager_cache.sqlite"
RECORD_MANAGER_DB_URL = f"sqlite:///{RECORD_DB_PATH}"

# 3. 索引命名空间 (固定 ID)
INDEX_NAMESPACE = "trip_guard/policy_v1"

# 4. 支持的文件格式
SUPPORTED_EXTENSIONS = {'.txt', '.pdf'}


# ===========================================

def get_embedding_model():
    """获取 Embedding 模型单例"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔌 设备状态: Using {device}")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )


def load_documents_from_directory(directory: Path):
    """扫描目录并加载所有文档"""
    if not directory.exists():
        print(f"❌ 错误: 目录 {directory} 不存在")
        return []

    docs = []
    files = [f for f in directory.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        print("⚠️ 目录为空，无需处理。")
        return []

    print(f"📂 正在扫描目录: {directory.name} (共 {len(files)} 个文件)")

    for file_path in files:
        try:
            ext = file_path.suffix.lower()
            if ext == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif ext == '.pdf':
                loader = PyPDFLoader(str(file_path))
            else:
                continue

            file_docs = loader.load()

            # 【元数据标准化】
            # 使用文件名作为唯一标识 (Source ID)
            for doc in file_docs:
                doc.metadata["source"] = file_path.name
                if "page" in doc.metadata:
                    doc.metadata["source"] += f" (p{doc.metadata['page'] + 1})"

            docs.extend(file_docs)
            print(f"   - ✅ 已加载: {file_path.name}")

        except Exception as e:
            print(f"   - ❌ 加载失败: {file_path.name} | 原因: {e}")

    return docs


def sync_knowledge_base():
    """主同步逻辑"""
    print(f"\n{'=' * 40}")
    print(f"🚀 开始同步知识库 (Mode: SQLite Local)")
    print(f"{'=' * 40}\n")

    # 1. 准备向量数据库 (Chroma)
    embedding_model = get_embedding_model()
    vector_db = Chroma(
        persist_directory=str(PERSIST_DIRECTORY),
        embedding_function=embedding_model,
        collection_name="trip_guard_collection"
    )

    # 2. 初始化记录管理器 (使用本地 SQLite)
    print(f"🔗 连接记录数据库: {RECORD_DB_PATH.name}")
    record_manager = SQLRecordManager(
        INDEX_NAMESPACE,
        db_url=RECORD_MANAGER_DB_URL  # <--- 强制使用本地 SQLite
    )
    record_manager.create_schema()

    # 3. 加载文档
    print("\n1️⃣  加载源文件...")
    docs = load_documents_from_directory(DATA_DIR)

    if not docs:
        print("   没有可处理的文档。")
        return

    # 4. 切分文档
    print("\n2️⃣  执行切分 (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "；", "！", "？", " ", ""],
        keep_separator=True
    )
    splits = text_splitter.split_documents(docs)
    print(f"   共生成 {len(splits)} 个切片")

    # 5. 执行增量同步
    print("\n3️⃣  执行智能同步 (Indexing)...")

    indexing_start = time.time()
    result = index(
        splits,
        record_manager,
        vector_db,
        cleanup="full",  # 保持全量同步模式 (本地删了库里也删)
        source_id_key="source"
    )
    indexing_end = time.time()

    print(f"\n📊 同步报告 (耗时 {indexing_end - indexing_start:.2f}s):")
    print(f"   🟢 新增 (Added):    {result['num_added']}")
    print(f"   🔵 更新 (Updated):  {result['num_updated']}")
    print(f"   ⚪ 跳过 (Skipped):  {result['num_skipped']}")
    print(f"   🔴 删除 (Deleted):  {result['num_deleted']}")
    print(f"\n{'=' * 40}")
    print("✅ 知识库同步完成！")


if __name__ == "__main__":
    sync_knowledge_base()
