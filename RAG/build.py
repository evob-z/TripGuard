import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
# LangChain 相关库
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载环境变量
load_dotenv()

# --- 配置项 ---
# 使用绝对路径，基于当前文件所在目录定位
CURRENT_FILE_DIR = Path(__file__).parent.resolve()
knowledge_base_file = CURRENT_FILE_DIR / "data" / "policy.txt"
persist_directory = CURRENT_FILE_DIR / "data" / "chroma_db"  # 数据库存储路径

# 1. 检查数据源
if not knowledge_base_file.exists():
    print(f"❌ 错误: 知识库文件 {knowledge_base_file} 未找到。请检查 data 目录。")
    exit()

# 2. 检查旧数据库
if persist_directory.exists():
    print(f"⚠️ 检测到已存在的向量数据库: {persist_directory}")
    user_input = input("是否删除旧数据并重新构建？(y/n): ")
    if user_input.lower() == 'y':
        print("正在删除旧数据库...")
        shutil.rmtree(persist_directory)  # 强制删除文件夹
    else:
        print("跳过构建。")
        exit()

print('--- 🚀 开始构建向量数据库 ---')

# 3. 加载文档
print(f'1. 正在加载文件: {knowledge_base_file}...')
loader = TextLoader(str(knowledge_base_file), encoding='utf-8')
docs = loader.load()

# 4. 文本分割 (使用递归分割，效果更好)
print('2. 正在进行文本切分...')
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每个块的大小
    chunk_overlap=50  # 重叠部分，防止上下文丢失
)
splits = text_splitter.split_documents(docs)
print(f'   - 共切分为 {len(splits)} 个片段')

# 5. 初始化 Embedding 模型
print('3. 正在初始化 Embedding 模型...')

model_name = "BAAI/bge-m3"
print(f"   - (注意) 正在下载本地模型 {model_name}，可能需要几分钟...")
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 6. 向量化并存储
print('4. 正在写入 ChromaDB (向量化)...')
db = Chroma(
    persist_directory=str(persist_directory),  # Chroma需要字符串路径
    embedding_function=embedding_model
)

BATCH_SIZE = 5000  # 必须 < 5461

total_docs = len(splits)

for i in range(0, total_docs, BATCH_SIZE):
    # 切片操作：取出当前这一批
    batch = splits[i: i + BATCH_SIZE]

    # 写入当前批次
    db.add_documents(batch)

    # 打印进度
    current_count = min(i + BATCH_SIZE, total_docs)
    print(f"   - 已插入进度: {current_count} / {total_docs}")

print(f'✅ 索引构建完毕！已保存到 {persist_directory}')
