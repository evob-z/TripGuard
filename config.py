import os
from dotenv import load_dotenv

# 1. 加载.env文件
load_dotenv()

# 2. 解决 UUID v7 警告
try:
    from langsmith import uuid7
    import uuid

    uuid.uuid4 = uuid7  # 全局替换
except ImportError:
    pass  # 如果没装 langsmith，省略


# 3. 解决TensorFlow & Numpy 兼容性告警
def silence_framework_warnings():
    import os
    import warnings
    import logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    logging.getLogger('tensorflow').setLevel(logging.ERROR)


silence_framework_warnings()

# 4. 获取API_KEYS
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
AMAP_MAPS_API_KEY = os.getenv("AMAP_MAPS_API_KEY")
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")

# 5. 数据库配置
# 从 .env 文件读取 DATABASE_URL，支持 MySQL/PostgreSQL/SQLite
# 未配置时默认使用 SQLite（仅用于快速测试）
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_verification.db')}"
)

if not DEEPSEEK_API_KEY:
    raise ValueError("× 请在.env中设置OPENAI_API_KEY")
if not QWEN_API_KEY:
    raise ValueError("× 请在.env中设置QWEN_API_KEY")
if not LANGCHAIN_API_KEY:
    raise ValueError("× 请在.env中设置LANGCHAIN_API_KEY")
if not AMAP_MAPS_API_KEY:
    raise ValueError("× 请在.env中设置AMAP_MAPS_API_KEY")
if not CHATGPT_API_KEY:
    raise ValueError("× 请在.env中设置CHATGPT_API_KEY")
