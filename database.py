from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, DECIMAL
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL

# 1. 数据库配置（从 config.py 统一读取）
# 支持通过 .env 文件的 DATABASE_URL 配置
# MySQL：mysql+pymysql://user:password@host:3306/dbname
# PostgreSQL：postgresql://user:password@host:5432/dbname
# SQLite（测试用）：sqlite:///path/to/agent_verification.db

Base = declarative_base()

# 根据数据库类型设置不同的 engine 参数
engine_kwargs = {"echo": False}
if DATABASE_URL.startswith("sqlite"):
    # SQLite 需要额外配置以支持多线程
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine)


# 2. 定义数据模型 (与你数据库里的表结构对应)

# [表1] 审批记录表
class TripRecord(Base):
    __tablename__ = "trip_records"
    # 如果数据库里已经建好了表，这里只是为了让 Python 知道表长什么样
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    job_rank = Column(String)
    destination = Column(String)
    days = Column(Integer)
    weather = Column(String)
    temp = Column(Integer)
    status = Column(String)
    final_decision = Column(Text)
    budget = Column(DECIMAL(10, 2), nullable=True)
    cost = Column(DECIMAL(10, 2), nullable=True)
    created_at = Column(DateTime, default=datetime.now)


# [表2] 对话日志表
class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String)  # user / ai / system
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.now)


# 3. 初始化
# 如果表已存在，create_all 会自动跳过，不会覆盖数据
def init_db():
    Base.metadata.create_all(bind=engine)


# 4. 写入工具函数

def save_trip_record(session_id, job_rank, destination, days, weather, temp, status, final_decision, budget=None, cost=None):
    """保存审批单"""
    session = SessionLocal()
    try:
        record = TripRecord(
            session_id=session_id,
            job_rank=job_rank,
            destination=destination,
            days=days,
            weather=weather,
            temp=temp,
            status=status,
            final_decision=final_decision,
            budget=budget,
            cost=cost
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        print(f">>> [DB] 审批单已归档 (ID: {record.id})")
        return record.id
    except Exception as e:
        print(f"!!! [DB Error] 保存审批失败: {e}")
        session.rollback()
    finally:
        session.close()


def save_chat_log(session_id, role, content):
    """保存对话日志"""
    session = SessionLocal()
    try:
        # 确保内容是字符串
        if not isinstance(content, str):
            content = str(content)

        log = ChatLog(
            session_id=session_id,
            role=role,
            content=content
        )
        session.add(log)
        session.commit()
    except Exception as e:
        print(f"!!! [DB Error] 保存对话日志失败: {e}")
    finally:
        session.close()


def get_chat_history(session_id):
    """根据 session_id 获取历史聊天记录"""
    session = SessionLocal()
    try:
        # 按时间顺序查询该 session 的所有日志
        logs = session.query(ChatLog) \
            .filter(ChatLog.session_id == session_id) \
            .order_by(ChatLog.created_at) \
            .all()

        # 返回格式：[(role, content), ...]
        return [(log.role, log.content) for log in logs]
    except Exception as e:
        print(f"!!! [DB Error] 读取历史记录失败: {e}")
        return []
    finally:
        session.close()


# 模块被导入时自动检查连接
init_db()
