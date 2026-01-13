from core import base
from database import save_chat_log


def run_demo():
    print("=== TripGuard 智能差旅合规助手 ===")
    print("我可以帮您：")
    print("  1. 查询差旅政策和规定")
    print("  2. 提交差旅申请并进行审批")
    print("  3. 回答差旅相关问题")
    print("\n输入 'quit' 或 'exit' 退出\n")

    # 配置线程 ID
    thread_id = "user_123_session"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            # 获取用户输入
            query = input("\n👤 您: ").strip()
            
            if not query:
                continue

            # 记录用户发言
            save_chat_log(thread_id, "user", query)

            if query.lower() in ['quit', 'exit', '退出']:
                print("\n👋 再见！祝您旅途愉快！")
                break

            # 构造消息格式（LangGraph 需要 messages 格式）
            inputs = {"messages": [("user", query)]}

            # 运行工作流
            print("\n🤖 TripGuard: ", end="", flush=True)
            result = base.app.invoke(inputs, config=config)
            
            # 获取 AI 的最后回复
            last_message = result["messages"][-1]
            content = last_message.content
            print(content)

            # 记录 AI 发言
            save_chat_log(thread_id, "ai", content)
                
        except KeyboardInterrupt:
            print("\n\n👋 检测到中断信号，再见！")
            break
        except Exception as e:
            # 修复原始错误：不再尝试将异常对象当作函数调用
            print(f"\n⚠️  抱歉，处理您的请求时出现错误: {str(e)}")
            print("请重新输入或尝试其他问题。\n")
            # 在异常情况下也记录错误信息，但使用异常的字符串表示
            try:
                save_chat_log(thread_id, "ai", f"抱歉，处理您的请求时出现错误: {str(e)}")
            except Exception as log_error:
                print(f"记录日志时也出现错误: {log_error}")


if __name__ == '__main__':
    run_demo()
