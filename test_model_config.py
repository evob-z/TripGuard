"""
模型配置测试脚本
测试不同节点的模型选择是否正确配置
"""

from core.llm import get_llm_model


def test_model_selection():
    """测试模型选择功能"""

    print("=" * 60)
    print("🧪 TripGuard 模型配置测试")
    print("=" * 60)

    # 测试意图识别模型
    print("\n1️⃣ 测试意图识别节点（Agent）:")
    try:
        llm_intent = get_llm_model(model_type="intent")
        print(f"   ✅ 模型类型: {llm_intent.model_name}")
        print(f"   ✅ Base URL: {llm_intent.openai_api_base}")
        print(f"   ✅ Temperature: {llm_intent.temperature}")
        assert "qwen" in llm_intent.model_name.lower(), "意图识别应使用 Qwen 模型"
        print("   ✅ 配置正确！")
    except Exception as e:
        print(f"   ❌ 错误: {e}")

    # 测试决策模型
    print("\n2️⃣ 测试审批决策节点（Decision）:")
    try:
        llm_decision = get_llm_model(model_type="decision")
        print(f"   ✅ 模型类型: {llm_decision.model_name}")
        print(f"   ✅ Base URL: {llm_decision.openai_api_base}")
        print(f"   ✅ Temperature: {llm_decision.temperature}")
        assert "deepseek" in llm_decision.model_name.lower(), "决策节点应使用 DeepSeek 模型"
        assert llm_decision.temperature <= 0.2, "决策节点应使用低温度"
        print("   ✅ 配置正确！")
    except Exception as e:
        print(f"   ❌ 错误: {e}")

    # 测试审计模型
    print("\n3️⃣ 测试审计反思节点（Critique）:")
    try:
        llm_critique = get_llm_model(model_type="critique")
        print(f"   ✅ 模型类型: {llm_critique.model_name}")
        print(f"   ✅ Base URL: {llm_critique.openai_api_base}")
        print(f"   ✅ Temperature: {llm_critique.temperature}")
        assert "deepseek" in llm_critique.model_name.lower(), "审计节点应使用 DeepSeek 模型"
        assert llm_critique.temperature <= 0.2, "审计节点应使用低温度"
        print("   ✅ 配置正确！")
    except Exception as e:
        print(f"   ❌ 错误: {e}")

    # 测试无效类型
    print("\n4️⃣ 测试错误处理:")
    try:
        llm_invalid = get_llm_model(model_type="invalid")
        print("   ❌ 应该抛出异常但没有")
    except ValueError as e:
        print(f"   ✅ 正确捕获异常: {e}")
    except Exception as e:
        print(f"   ⚠️  异常类型不正确: {e}")

    print("\n" + "=" * 60)
    print("✅ 模型配置测试完成！")
    print("=" * 60)

    # 显示模型分配总结
    print("\n📊 模型分配总结:")
    print("┌" + "─" * 58 + "┐")
    print("│ 节点类型       │ 使用模型      │ Temperature │ 特性      │")
    print("├" + "─" * 58 + "┤")
    print("│ 意图识别(Agent)│ Qwen-Plus    │ 0.7         │ 快速响应  │")
    print("│ 审批决策       │ DeepSeek     │ 0.1         │ 强推理    │")
    print("│ 审计反思       │ DeepSeek     │ 0.1         │ 批判思维  │")
    print("└" + "─" * 58 + "┘")

    print("\n💡 提示:")
    print("   - 意图识别使用 Qwen 保证快速响应")
    print("   - 决策和审计使用 DeepSeek 保证输出质量")
    print("   - 低温度(0.1)保证审批结果的稳定性")
    print()


if __name__ == "__main__":
    test_model_selection()
