import requests
from typing import Dict, Union


def get_real_weather(city: str) -> Dict[str, Union[str, int, None]]:
    """
    使用 wttr.in 获取实时天气 (无需 API Key)
    """
    try:
        print(f">>> [Tool] 正在通过网络查询 {city} 的天气...")

        # format=j1 代表以 JSON 格式返回
        url = f"https://wttr.in/{city}?format=j1"

        # 设置超时时间，防止网络卡死
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # 如果 404 或 500 报错

        data = response.json()

        # 解析 wttr.in 的复杂 JSON 结构
        current_condition = data['current_condition'][0]

        weather_desc = current_condition['weatherDesc'][0]['value']  # 如 "Sunny"
        temp_c = int(current_condition['temp_C'])  # 如 25

        # 简单翻译一下常见天气词汇 (可选，为了让 LLM 更好理解)
        # 实际项目中通常不需要翻译，LLM 能看懂英文

        return {
            "weather": weather_desc,
            "temp": temp_c,
            "error": None
        }

    except Exception as e:
        print(f"!!! [Tool Error] 获取天气失败: {e}")
        # 如果接口挂了，返回一个兜底的默认值，防止程序崩掉
        return {
            "weather": "Unknown",
            "temp": 0,  # 默认 0 度，避免报错
            "error": str(e)
        }


# 单元测试：直接运行这个文件试试
if __name__ == "__main__":
    print(get_real_weather("Shanghai"))
    print(get_real_weather("Beijing"))
