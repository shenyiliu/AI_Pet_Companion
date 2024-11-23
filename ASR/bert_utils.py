import requests
import json
from typing import List, Dict, Any
import tools_utils as tools

# 定义工具函数映射表
TOOL_MAP = {
    # 开关类工具: 直接使用布尔值参数
    "control_calculator": {
        "func": tools.control_calculator,
        "process_arg": lambda x: x  # 直接返回参数值，或者直接设为 None
    },
    "set_power_mode": {
        "func": tools.set_power_mode,
        "process_arg": lambda x: x   # 不需要处理参数
    },
    "set_airplane_mode": {
        "func": tools.set_airplane_mode,
        "process_arg": lambda x: x 
    },
    "control_task_manager": {
        "func": tools.control_task_manager,
        "process_arg": lambda x: x 
    },
    "control_camera": {
        "func": tools.control_camera,
        "process_arg": lambda x: x 
    },
    "camera_to_vLLM": {
        "func": tools.camera_to_vLLM,
        "process_arg": lambda x: x 
    },
    
    # 数值类工具: 需要int类型参数
    "set_volume": {
        "func": tools.set_volume,
        "process_arg": int
    },
    "set_brightness": {
        "func": tools.set_brightness,
        "process_arg": int
    },
    
    # 无参数工具
    "capture_screen": {
        "func": tools.capture_screen,
        "process_arg": None
    },
    "get_system_info": {
        "func": tools.get_system_info,
        "process_arg": None
    }
}

def execute_tool(tool_name: str, tool_value: str) -> Dict:
    """
    执行指定的工具函数
    
    Args:
        tool_name: 工具名称
        tool_value: 工具参数值
        
    Returns:
        Dict: 工具执行结果
    """
    try:
        if tool_name not in TOOL_MAP:
            return {"status": "failed", "message": f"未知的工具名称: {tool_name}"}
            
        tool_info = TOOL_MAP[tool_name]
        func = tool_info["func"]
        process_arg = tool_info["process_arg"]
        
        # 执行函数
        if process_arg is None:
            # 无参数工具
            result = func()
        else:
            # 有参数工具
            processed_arg = process_arg(tool_value)
            result = func(processed_arg)
            
        return result
        
    except Exception as e:
        return {"status": "failed", "message": f"执行工具时出错: {str(e)}"}

def classify_tool(text_list: List[str], threshold: float = 0.8) -> Dict[str, Any]:
    """
    发送POST请求到工具分类API并解析返回结果
    
    Args:
        text_list: 需要分类的文本列表
        threshold: 分类阈值，默认0.6
        
    Returns:
        解析后的响应数据字典
    """
    # 设置API地址
    url = "http://127.0.0.1:5518/tool_classify/"
    
    # 准备请求头
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    # 准备请求数据
    payload = {
        "text_list": text_list,
        "threshold": threshold
    }
    
    try:
        # 发送POST请求
        response = requests.post(url, headers=headers, json=payload)
        
        # 检查请求是否成功
        response.raise_for_status()
        
        # 解析JSON响应
        result = response.json()
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    '''
    这里添加标准对话测试
    
    
    '''
    # 测试函数
    text_list = ["请你帮我音量调整小一些"]
    result = classify_tool(text_list)
    
    if result:
        print("请求成功!")
        print(f"处理时间: {result['during']}秒")

        # 1.解析工具名称、参数和消息
        tool_name = result['data'][0]['data']['func']
        if tool_name is not None:
            tool_args_action = result['data'][0]['data']['args']['action']
            tool_args_value = result['data'][0]['data']['args']['value']
            tool_message = result['data'][0]['message']
            
            print(f"工具名称: {tool_name}")
            print(f"工具参数: action: {tool_args_action} value: {tool_args_value}")
            print(f"工具消息: {tool_message}")

            # 控制加减亮度/屏幕
            if tool_args_action is not None:
                num = 0
                # 判断获取音量还是获取亮度
                if tool_name == "set_brightness":
                    num = tools.get_brightness()["data"]
                
                if tool_name == "set_volume":
                    num = tools.get_volume()["data"]

                if tool_args_action == "+":
                    num += 10
                elif tool_args_action == "-":
                    num -= 10

                tool_result = execute_tool(tool_name, num)
            else:
                # 2.执行工具
                tool_result = execute_tool(tool_name, tool_args_value)

            status = tool_result['status']
            if status == "success":
                response_message = tool_result['message']
                print(f"工具执行结果: {response_message}")
            else:
                print(f"工具执行失败: {tool_result}")
        else:
            print("工具名称不存在")
