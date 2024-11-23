import requests
import json
from typing import List, Dict, Any
import tools_utils as tu

# 定义工具函数映射表
TOOL_MAP = {
    # 开关类工具: 直接使用布尔值参数
    "control_calculator": {
        "func": tu.control_calculator,
        "process_arg": lambda x: x  # 直接返回参数值，或者直接设为 None
    },
    "set_power_mode": {
        "func": tu.set_power_mode,
        "process_arg": lambda x: x   # 不需要处理参数
    },
    "set_airplane_mode": {
        "func": tu.set_airplane_mode,
        "process_arg": lambda x: x 
    },
    "control_task_manager": {
        "func": tu.control_task_manager,
        "process_arg": lambda x: x 
    },
    "control_camera": {
        "func": tu.control_camera,
        "process_arg": lambda x: x 
    },
    "camera_to_vLLM": {
        "func": tu.camera_to_vLLM,
        "process_arg": lambda x: x 
    },
    
    # 数值类工具: 需要int类型参数
    "set_volume": {
        "func": tu.set_volume,
        "process_arg": int
    },
    "set_brightness": {
        "func": tu.set_brightness,
        "process_arg": int
    },
    
    # 无参数工具
    "capture_screen": {
        "func": tu.capture_screen,
        "process_arg": None
    },
    "get_system_info": {
        "func": tu.get_system_info,
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


# BERT意图识别进行工具调用
def BERT_tool_call(current_transcription:str):
    '''
    1.意图识别
    2.工具调用
    '''
    text_list = [current_transcription]

    result = classify_tool(text_list)
    
    if result:
        print("请求成功!")
        print(f"处理时间: {result['during']}秒")

        # 1.解析工具名称
        tool_name = result['data'][0]['data']['func']
        if tool_name is not None:
            # 安全地获取参数
            tool_args = result['data'][0]['data']['args']
            tool_message = result['data'][0]['message']
            
            print(f"工具名称: {tool_name}")
            print(f"工具参数: {tool_args}")
            print(f"工具消息: {tool_message}")

            # 只有在需要action的工具才处理action
            if 'action' in tool_args and tool_args['action'] is not None:
                num = 0
                # 判断获取音量还是获取亮度
                if tool_name == "set_brightness":
                    num = tu.get_brightness()["data"]
                elif tool_name == "set_volume":
                    num = tu.get_volume()["data"]

                if tool_args['action'] == "+":
                    num += 10
                elif tool_args['action'] == "-":
                    num -= 10

                tool_result = execute_tool(tool_name, num)
            else:
                # 对于不需要action的工具，直接使用value参数（如果有的话）
                tool_value = tool_args.get('value', None)
                tool_result = execute_tool(tool_name, tool_value)

            status = tool_result['status']
            response_message = ""
            if status == "success":
                response_message = tool_result['message']
                print(f"工具执行结果: {response_message}")
            else:
                response_message = tool_result['message']
                print(f"工具执行失败: {response_message}")
            
            return response_message
        else:
            print("工具名称不存在")
            return None


# 使用示例
if __name__ == "__main__":
    '''
    这里添加标准对话测试
    
    1.帮我把屏幕亮度/声音调大一些
    2.

    帮我查询一下当前的内存信息
    '''
    # 测试函数
    BERT_tool_call("帮我把屏幕亮度调大一些")