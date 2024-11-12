import getpass
import os
import re
import argparse
import openvino_genai
import screen_brightness_control as sbc
import inspect
def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped.
    # False means continue generation.
    return False

global pipe, config,model_path

def init_LLM(model_path):
    global pipe, config
    # 初始化模型配置

    device = 'GPU'  # GPU can be used as well
    pipe = openvino_genai.LLMPipeline(model_path, device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 1024
    config.temperature = 0.8
    # 惩罚系数
    #config.repetition_penalty = 1.2

'''
    1.实现了将工具库的内容衔接到提示词模板中，第一次调用大模型
'''


# 定义数据库操作工具SystemOperations
class SystemOperations:
    def __init__(self):
        self.num = 0

    def Set_brightness(self, num: str):
        try:
            # 将字符串的亮度值转为整数
            brightness_level = int(num)
            # 调整屏幕亮度
            sbc.set_brightness(brightness_level)
            return f"设置屏幕亮度为 {brightness_level}。"
        except ValueError:
            return "亮度值必须是一个整数。"
        except Exception as e:
            return f"设置屏幕亮度失败: {e}"

# 注册工具
TOOLS = [
    {
        'name_for_human': '系统操作工具',
        'name_for_model': 'SystemOperations',
        'description_for_model': '系统操作工具提供了对windows系统级别的操作',
        'parameters': [
        {
            'name': 'Set_brightness',
            'description': '需要控制屏幕亮度时，控制亮度的大小，亮度为1~100之间的数字',
            'required': True,
            'schema': {
                'type': 'string',
                'properties': {
                    'num': {'type': 'string'},
                },
                'required': ['num']
            },
        }],
    },
    # 其他工具的定义可以在这里继续添加
]

# 将一个插件的关键信息拼接成一段文本的模板
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters:{parameters}
"""

PROMPT_REACT = """Answer the following questions as best you con. You have access to the following
{tool_descs}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {query}"""

import json


def generate_action_prompt(query):
    """
    根据用户查询生成最终的动作提示字符串。
    函数内部直接引用全局变量 TOOLS, TOOL_DESC, 和 PROMPT_REACT.
    参数：
    - query: 用户的查询字符串。
    返回：
    - action_prompt: 格式化后的动作提示字符串。
    """

    tool_descs = []
    tool_names = []

    for info in TOOLS:
        tool_descs.append(
            TOOL_DESC.format(
                name_for_model=info['name_for_model'],
                name_for_human=info['name_for_human'],
                description_for_model=info['description_for_model'],
                parameters=json.dumps(info['parameters'], ensure_ascii=False),
            )
        )
        tool_names.append(info['name_for_model'])

    tool_descs_str = '\n\n'.join(tool_descs)
    tool_names_str = ','.join(tool_names)

    # 拼接字符串
    action_prompt = PROMPT_REACT.format(tool_descs=tool_descs_str, tool_names=tool_names_str, query=query)
    return action_prompt


'''
    2.解析大模型的输出内容
'''


def parse_plugin_action(text: str):
    """
    解析模型的ReAct输出文本提取名称及其参数。
    参数：
    - text： 模型ReAct提示的输出文本
    返回值：
    - action_name: 要调用的动作（方法）名称。
    - action_arguments: 动作（方法）的参数。
    """
    # 查找“Action:”和“Action Input：”的最后出现位置
    action_index = text.rfind('\nAction:')
    action_input_index = text.rfind('\nAction Input:')
    observation_index = text.rfind('\nObservation:')

    # 如果文本中有“Action:”和“Action Input：”
    if 0 <= action_index < action_input_index:
        if observation_index < action_input_index:
            text = text.rstrip() + '\nObservation:'
            observation_index = text.rfind('\nObservation:')

    # 确保文本中同时存在“Action:”和“Action Input：”
    if 0 <= action_index < action_input_index < observation_index:
        # 提取“Action:”和“Action Input：”之间的文本为动作名称
        action_name = text[action_index + len('\nAction:'):action_input_index].strip()
        # 提取“Action Input：”之后的文本为动作参数
        action_arguments = text[action_input_index + len('\nAction Input:'):observation_index].strip()
        return action_name, action_arguments

    # 如果没有找到符合条件的文本，返回空字符串
    return '', ''


import json


def execute_plugin_from_react_output(response):
    """
    根据模型的 ReAct 输出执行相应的插件调用，并返回调用结果。
    参数：
    - response: 模型的 ReAct 输出字符串。
    返回：
    - result_dict: 包括状态码和插件调用结果的字典。
    """
    # 从模型的 ReAct 输出中提取函数名称及函数入参
    result_dict = {}
    
    result_dict["status_code"] = 404
    result_dict["result"] = "未找到匹配的插件配置"
    
    plugin_configuration = parse_plugin_action(response)
    print("plugin_configuration:", plugin_configuration)
    first_config_line = plugin_configuration[1:][0].split('\n')[0]
    print("first_config_line:", first_config_line)

    try:
        config_parameters = json.loads(first_config_line)
        print("config_parameters:", config_parameters)
    except Exception as e:
        print("Error parsing JSON:", e)
        return result_dict


    result_dict = {"status_code": 200}

    for k, v in config_parameters.items():
        print("k:", k)
        print("v:", v)
        print("config_parameters.items():", config_parameters.items())

        for i in range(len(TOOLS)):
            if k in TOOLS[i]["parameters"][0]['name']:
                # 通过 eval 函数执行存储在字符串中的 python 表达式，并返回表达式计算结果。其执行过程实质上是实例化类
                tool_instance = eval(TOOLS[i]["name_for_model"])()
                # 然后通过 getattr 函数传递对象和字符串形式的属性或方法名来动态的访问该属性和方法 h
                tool_func = getattr(tool_instance, k)
                print("tool_func:", tool_func)
                print("v:", v)
                # tool_result = tool_func(v)
                
                sig = inspect.signature(tool_func)
                params = sig.parameters  # 获取参数列表

                # 检查 v 的类型并构造调用参数
                if isinstance(v, dict):
                    # 使用字典解包方式传参
                    tool_result = tool_func(**{key: v[key] for key in params if key in v})
                elif len(params) == 1:
                    # 单参数函数时，直接传入 v
                    tool_result = tool_func(v)
                else:
                    # 多参数函数时，适配 v 为参数元组传入
                    tool_result = tool_func(*v if isinstance(v, (list, tuple)) else (v,))
                
                print("tool_result:", tool_result)
                result_dict["result"] = tool_result
                return result_dict

    return result_dict


def AgentLLM(query):
    # 处理输入，解析prompt
    prompt = generate_action_prompt(query)
    prompt_machine = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(prompt)
    # 第一次调用LLM执行函数
    response = pipe.generate(prompt_machine, config, streamer)
    # 执行工具调用
    execute_plugin_from_react_output(response)
    # print("第一次执行LLM："+response)
    # print("==============")
    # 处理输出，截断输出
    # response = response.split('Observation:')[0] if 'Observation:' in response else response
    # print("截断后的输出：",response)
    match = re.search(r"Final Answer:\s*(.*)", response, re.DOTALL)
    final_answer = match.group(1).strip()
    
    return final_answer



def ChatLLM(response):
    # 调用函数并返回执行结果
    tool_result = execute_plugin_from_react_output(response)
    # print("tool_result:", tool_result)
    # print("=======")
    # 拼接成第二次对话的输入
    response += " " + str(tool_result)
    # print(response)

    # print("=======")
    # 第二次调用LLM，返回具体消息
    prompt_machine = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(response)

    return prompt_machine


