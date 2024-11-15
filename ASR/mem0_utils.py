import os
from mem0 import Memory
from typing import List, Dict
import time

config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "test",
            "path": "db",
        }
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "qwen2.5:7b-instruct-q4_K_M",
            "temperature": 0,
            "max_tokens": 2048,
            "ollama_base_url": "http://localhost:11434",  # Ensure this URL is correct
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "embedding_dims": 1536,
            # Alternatively, you can use "snowflake-arctic-embed:latest"
            "ollama_base_url": "http://localhost:11434",
        },
    }
}

mem0 = Memory.from_config(config_dict=config)


from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b-instruct-q4_K_M", 
                 streaming=True,
                 cache=True,
                 keep_alive=-1
                 )

config = {"configurable": {"thread_id": "abc234"}}

# 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""你是一个乐于助人的人工智能。使用提供的上下文来个性化您的响应并记住用户偏好和过去的交互。
                  你只需从历史信息中找到和用户问题相关的信息，然后根据这些信息生成回复。"""),
    MessagesPlaceholder(variable_name="context"),
    HumanMessage(content="{input}")
])


def retrieve_context(query: str, user_id: str) -> List[Dict]:
    """从Mem0检索相关上下文"""
    memories = mem0.search(query, user_id=user_id)
    seralized_memories = ' '.join([mem["memory"] for mem in memories])
    context = [
        {
            "role": "system", 
            "content": f"相关信息: {seralized_memories}"
        },
        {
            "role": "user",
            "content": query
        }
    ]
    return context

def generate_response(input: str, context: List[Dict]) -> str:
    """使用语言模型生成响应"""
    chain = prompt | llm

    response = {
        "context": context,
        "input": input
    }

    context = ""
    for chunk in chain.stream(response):
        print(chunk.content,end='')
        context += chunk.content
    return context


# 存储所有的对话信息
all_interactions = []

def save_interaction(user_id: str, user_input: str, assistant_response: str):
    """将交互保存到Mem0"""
    interaction = [
        {
          "role": "user",
          "content": user_input
        },
        {
            "role": "assistant",
            "content": assistant_response
        }
    ]
    # 将当前交互添加到全局变量
    all_interactions.append(interaction)
    


def chat_turn(user_input: str, user_id: str) -> str:
    """创建单个对话轮次"""
    # 记录检索上下文的开始时间
    start_time_retrieve = time.time()
    # 检索上下文
    context = retrieve_context(user_input, user_id)
    # 记录检索上下文的结束时间
    end_time_retrieve = time.time()
    print(f"检索上下文耗时: {end_time_retrieve - start_time_retrieve} 秒")
    
    # 记录产生响应的开始时间
    start_time_generate = time.time()
    # 产生响应
    response = generate_response(user_input, context)
    # 记录产生响应的结束时间
    end_time_generate = time.time()
    print(f"产生响应耗时: {end_time_generate - start_time_generate} 秒")
    

    # 保存交互
    save_interaction(user_id, user_input, response)

    return response


if __name__ == "__main__":

    print("欢迎来到您的个人旅行社计划！我如何帮助您今天的旅行计划?")
    user_id = "john"
    
    # 获取全部记忆
    response = mem0.get_all(user_id=user_id)

    for item in response:
        print(f"记忆：{item}")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("旅行社代理：感谢您使用我们的旅行计划服务。旅途愉快!")
            break
        
        response = chat_turn(user_input, user_id)
        print(f"Travel Agent: {response}")

    # 记录保存交互的开始时间
    start_time_save = time.time()
    # 将所有的对话信息添加到记忆中
    for interaction in all_interactions:
        mem0.add(interaction, user_id=user_id)
    # 记录保存交互的结束时间
    end_time_save = time.time()
    print(f"保存交互耗时: {end_time_save - start_time_save} 秒")

# # Initialize Memory with the configuration
# m = Memory.from_config(config_dict=config)
# app_id = "app-1"
# # 添加记忆
# response = m.add("我喜欢车", user_id="alice",metadata={"app_id": app_id})
# #print(response)
# # 添加记忆
# response = m.add("我爱冉冉宝贝", user_id="alice",metadata={"app_id": app_id})
# #print(response)
# # 查询相关记忆
# response = m.search(query="我喜欢谁呀？", user_id="alice")
# print(response)

# print("================")

# # 获取全部记忆
# # response = m.get_all(user_id="alice")

# # for item in response:
# #     print(item)
