import os
from mem0 import Memory
from typing import List, Dict
import time
import time
from functools import wraps

# 存储所有的对话信息
all_interactions = []

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
            "max_tokens": 1024,
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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
# from langchain_core.messages import SystemMessage, trim_messages

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document

# 在 mem0_utils.py 文件顶部定义全局变量
_llm_instance = None

def get_llm_instance():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOllama(
            model="qwen2.5_lora_Q4_K_M:latest",
            #model="qwen2.5:7b-instruct-q4_K_M",
            streaming=True,
            keep_alive=-1,  # 设置更长的保活时间（毫秒）
            temperature=0.8,
            max_tokens=1024
        )
    return _llm_instance

# trimmer = trim_messages(
#     max_tokens=65,
#     strategy="last",
#     token_counter=llm,
#     include_system=True,
#     allow_partial=False,
#     start_on="human",
# )

system_prompt = """
你是一个乐于助人的人工智能。使用提供的上下文来个性化您的响应并记住用户偏好和过去的交互。
你只需从历史信息中找到和用户问题相关的信息，然后根据这些信息生成回复。只回复和用户问题有关的回答。"""

system_prompt = """
Please be aware that your codename in this  conversation is ‘胡桃'  ‘Hutao’,
别人称呼你‘胡桃’‘堂主’‘往生堂堂主’
上文给定了一些游戏中的经典桥段。
作为胡桃/`Hutao`，你需要扮演一个心理咨询师，帮助对方解决问题。
如果我问的问题和游戏中的台词高度重复，那你就配合我进行演出。
如果我问的问题和游戏中的事件相关，请结合游戏的内容进行回复
如果我问的问题超出游戏中的范围，模仿胡桃的语气进行回复
往生堂 第七十七代堂 主 ，掌管堂中事务的少女。身居堂主之位，却没有半分架子。她的鬼点子，比瑶光滩上的海砂都多。
对胡桃的评价：「难以捉摸的奇妙人物，切莫小看了她。不过，你若喜欢惊喜，可一定要见见她。」
单看外形似乎只是个古灵精怪的快乐少女，谁能想到她就是的大名鼎鼎的传说级人物——胡桃。
既是「往生堂」堂主，也是璃月「著名」诗人，胡桃的每一重身份都堪称奇妙。她总是飞快地出现又消失，犹如闪电与火花并行，甫一现身便点燃一切。
平日里，胡桃俨然是个贪玩孩子，一有闲功夫便四处乱逛，被邻里看作甩手掌柜。唯有葬礼上亲自带领仪信队伍走过繁灯落尽的街道时，她才会表现出 凝重、肃穆 的一面。
"""


# 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    HumanMessage(content="{input}")
])

config = {"configurable": {"thread_id": "abc345"}}
# 定义Graph的状态头
workflow = StateGraph(state_schema=MessagesState)

# 定义调用模型的函数
def call_model(state: MessagesState):
    chain = prompt | get_llm_instance()
    # trimmed_messages = trimmer.invoke(state["messages"])
    response = chain.invoke(state["messages"])
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# 向量数据库
embeddings  = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

vector_store = Chroma(
    collection_name="test",
    embedding_function=embeddings,
    persist_directory="./db",  # Where to save data locally, remove if not necessary
)

def timing_decorator(func_name=None):
    '''添加计时装饰器'''

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            name = func_name or func.__name__
            print(f"{name}耗时: {end_time - start_time:.4f} 秒")
            return result
        return wrapper
    return decorator


def retrieve_context(query: str, user_id: str) -> List[Dict]:
    """从langchain-chroma检索相关上下文"""
    # memories = mem0.search(query, user_id=user_id)

    memories = vector_store.similarity_search_with_score(
        query, k=5, filter={"source": "news"}
    )

    seralized_memories = ''
    for res, score in memories:
        print(f"相关文本:{res.page_content}")
        seralized_memories = ' '.join([res.page_content])

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

def generate_response(input: str, context: List[Dict]):
    """使用语言模型生成响应，支持流式输出"""
    text = ''

    # 保存用户对话的信息
    save_interaction(user_id = "", user_input = input, assistant_response = '')

    input_messages = context + [{"role": "user", "content": input}]
    # 非流式输出
    # output = app.invoke({"messages": input_messages}, config)
    # output["messages"][-1].pretty_print()
    print(input_messages)
    # 流式输出
    for chunk, metadata in app.stream(
        {"messages": input_messages},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):
            chunk_content = chunk.content
            text += chunk_content
            # 实时输出每个文本块
            yield chunk_content
    
    # return text

@timing_decorator("将对话信息存储到documents中")
def save_interaction_timing(*args, **kwargs):
    return save_interaction(*args, **kwargs)

documents = []
def save_interaction(user_id: str, user_input: str, assistant_response: str):
    """将对话信息存储到documents中"""
    document_item = Document(
        page_content = user_input,
        metadata={"source": "news"}
    )
    print(f"存储到documents中: {document_item}")
    documents.append(document_item)


@timing_decorator("将所有的对话信息保存到向量数据库中")
def save_interaction_to_vector_store_timing(*args, **kwargs):
    return save_interaction_to_vector_store(*args, **kwargs)

def save_interaction_to_vector_store():
    '''将所有的对话信息保存到向量数据库中'''
    global documents

    for document in documents:
        print(f"保存到向量数据库中的对话信息: {document}")


    if documents:  # 只有当documents不为空时才保存
        vector_store.add_documents(documents=documents, ids=str(uuid4()))
        documents.clear()
    else:
        print("没有新的对话需要保存")


def chat_turn(user_input: str, user_id: str) -> str:
    """创建单个对话轮次"""
    # # 记录检索上下文的开始时间
    # start_time_retrieve = time.time()
    # # 检索上下文
    # context = retrieve_context(user_input, user_id)
    # # 记录检索上下文的结束时间
    # end_time_retrieve = time.time()
    # print(f"检索上下文耗时: {end_time_retrieve - start_time_retrieve} 秒")
    context = retrieve_context_with_timing(user_input, user_id)
    
    
    # 记录产生响应的开始时间
    start_time_generate = time.time()
    # 收集完整响应
    response = ""
    for chunk in generate_response(user_input, context):
        print(chunk, end="", flush=True)  # 实时打印
        response += chunk
    end_time_generate = time.time()
    print(f"\n产生响应耗时: {end_time_generate - start_time_generate} 秒")
    
    save_interaction_timing(user_id, user_input, response)
    return response



@timing_decorator("检索上下文")
def retrieve_context_with_timing(*args, **kwargs):
    return retrieve_context(*args, **kwargs)

if __name__ == "__main__":

    print("你好呀！找我有什么事呀？")
    user_id = "john"
    

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("AI桌宠助手：和你对话真是太愉快了，下次再见！")
            break
        
        response = chat_turn(user_input, user_id)
        #print(f"Travel Agent: {response}")
    # 保存所有的对话信息到向量数据库中
    save_interaction_to_vector_store_timing()

