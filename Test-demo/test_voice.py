
# 测试链接chroma数据库
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
# from langchain_core.messages import SystemMessage, trim_messages

# 在 mem0_utils.py 文件顶部定义全局变量
_llm_instance = None

def get_llm_instance():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOllama(
            model="qwen2.5:7b-instruct-q4_K_M",
            streaming=True,
            keep_alive=-1,  # 设置更长的保活时间（毫秒）
            temperature=0.8,
            max_tokens=1024
        )
    return _llm_instance

# 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""你是一个乐于助人的人工智能。使用提供的上下文来个性化您的响应并记住用户偏好和过去的交互。
                  你只需从历史信息中找到和用户问题相关的信息，然后根据这些信息生成回复。只回复和用户问题有关的回答。"""),
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


# 添加向量数据库存储

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document
import time

embeddings  = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)
# 创建向量数据库
vector_store = Chroma(
    collection_name="test",
    embedding_function=embeddings,
    persist_directory="./db",  # Where to save data locally, remove if not necessary
)

# 添加数据
document_1 = Document(
    page_content="我现在在深圳",
    metadata={"source": "news"}
)

document_2 = Document(
    page_content="我的职业是AI工程师",
    metadata={"source": "news"}

)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "news"}
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"}

)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "news"}

)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "news"}

)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "news"}

)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "news"}

)

document_9 = Document(
    page_content="你怎么能这个样子",
    metadata={"source": "news"}

)

document_10 = Document(
    page_content="今天天气真不错",
    metadata={"source": "news"}
)

document_11 = Document(
    page_content="梦想是成为一名算法专家",
    metadata={"source": "news"}
)

# 循环添加记忆
documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
    document_11
]
# start_time = time.time()
# uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=str(uuid4()))
# end_time = time.time()
# print(f"添加文档执行时间: {end_time - start_time:.4f} 秒")

# 根据问题查询相关数据
start_time = time.time()
results = vector_store.similarity_search_with_score(
    "LangGraph?", k=5, filter={"source": "news"}
)
for res, score in results:
    #print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
    print(res.page_content)
end_time = time.time()
print(f"执行时间: {end_time - start_time:.4f} 秒")

