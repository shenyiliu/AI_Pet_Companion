from langchain_huggingface import HuggingFacePipeline

device = "GPU"
model_path = "C:\AI_Pet_Companion\AI_Pet_Companion\output\ov-qwen2.5-7b-instruct-int4"

ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

ov_llm = HuggingFacePipeline.from_model_id(
    model_id=model_path,
    task="text-generation",
    backend="openvino",
    model_kwargs={"device": device, "ov_config": ov_config},
    pipeline_kwargs={"max_new_tokens": 2048},
)


ov_llm = ov_llm.bind(skip_prompt=True)


    
while True:
    question = input("请输入你的问题：")
    
    prompt_machine = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(question)
    
    generation_config = {"skip_prompt": True, "pipeline_kwargs": {"max_new_tokens": 2048}}
    chain = ov_llm.bind(**generation_config)

    for chunk in chain.stream(prompt_machine):
        print(chunk, end="", flush=True)