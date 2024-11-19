from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_and_save_lora(base_model_name, lora_model_name, output_dir):
    # 加载原始模型
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # 加载 LoRA 微调权重
    lora_model = PeftModel.from_pretrained(model, lora_model_name)

    # 合并 LoRA 权重
    merged_model = lora_model.merge_and_unload()

    # 保存合并后的模型
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# 使用示例
base_model_name = "D:\LianXi\AI\AICat\LLM_model\Qwen2.5-7B-Instruct"
lora_model_name = "D:\LianXi\AI\AICat\LLM_model\lora-2024-11-18"
output_dir = "merge_lora"
# merge_and_save_lora(base_model_name, lora_model_name, output_dir)

# 测试合并后模型对话效果
model = AutoModelForCausalLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

input_text = "你好，世界！"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

