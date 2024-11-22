from ov_qwen2_vl import OVQwen2VLModel

model_dir = "./Qwen2-VL-2B-Instruct"
model = OVQwen2VLModel(model_dir, "GPU")


from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from transformers import TextStreamer
import requests
import time

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)

if processor.chat_template is None:
    tok = AutoTokenizer.from_pretrained(model_dir)
    processor.chat_template = tok.chat_template

example_image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
example_image_path = Path("demo.jpeg")

if not example_image_path.exists():
    Image.open(requests.get(example_image_url, stream=True).raw).save(example_image_path)

image = Image.open(example_image_path)
question = "描述这张图片"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": f"file://{example_image_path}",
            },
            {"type": "text", "text": question},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

print("Question:")
print(question)
print("Answer:")

# 添加性能测量
start_time = time.time()
first_token_time = None
total_tokens = 0

class PerformanceStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_special_tokens=True):
        super().__init__(tokenizer, skip_special_tokens=skip_special_tokens)
        self.token_count = 0
        
    def put(self, value):
        global first_token_time, total_tokens
        if self.token_count == 0:
            first_token_time = time.time() - start_time
        self.token_count += 1
        total_tokens += 1
        super().put(value)

streamer = PerformanceStreamer(processor.tokenizer, skip_special_tokens=True)
generated_ids = model.generate(**inputs, max_new_tokens=100, streamer=streamer)

end_time = time.time()
total_time = end_time - start_time

print("\n性能指标:")
print(f"首个token生成延迟: {first_token_time:.3f}秒")
print(f"总推理时间: {total_time:.3f}秒")
print(f"生成的总token数: {total_tokens}")
print(f"平均token生成速度: {total_tokens/total_time:.2f} tokens/秒")
