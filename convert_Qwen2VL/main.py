import os
from ov_qwen2_vl import OVQwen2VLModel
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from transformers import TextStreamer
import time


pt_model_id = "Qwen2-VL-2B-Instruct"
now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
download_dir = os.path.join(project_dir, "download")
output_dir = os.path.join(project_dir, "output")
pt_model_dir = os.path.join(download_dir, pt_model_id)
ov_model_dir = os.path.join(output_dir, "Qwen2-VL-2B-Instruct-ov")
ov_model = OVQwen2VLModel(ov_model_dir, device="GPU")


min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(pt_model_dir, min_pixels=min_pixels, max_pixels=max_pixels)

if processor.chat_template is None:
    tokenizer = AutoTokenizer.from_pretrained(pt_model_dir)
    processor.chat_template = tokenizer.chat_template


image_path = os.path.join(now_dir, "demo.jpeg")
# 拿拍出来的照片做测试（选第一张照片，或者按日期排序，选最新一张照片？）
image_dir = os.path.join(output_dir, "image")
for file in os.listdir(image_dir):
    if file.endswith("jpg"):
        image_path = os.path.join(image_dir, file)
        break

example_image_path = Path(image_path)


image = Image.open(example_image_path)
question = "描述一下图片内容，如果有人物，单独描述人物的表情和外貌特征，不要描述背景。"

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
        if self.token_count == 1:
            first_token_time = time.time() - start_time
        self.token_count += 1
        total_tokens += 1
        super().put(value)

streamer = PerformanceStreamer(processor.tokenizer, skip_special_tokens=True)
generated_ids = ov_model.generate(**inputs, max_new_tokens=60, streamer=streamer)

end_time = time.time()
total_time = end_time - start_time

print("\n性能指标:")
print(f"首个token生成延迟: {first_token_time:.3f}秒")
print(f"总推理时间: {total_time:.3f}秒")
print(f"生成的总token数: {total_tokens}")
print(f"平均token生成速度: {total_tokens/total_time:.2f} tokens/秒")
