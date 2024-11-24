import openvino as ov
import torch
import os
from my_config import MyConfig
from my_tools.model import DIETClassifier
from transformers import AutoTokenizer

params = MyConfig()

now_dir = os.path.dirname(os.path.abspath(__file__))
output_model_dir = os.path.join(now_dir, "out_model")
model_path = os.path.join(output_model_dir, "best.pth")
output_ov_dir = os.path.join(output_model_dir, "bert_ov")
if not os.path.exists(output_ov_dir):
    os.mkdir(output_ov_dir)
output_ov_path = os.path.join(output_ov_dir, "bert.xml")
# load model and tokenizer
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model = DIETClassifier.from_pretrained(params.pre_model_path)
model.load_state_dict(state_dict)
tokenizer = AutoTokenizer.from_pretrained(
    params.pre_model_path
)
# get example input
example_text = "音量设置为50"
data = tokenizer(
    example_text,
    return_tensors="pt",
    return_offsets_mapping=True,
    padding="max_length",
    truncation=True,
    max_length=params.model_info["max_seq_len"]
)
input_ids = data["input_ids"]
attention_mask = data["attention_mask"]
token_type_ids = data["token_type_ids"]
example_input = (input_ids, attention_mask, token_type_ids)
# print("example_input: ", example_input)

ov_model = ov.convert_model(model, example_input=example_input)
# save result
ov.save_model(ov_model, output_ov_path)
print("openvino model save in ", ov_model)