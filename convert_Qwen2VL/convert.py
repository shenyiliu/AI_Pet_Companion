import os
from ov_qwen2_vl import convert_qwen2vl_model
import nncf
# from pathlib import Path
# import requests


# if not Path("ov_qwen2_vl.py").exists():
#     r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/qwen2-vl/ov_qwen2_vl.py")
#     with open("ov_qwen2_vl.py", "w", encoding="utf-8") as f:
#         f.write(r.text)
pt_model_id = "Qwen2-VL-2B-Instruct"
now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
download_dir = os.path.join(project_dir, "download")
output_dir = os.path.join(project_dir, "output")
pt_model_dir = os.path.join(download_dir, pt_model_id)
ov_model_dir = os.path.join(output_dir, "Qwen2-VL-2B-Instruct-ov")

compression_configuration = {
    "mode": nncf.CompressWeightsMode.INT4_ASYM,
    "group_size": 128,
    "ratio": 1.0,
}

convert_qwen2vl_model(pt_model_dir, ov_model_dir, compression_configuration)