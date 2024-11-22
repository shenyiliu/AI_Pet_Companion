from pathlib import Path
import requests

if not Path("ov_qwen2_vl.py").exists():
    r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/qwen2-vl/ov_qwen2_vl.py")
    with open("ov_qwen2_vl.py", "w", encoding="utf-8") as f:
        f.write(r.text)


from ov_qwen2_vl import model_selector

model_id = model_selector()

print(model_id)


print(f"Selected {model_id.value}")
pt_model_id = model_id.value
model_dir = Path(pt_model_id.split("/")[-1])

from ov_qwen2_vl import convert_qwen2vl_model
import nncf

compression_configuration = {
    "mode": nncf.CompressWeightsMode.INT4_ASYM,
    "group_size": 128,
    "ratio": 1.0,
}

convert_qwen2vl_model(pt_model_id, model_dir, compression_configuration)