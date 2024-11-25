import os
import openvino as ov
import sys
import nncf
import torch
import nltk
from pathlib import Path



now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
repo_dir = os.path.join(project_dir, "OpenVoice")
sys.path.append(repo_dir)
sys.path.append(project_dir)
download_dir = os.path.join(project_dir, "download")
output_dir = os.path.join(project_dir, "output")

# 设置从本地目录加载模型权重，方便网络不好的童鞋
torch_hub_local = os.path.join(download_dir, "torch_hub_local")
if not os.path.exists(torch_hub_local):
    os.mkdir(torch_hub_local)
os.environ["TORCH_HOME"] = torch_hub_local
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
torch_hub_dir = os.path.join(torch_hub_local, "hub")
torch.hub.set_dir(torch_hub_dir)
nltk.data.path.append(os.path.join(download_dir, "nltk_data"))

from melo.api import TTS
from open_voice_v2.utils import OVOpenVoiceTTS, OVOpenVoiceConverter, core
from openvoice.api import ToneColorConverter, OpenVoiceBaseClass

# core = ov.Core()
output_ov_path = os.path.join(output_dir, "OpenVoice_v2_ov")
assert os.path.exists(download_dir)
checkpoint_dir = os.path.join(download_dir, "OpenVoice", "checkpoints_v2")
assert os.path.exists(checkpoint_dir)
source_se_dir = os.path.join(checkpoint_dir, "base_speakers", "ses")
assert os.path.exists(source_se_dir)
converter_dir = os.path.join(checkpoint_dir, "converter")
assert os.path.exists(converter_dir)
zh_tts_model_dir = os.path.join(checkpoint_dir, "myshell-ai", "MeloTTS-Chinese")
en_tts_model_dir = os.path.join(checkpoint_dir, "myshell-ai", "MeloTTS-English-v2")
# CKPT_BASE_PATH = Path(checkpoint_dir)

en_source_se_path = os.path.join(source_se_dir, "en-newest.pth")
zh_source_se_path = os.path.join(source_se_dir, "zh.pth")

pt_device = "cpu"

print("load chinese speaker")
zh_tts_model = TTS(
    language="ZH",
    device=pt_device,
    config_path=os.path.join(zh_tts_model_dir, "config.json"),
    ckpt_path=os.path.join(zh_tts_model_dir, "checkpoint.pth")
)
print("load english speaker")
en_tts_model = TTS(
    language="EN_NEWEST",
    device=pt_device,
    config_path=os.path.join(en_tts_model_dir, "config.json"),
    ckpt_path=os.path.join(en_tts_model_dir, "checkpoint.pth")
)

print("load tone converter")
tone_color_converter = ToneColorConverter(
    os.path.join(converter_dir,"config.json"),
    device=pt_device,
)
tone_color_converter.load_ckpt(
    os.path.join(converter_dir, "checkpoint.pth")
)


EN_TTS_IR = os.path.join(output_ov_path, "openvoice_en_tts.xml")
ZH_TTS_IR = os.path.join(output_ov_path, "openvoice_zh_tts.xml")
VOICE_CONVERTER_IR = os.path.join(output_ov_path, "openvoice_tone_conversion.xml")

paths = [EN_TTS_IR, VOICE_CONVERTER_IR, ZH_TTS_IR]
models = [
    OVOpenVoiceTTS(en_tts_model),
    OVOpenVoiceConverter(tone_color_converter),
    OVOpenVoiceTTS(zh_tts_model)
]
ov_models = []

for model, path in zip(models, paths):
    if not os.path.exists(path):
        ov_model = ov.convert_model(model, example_input=model.get_example_input())
        ov_model = nncf.compress_weights(ov_model)
        ov.save_model(ov_model, path)
    else:
        ov_model = core.read_model(path)
    ov_models.append(ov_model)

ov_en_tts, ov_voice_conversion = ov_models[:2]
ov_zh_tts = ov_models[-1]