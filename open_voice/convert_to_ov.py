import os
import openvino as ov
import sys
import nncf
from pathlib import Path
from open_voice.utils import OVOpenVoiceTTS, OVOpenVoiceConverter
now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
repo_dir = os.path.join(project_dir, "OpenVoice")
sys.path.append(repo_dir)
from openvoice.api import BaseSpeakerTTS, ToneColorConverter, OpenVoiceBaseClass
import openvoice.se_extractor as se_extractor

core = ov.Core()
download_dir = os.path.join(project_dir, "download")
output_dir = os.path.join(project_dir, "output")
output_ov_path = os.path.join(output_dir, "OpenVoice_ov")
assert os.path.exists(download_dir)
checkpoint_dir = os.path.join(download_dir, "OpenVoice", "checkpoints")
assert os.path.exists(checkpoint_dir)
CKPT_BASE_PATH = Path(checkpoint_dir)

en_suffix = CKPT_BASE_PATH / "base_speakers/EN"
zh_suffix = CKPT_BASE_PATH / "base_speakers/ZH"
converter_suffix = CKPT_BASE_PATH / "converter"
# support Chinese
enable_chinese_lang = True

pt_device = "cpu"

print("load english speak")
en_base_speaker_tts = BaseSpeakerTTS(en_suffix / "config.json", device=pt_device)
en_base_speaker_tts.load_ckpt(en_suffix / "checkpoint.pth")

print("load tone converter")
tone_color_converter = ToneColorConverter(
    converter_suffix / "config.json",
    device=pt_device,
)
tone_color_converter.load_ckpt(converter_suffix / "checkpoint.pth")

if enable_chinese_lang:
    print("load chinese speak")
    zh_base_speaker_tts = BaseSpeakerTTS(zh_suffix / "config.json", device=pt_device)
    zh_base_speaker_tts.load_ckpt(zh_suffix / "checkpoint.pth")
else:
    print("not load chinese speak")
    zh_base_speaker_tts = None


EN_TTS_IR = os.path.join(output_ov_path, "openvoice_en_tts.xml")
ZH_TTS_IR = os.path.join(output_ov_path, "openvoice_zh_tts.xml")
VOICE_CONVERTER_IR = os.path.join(output_ov_path, "openvoice_tone_conversion.xml")

paths = [EN_TTS_IR, VOICE_CONVERTER_IR]
models = [
    OVOpenVoiceTTS(en_base_speaker_tts),
    OVOpenVoiceConverter(tone_color_converter),
]
if enable_chinese_lang:
    models.append(OVOpenVoiceTTS(zh_base_speaker_tts))
    paths.append(ZH_TTS_IR)
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
if enable_chinese_lang:
    ov_zh_tts = ov_models[-1]