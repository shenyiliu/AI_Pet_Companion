import openvino as ov
import os
import sys
import torch

core = ov.Core()
now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
repo_dir = os.path.join(project_dir, "OpenVoice")
sys.path.append(repo_dir)
download_dir = os.path.join(project_dir, "download")
output_dir = os.path.join(project_dir, "output")
output_ov_path = os.path.join(output_dir, "OpenVoice_ov")
from openvoice.api import BaseSpeakerTTS, ToneColorConverter, OpenVoiceBaseClass
import openvoice.se_extractor as se_extractor

CKPT_BASE_PATH = os.path.join(download_dir, "OpenVoice", "checkpoints")

en_suffix = os.path.join(CKPT_BASE_PATH, "base_speakers", "EN")
zh_suffix = os.path.join(CKPT_BASE_PATH, "base_speakers", "ZH")
converter_suffix = os.path.join(CKPT_BASE_PATH, "converter")
# enable_chinese_lang = True
pt_device = "cpu"
ov_device = "GPU"

en_source_default_se = torch.load(f"{en_suffix}/en_default_se.pth", weights_only=True)
en_source_style_se = torch.load(f"{en_suffix}/en_style_se.pth", weights_only=True)
zh_source_se = torch.load(f"{zh_suffix}/zh_default_se.pth", weights_only=True)

en_base_speaker_tts = BaseSpeakerTTS(f"{en_suffix}/config.json", device=pt_device)
en_base_speaker_tts.load_ckpt(f"{en_suffix}/checkpoint.pth")

tone_color_converter = ToneColorConverter(f"{converter_suffix}/config.json", device=pt_device)
tone_color_converter.load_ckpt(f"{converter_suffix}/checkpoint.pth")

print("load chinese speak")
zh_base_speaker_tts = BaseSpeakerTTS(os.path.join(zh_suffix,"config.json"), device=pt_device)
zh_base_speaker_tts.load_ckpt(os.path.join(zh_suffix, "checkpoint.pth"))

def get_pathched_infer(ov_model: ov.Model, device: str) -> callable:
    compiled_model = core.compile_model(ov_model, device)

    def infer_impl(x, x_lengths, sid, noise_scale, length_scale, noise_scale_w):
        ov_output = compiled_model((x, x_lengths, sid, noise_scale, length_scale, noise_scale_w))
        return (torch.tensor(ov_output[0]),)

    return infer_impl


def get_patched_voice_conversion(ov_model: ov.Model, device: str) -> callable:
    compiled_model = core.compile_model(ov_model, device)

    def voice_conversion_impl(y, y_lengths, sid_src, sid_tgt, tau):
        ov_output = compiled_model((y, y_lengths, sid_src, sid_tgt, tau))
        return (torch.tensor(ov_output[0]),)

    return voice_conversion_impl

EN_TTS_IR = os.path.join(output_ov_path, "openvoice_en_tts.xml")
ZH_TTS_IR = os.path.join(output_ov_path, "openvoice_zh_tts.xml")
VOICE_CONVERTER_IR = os.path.join(output_ov_path, "openvoice_tone_conversion.xml")

# read ov model
paths = [EN_TTS_IR, VOICE_CONVERTER_IR, ZH_TTS_IR]
ov_models = [core.read_model(ov_path) for ov_path in paths]
ov_en_tts, ov_voice_conversion, ov_zh_tts = ov_models


en_base_speaker_tts.model.infer = get_pathched_infer(ov_en_tts, ov_device)
tone_color_converter.model.voice_conversion = get_patched_voice_conversion(
    ov_voice_conversion, ov_device)
zh_base_speaker_tts.model.infer = get_pathched_infer(
    ov_zh_tts, ov_device
)
def predict(
    language: str,
    prompt: str,
    audio_file_path: str,
    style: str,
    audio_output_dir: str
):
    supported_languages = ["zh", "en"]
    if language not in supported_languages:
        pass
    if language == "zh":
        tts_model = zh_base_speaker_tts
        source_se = zh_source_se
        language = "Chinese"
        if style not in ["default"]:
            text_hint = f"[ERROR] The style {style} is not supported for Chinese, which should be in ['default']\n"
            return {
                "status": "failed",
                "file_path": None,
                "message": text_hint
            }
    else:
        tts_model = en_base_speaker_tts
        if style == "default":
            source_se = en_source_default_se
        else:
            source_se = en_source_style_se
        language = "English"
        supported_styles = [
            "default",
            "whispering",
            "shouting",
            "excited",
            "cheerful",
            "terrified",
            "angry",
            "sad",
            "friendly",
        ]
        if style not in supported_styles:
            text_hint = f"[ERROR] The style {style} is not supported for English, which should be in {*supported_styles,}\n"
            return {
                "status": "failed",
                "file_path": None,
                "message": text_hint
            }
    if len(prompt) < 2:
        text_hint = "[ERROR] Please give a longer prompt text \n"
        return {
            "status": "failed",
            "file_path": None,
            "message": text_hint
        }
    if len(prompt) > 200:
        text_hint = (
            "[ERROR] Text length limited to 200 characters for this demo, please try shorter text. You can clone our open-source repo and try for your usage \n"
        )
        return {
            "status": "failed",
            "file_path": None,
            "message": text_hint
        }
    try:
        # 获取音色
        target_se, audio_name = se_extractor.get_se(
            audio_file_path,
            tone_color_converter,
            target_dir=output_dir,
            vad=True
        )
        src_path = f"{audio_output_dir}/tmp.wav"
        # 文本转语音
        tts_model.tts(prompt, src_path, speaker=style, language=language)
        save_path = f"{audio_output_dir}/output.wav"
        # 下面这一行应该是固定的
        encode_message = "@MyShell"
        # 音色克隆
        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=save_path,
            message=encode_message,
        )
        text_hint = "Get response successfully \n"
        return {
            "status": "success",
            "file_path": save_path,
            "message": text_hint
        }
    except Exception as e:
        text_hint = f"[ERROR] Get target tone color error {str(e)} \n"
        return {
            "status": "failed",
            "file_path": None,
            "message": text_hint
        }

# todo define fastapi api
