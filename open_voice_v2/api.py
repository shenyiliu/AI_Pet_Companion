import openvino as ov
import os
import sys
import torch
import random
import time
import re
import uvicorn
import shutil
import nltk
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
# import zipfile


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
repo_dir = os.path.join(project_dir, "OpenVoice")
sys.path.append(repo_dir)
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
from openvoice.api import ToneColorConverter, OpenVoiceBaseClass
import openvoice.se_extractor as se_extractor

core = ov.Core()

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
# enable_chinese_lang = True
pt_device = "cpu"
ov_device = "AUTO"
en_source_se_path = os.path.join(source_se_dir, "en-newest.pth")
zh_source_se_path = os.path.join(source_se_dir, "zh.pth")
zh_source_se = torch.load(zh_source_se_path, map_location=pt_device)
en_source_se = torch.load(en_source_se_path, map_location=pt_device)

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


print("load tone...")


# url = "https://github.com/snakers4/silero-vad/zipball/v3.0"

#
# zip_filename = "v3.0.zip"
# output_path = torch_hub_dir / "v3.0"
# if not (torch_hub_dir / zip_filename).exists():
#     download_file(url, directory=torch_hub_dir, filename=zip_filename)
#     zip_ref = zipfile.ZipFile((torch_hub_dir / zip_filename).as_posix(), "r")
#     zip_ref.extractall(path=output_path.as_posix())
#     zip_ref.close()
#
# v3_dirs = [d for d in output_path.iterdir() if "snakers4-silero-vad" in d.as_posix()]
# if len(v3_dirs) > 0 and not (torch_hub_dir / "snakers4_silero-vad_v3.0").exists():
#     v3_dir = str(v3_dirs[0])
#     os.rename(str(v3_dirs[0]), (torch_hub_dir / "snakers4_silero-vad_v3.0").as_posix())

tone_color_converter = ToneColorConverter(
    os.path.join(converter_dir, "config.json"),
    device=pt_device,
)
tone_color_converter.load_ckpt(
    os.path.join(converter_dir, "checkpoint.pth")
)


def get_pathched_infer(ov_model: ov.Model, device: str) -> callable:
    compiled_model = core.compile_model(ov_model, device)

    def infer_impl(
       x,
       x_lengths,
       speakers,
       tones,
       lang_ids,
       bert,
       ja_bert,
       noise_scale,
       length_scale,
       noise_scale_w,
       sdp_ratio,
       max_len=None,
    ):
        ov_output = compiled_model((
            x,
            x_lengths,
            speakers,
            tones,
            lang_ids,
            bert.contiguous(),
            ja_bert.contiguous(),
            noise_scale,
            length_scale,
            noise_scale_w,
            sdp_ratio
        ))
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
print("load openvino model...")
paths = [EN_TTS_IR, VOICE_CONVERTER_IR, ZH_TTS_IR]
ov_models = [core.read_model(ov_path) for ov_path in paths]
ov_en_tts, ov_voice_conversion, ov_zh_tts = ov_models
print("load openvino model finish")

print("patch english speaker infer...")
en_tts_model.model.infer = get_pathched_infer(ov_en_tts, ov_device)
print("patch english speaker infer finish")
print("patch tone convert infer...")
tone_color_converter.model.voice_conversion = get_patched_voice_conversion(
    ov_voice_conversion, ov_device)
print("patch tone convert infer finish")
print("patch chinese speaker infer...")
zh_tts_model.model.infer = get_pathched_infer(
    ov_zh_tts, ov_device
)
print("patch chinese speaker infer finish")

# 启动fastapi
app = FastAPI()

# 存放说话人的位置
speaker_wav_dir = os.path.join(output_dir, "speaker")
if not os.path.exists(speaker_wav_dir):
    os.mkdir(speaker_wav_dir)
# 存放临时生成的cache的文件
cache_dir = os.path.join(output_dir, "cache")
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)
# 加一个默认说话人HuTao
print("[INFO] begin get default se...")
default_audio_path = os.path.join(speaker_wav_dir, "HuTao", "speaker.wav")
default_se_output_dir = os.path.join(cache_dir, "default")
if not os.path.exists(default_se_output_dir):
    os.mkdir(default_se_output_dir)
default_se, default_name = se_extractor.get_se(
    default_audio_path,
    tone_color_converter,
    target_dir=default_se_output_dir,
    vad=True
)
print("[INFO] get default se ok")
# 缓存se
se_cache_dict = {default_audio_path: default_se}


def delete_folder(folder_path):
    """
    删除指定的文件夹及其所有内容。

    参数:
        folder_path (str): 要删除的文件夹路径。

    返回:
        None
    """
    try:
        # 检查文件夹是否存在
        if os.path.exists(folder_path):
            # 删除文件夹及其所有内容
            shutil.rmtree(folder_path)
            # print(f"Folder '{folder_path}' and all its contents have been successfully deleted.")
        else:
            print(f"The folder '{folder_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied to delete the folder '{folder_path}'.")
    except Exception as e:
        print(f"An error occurred while deleting the folder: {e}")


def predict(
    language: str,
    prompt: str,
    audio_file_path: str,
    audio_output_dir: str,
    se_output_dir: str
):
    st = time.time()
    supported_languages = ["zh", "en"]
    if language not in supported_languages:
        text_hint = "not supported language, only support {}".format(", ".join(supported_languages))
        return {
            "status": "failed",
            "file_path": None,
            "message": text_hint
        }
    if language == "zh":
        tts_model = zh_tts_model
        source_se = zh_source_se
        # language = "Chinese"
        speaker_id = 1
    else:
        tts_model = en_tts_model
        source_se = en_source_se
        # language = "English"
        speaker_id = 0
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
    # try:
    # 获取音色（加一个缓存）
    print("[INFO] Begin get tone...")
    if audio_file_path in se_cache_dict:
        target_se = se_cache_dict[audio_file_path]
    else:
        target_se, audio_name = se_extractor.get_se(
            audio_file_path,
            tone_color_converter,
            target_dir=se_output_dir,
            vad=True
        )
        se_cache_dict[audio_file_path] = target_se
    et1 = time.time()
    print("[INFO] get tone duration: ", et1 - st)
    audio_tts_path = f"{audio_output_dir}/tts.wav"
    # 文本转语音
    print("[INFO] Begin TTS...")
    tts_model.tts_to_file(prompt, speaker_id, audio_tts_path)
    et2 = time.time()
    print("[INFO] TTS duration: ", et2 - st)
    save_path = f"{audio_output_dir}/output.wav"
    # 下面这一行应该是固定的
    encode_message = "@MyShell"
    print("[INFO] Begin tone color clone...")
    # 音色克隆
    tone_color_converter.convert(
        audio_src_path=audio_tts_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message,
    )
    et3 = time.time()
    print("[INFO] tone color clone duration: ", et3 - st)
    text_hint = "Get response successfully \n"
    return {
        "status": "success",
        "file_path": save_path,
        "message": text_hint
    }
    # except Exception as e:
    #     text_hint = f"[ERROR] Get target tone color error {str(e)} \n"
    #     return {
    #         "status": "failed",
    #         "file_path": None,
    #         "message": text_hint
    #    }




def is_english_or_digit(s):
    pattern = re.compile(r'^[a-zA-Z0-9]*$')
    if pattern.fullmatch(s):
        return True
    else:
        return False


@app.post("/api/upload")
async def api_upload(speaker_id: str, file: UploadFile = File(...)):
    """
    用于上传文件<br>
    :param speaker_id: 说话人的id, 支持人名，称呼，只支持英文和数字<br>
    :param file: 说话人的录音文件，只支持.wav文件
    """
    if os.path.splitext(file.filename)[-1].lower() != ".wav":
        return {"status": "failed", "data": "only .wav file can upload"}
    print("speaker_id", speaker_id)
    if not is_english_or_digit(speaker_id):
        return {"status": "failed", "data": "speaker_id only support english"}
    temp_dir = os.path.join(speaker_wav_dir, speaker_id)
    is_new = False
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
        is_new = True
    file_path = os.path.join(temp_dir, "speaker.wav")
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    if is_new:
        msg = "data upload OK!"
    else:
        msg = "data update OK!"
    return {"status": "success", "data": msg}


class TTSData(BaseModel):
    prompt: str
    speaker_id: str = "HuTao"
    language: str = 'zh'
    style: str = "default"


@app.post("/api/tts")
def api_tts(data: TTSData):
    """
    声音克隆，文字转语音<br>
    :param data:{<br>
        "prompt" 想要AI输出的最终文本。<br>
        "language": 输出语言是？建议选择'zh', 也就是中文<br>
        "style": 说话风格, 中文不支持说话风格修改<br>
    }<br>
    """
    speaker_wav_path = os.path.join(
        speaker_wav_dir,
        data.speaker_id,
        "speaker.wav"
    )
    if not os.path.exists(speaker_wav_path):
        return {
            "status": "failed",
            "data": "file not exists, you need upload file before tts",
        }
    random_seed = random.randint(10000, 99999)
    audio_output_dir = os.path.join(cache_dir, str(random_seed) + "_audio")
    se_output_dir = os.path.join(cache_dir, str(random_seed) + "_se")
    if not os.path.exists(audio_output_dir):
        os.mkdir(audio_output_dir)
    if not os.path.exists(se_output_dir):
        os.mkdir(se_output_dir)
    result_data = predict(
        language=data.language,
        prompt=data.prompt,
        audio_file_path=speaker_wav_path,
        audio_output_dir=audio_output_dir,
        se_output_dir=se_output_dir
    )
    # 清除se目录缓存
    delete_folder(se_output_dir)
    if result_data["status"] == "success":
        speaker_output_path = result_data["file_path"]
        with open(speaker_output_path, "rb") as f2:
            binary_data = f2.read()
        output = BytesIO(binary_data)
        file_name = data.speaker_id + "_output.wav"
        headers = {
            'Content-Disposition': 'attachment; filename="{}"'.format(file_name)
        }
        # 清理旧数据
        delete_folder(audio_output_dir)
        return StreamingResponse(output, headers=headers)
    else:
        # 清理旧数据
        delete_folder(audio_output_dir)
        return result_data


if __name__ == '__main__':
    uvicorn.run(
        app='api:app', host="127.0.0.1", port=5059, reload=False, workers=1,
    )
    """
    uvicorn api:app --host 127.0.0.1  --port 5059 --workers 1
    """





