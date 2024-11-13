import librosa
import uvicorn
import os
import re
import argparse
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from fastapi import FastAPI
from funasr import AutoModel
import threading
from utils import init_LLM, AgentLLM
import requests
import pyaudio
import sys
import logging

# 设置日志级别，看更多细节
logging.basicConfig(level=logging.DEBUG)

# 创建FastAPI应用
app = FastAPI()

# 录音参数
sample_rate = 16000  # 采样率
mic_output_file = "audio/mic_output.wav"

# 保存转录信息
save_transcription_messages = None
recording_thread = None
is_recording = False

# 加载音频文件并调整采样率
def load_audio(file_path, sample_rate=16000):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio

# 录制麦克风声音
def record_microphone():
    global is_recording
    recording = []

    while is_recording:
        # 录制音频
        data = sd.rec(int(sample_rate * 2), samplerate=sample_rate, channels=1, dtype=np.int16)
        sd.wait()  # 等待录制完成
        recording.append(data)

    # 将录音合并并保存
    audio_data = np.concatenate(recording, axis=0)
    
    # 如果没有audio文件夹的话就创建一个
    os.makedirs(os.path.dirname(mic_output_file), exist_ok=True)
    
    write(mic_output_file, sample_rate, audio_data)
    print(f"Microphone recording saved to {mic_output_file}")

# 转录函数
def transcribe_audio(audio_file):
    global save_transcription_messages
    try:
        print("开始转录...")
        # 转录的结果
        response = create_Transcription(audio_file, "auto")
        response = re.sub(r"<[^>]*>", "", response)

        print("Transcription result:", response)

        if not save_transcription_messages:
            save_transcription_messages = ''
        save_transcription_messages += response
    except Exception as e:
        print("Transcription failed:", e)

# 处理音频转换为文字的请求
def create_Transcription(audioFile, language):
    global model
    try:
        res = model.generate(
            input=audioFile,
            cache={},
            language=language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        transcription = res[0]["text"]
        return transcription
    except Exception as e:
        return ""

# openVINO LLM接口
def LLM(query:str):
    
    # query = "设置屏幕亮度为50"
    #query = "你能帮我做什么呀？"
    #query = "深圳一日游"

    response = AgentLLM(query)
    return response

# TTS GPT-SOVIS API
def TTS_stream(context:str):
	# 流式传输音频的URL，你可以自由改成Post
	#stream_url = 'http://127.0.0.1:5000/tts?text=这是一段测试文本，旨在通过多种语言风格和复杂性的内容来全面检验文本到语音系统的性能。接下来，我们会探索各种主题和语言结构，包括文学引用、技术性描述、日常会话以及诗歌等。首先，让我们从一段简单的描述性文本开始：“在一个阳光明媚的下午，一位年轻的旅者站在山顶上，眺望着下方那宽广而繁忙的城市。他的心中充满了对未来的憧憬和对旅途的期待。”这段文本测试了系统对自然景观描写的处理能力和情感表达的细腻程度。&stream=true'
    # 流式传输音频的URL
    stream_url = f'http://127.0.0.1:5000/tts?text={context}&stream=true'

    # 初始化pyaudio
    p = pyaudio.PyAudio()

    try:
        # 打开音频流
        stream = p.open(format=p.get_format_from_width(2),
                        channels=1,
                        rate=32000,
                        output=True)

        # 使用requests获取音频流
        response = requests.get(stream_url, stream=True)

        # 检查响应状态码
        response.raise_for_status()

        # 读取数据块并播放
        for data in response.iter_content(chunk_size=1024):
            stream.write(data)

    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
    except pyaudio.paError as e:
        print(f"音频错误: {e}")
    except Exception as e:
        print(f"其他错误: {e}")
    finally:
        # 停止和关闭流
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()

        # 终止pyaudio
        p.terminate()


# 启动录音
@app.post("/start_recording/")
def start_recording():
    global is_recording, recording_thread,save_transcription_messages
    
    save_transcription_messages = ''

    # 判断mic_output_file文件是否存在，如果存在则删除
    if os.path.exists(mic_output_file):
        os.remove(mic_output_file)

    if not is_recording:
        is_recording = True
        recording_thread = threading.Thread(target=record_microphone)
        recording_thread.start()
        return {"message": "Recording started."}
    else:
        return {"message": "Recording is already in progress."}

# 停止录音并转录
@app.post("/stop_recording/")
def stop_recording():
    global is_recording, recording_thread, save_transcription_messages
    if is_recording:
        is_recording = False
        recording_thread.join()  # 等待录音线程结束
        transcribe_audio(mic_output_file)  # 执行转录
        
        # 调用LLM逻辑
        context = LLM(save_transcription_messages)
        
        # 检查context是否为None
        if context is None:
            context = "未能获取有效的转录内容。"
            
        # 调用TTS
        TTS_stream(context)
        
        
        return {"message": "Recording stopped.", "transcription": context}
    else:
        return {"message": "No recording in progress."}


# 主函数入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simplified model')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='ip号')
    parser.add_argument('--port', type=int, default=8081, help='端口号')
    args = parser.parse_args()
    now_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(now_dir)
    download_dir = os.path.join(project_dir, "download")
    output_dir = os.path.join(project_dir, "output")
    def check_dir(dir1: str):
        if not os.path.exists(dir1) or len(os.listdir(dir1)) == 0:
            raise Exception(f"{dir1} not exists")
    asr_model_dir = os.path.join(download_dir, "SenseVoiceSmall")
    check_dir(asr_model_dir)
    vad_model_dir = os.path.join(
        download_dir, "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    )
    check_dir(vad_model_dir)
    qwen_ov_model_dir = os.path.join(
        output_dir, "ov-qwen2.5-7b-instruct-int4"
    )
    check_dir(qwen_ov_model_dir)

    # 添加调试信息
    print("当前工作目录:", os.getcwd())
    print("ASR模型路径:", asr_model_dir)
    print("VAD模型路径:", vad_model_dir)
    print("LLM模型路径:", qwen_ov_model_dir)

    # 使用模型时
    try:
        print("正在加载模型...")
        model = AutoModel(
            model=asr_model_dir,
            model_revision="",
            vad_model=vad_model_dir,
            vad_model_revision="",
            device="cpu",
            use_offline=True,
            use_vad=True,
            model_hub="",
            disable_update=True,  # 禁用更新检查
            local_files_only=True  # 强制只使用本地文件
        )
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        import traceback
        print(traceback.format_exc())  # 打印详细错误信息
        raise

    # 初始化LLM
    init_LLM(qwen_ov_model_dir)
    
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
