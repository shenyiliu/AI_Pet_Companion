import librosa
import uvicorn
import time
import os
import re
import argparse
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from fastapi import FastAPI,Body
from funasr import AutoModel
import threading
from utils import init_LLM, AgentLLM
import requests
import pyaudio
import sys
import logging
import mem0_utils as mu
from typing import Dict

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

# 添加新的全局变量
detected_wake_word = False
audio_buffer = []  # 用于存储检测到唤醒词后的音频
audio_queue = []  # 用于存储待处理的音频片段
processing_thread = None  # 新增音频处理线程

# 加载音频文件并调整采样率
def load_audio(file_path, sample_rate=16000):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio

# 添加一个函数来检查唤醒词
def check_wake_word(transcription: str) -> bool:
    """检查文本中是否包含唤醒词，支持模糊匹配"""
    wake_words = ["胡桃胡桃", "胡桃", "胡套", "护套", "糊桃", "呼桃"]
    transcription = transcription.lower()  # 转换为小写
    print(f"当前识别文本: {transcription}")  # 调试信息
    
    for word in wake_words:
        if word in transcription:
            return True
    return False

# 录制麦克风声音
def record_microphone():
    """修改后的录音函数"""
    global is_recording, detected_wake_word, audio_buffer, audio_queue
    recording = []

    while is_recording:
        # 录制较短的音频片段（如0.5秒）提高响应速度
        duration = 1  # 缩短单次录制时间
        data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()
        
        if not detected_wake_word:
            audio_queue.append(data)  # 将音频片段添加到处理队列
        else:
            # 检测到唤醒词后，直接保存录音
            recording.append(data)

    # 保存完整的录音
    if recording:
        audio_data = np.concatenate(recording, axis=0)
        os.makedirs(os.path.dirname(mic_output_file), exist_ok=True)
        write(mic_output_file, sample_rate, audio_data)
        print(f"麦克风录音保存到: {mic_output_file}")

# 使用正则表达式提取第二个<|...|>中的内容
def extract_second_tag(text):
    '''过滤转录的情绪标签'''
    pattern = r'<\|[^|]*\|><\|([^|]*)\|>'
    match = re.search(pattern, text)
    if match:
        return match.group(1)  # 返回第一个捕获组的内容
    return None

# 转录函数
def transcribe_audio(audio_file):
    global save_transcription_messages
    try:
        print("开始转录...")
        # 转录的结果
        response = create_Transcription(audio_file, "auto")
        print(f"response:{response}")

        emotion ="情绪：" + extract_second_tag(response)
        print(emotion)  # 输出: ANGRY    

        response =re.sub(r"<[^>]*>", "", response) 

        print("Transcription result:", response)

        if not save_transcription_messages:
            save_transcription_messages = ''
        save_transcription_messages += response

        return response
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


def LLM(query:str):
    '''openVINO LLM接口'''
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

def process_audio():
    """处理音频的独立线程函数"""
    global detected_wake_word, audio_queue
    
    # 确保audio目录存在
    temp_wake_word_path = os.path.join("audio", "temp_wake_word.wav")
    os.makedirs("audio", exist_ok=True)
    
    while is_recording:
        if len(audio_queue) > 1:
            # 获取积累的音频片段进行处理
            temp_audio = np.concatenate(audio_queue, axis=0)
            audio_queue.clear()  # 清空队列
            
            if not detected_wake_word:
                write(temp_wake_word_path, sample_rate, temp_audio)
                try:
                    transcription = create_Transcription(temp_wake_word_path, "auto")
                    print(f"唤醒词检测中... 当前音频长度: {len(temp_audio)/sample_rate}秒")
                    
                    if check_wake_word(transcription):
                        print("检测到唤醒词！准备开始记录对话...")
                        detected_wake_word = True
                except Exception as e:
                    print(f"唤醒词检测出错: {e}")
        #time.sleep(0.1)  # 短暂休眠避免CPU过度使用

# 启动录音
@app.post("/start_recording/")
def start_recording():
    global is_recording, recording_thread, processing_thread, save_transcription_messages, detected_wake_word, audio_queue
    
    save_transcription_messages = ''
    detected_wake_word = False
    audio_queue = []  # 重置音频队列

    # 判断mic_output_file文件是否存在，如果存在则删除
    if os.path.exists(mic_output_file):
        os.remove(mic_output_file)

    if not is_recording:
        is_recording = True
        # 启动录音线程
        recording_thread = threading.Thread(target=record_microphone)
        # 启动处理线程
        processing_thread = threading.Thread(target=process_audio)
        
        recording_thread.start()
        processing_thread.start()
        return {"message": "Recording started."}
    else:
        return {"message": "Recording is already in progress."}



# 停止录音并转录
@app.post("/stop_recording/")
def stop_recording(request_data: Dict = Body(...)):
    global is_recording, recording_thread, save_transcription_messages

    # 获取参数user_id
    user_id = request_data.get("user_id", "john")

    if is_recording:
        is_recording = False
        recording_thread.join()  # 等待录音线程结束
        response = transcribe_audio(mic_output_file)  # 执行转录

        # 调用openvino LLM逻辑
        # context = LLM(save_transcription_messages)

        # 检索上下文
        # context = mu.retrieve_context_with_timing(save_transcription_messages, user_id)

        # # 收集完整响应
        # start_time_generate = time.time()
        # # 收集完整响应
        
        # for chunk in mu.generate_response(save_transcription_messages, context):
        #     print(chunk, end="", flush=True)  # 实时打印
        #     response += chunk
        # end_time_generate = time.time()
        # print(f"\nollama LLM耗时: {end_time_generate - start_time_generate} 秒")
        
        # 检查response是否为None
        if response is None:
            response = "未能获取有效的转录内容。"
            
        # 调用TTS
        # TTS_stream(response)
        
        return {"message": "Recording stopped.", "transcription": response}
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
    # 先屏蔽掉openvino的加载
    # check_dir(qwen_ov_model_dir)

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
    # init_LLM(qwen_ov_model_dir)
    
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
