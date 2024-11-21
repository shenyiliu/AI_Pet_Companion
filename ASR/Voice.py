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
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from collections import deque

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
audio_queue = []  # 用于存储待处理的音频片段
processing_thread = None  # 新增音频处理线程
last_speech_time = None  # 用于记录最后一次检测到语音的时间
SILENCE_THRESHOLD = 2  # 无声检测阈值（秒）
MAX_IDLE_TIME = 60  # 最大空闲时间（秒）

# 添加TTS播放状态标识
tts_playing = False

# 添加全局队列
tts_queue = Queue()
tts_thread = None
is_tts_running = False

# 加载音频文件并调整采样率
def load_audio(file_path, sample_rate=16000):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio

# 添加一个函数来检查唤醒词
def check_wake_word(transcription: str) -> bool:
    """检查文本中是否包含唤醒词，支持模糊匹配"""
    wake_words = ["胡桃胡桃", "胡桃", "胡套", "护套", "糊桃", "呼桃","胡涛"]
    transcription = transcription.lower()  # 转换为小写
    print(f"当前识别文本: {transcription}")  # 调试信息
    
    for word in wake_words:
        if word in transcription:
            return True
    return False

# 录制麦克风声音
def record_microphone():
    """修改后的录音函数"""
    global is_recording, detected_wake_word, audio_queue, tts_playing
    recording = []

    while is_recording:
        if tts_playing:
            time.sleep(0.1)
            continue  # 跳过录音处理

        # 录制较短的音频片段（如1.5秒）提高响应速度
        duration = 1.5  # 缩短单次录制时间
        data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()
        
        # 不管是否检测到唤醒词，都将音频添加到队列中
        audio_queue.append(data)
        
        # 如果检测到唤醒词，同时保存到recording中
        if detected_wake_word:
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
        #print(f"response:{response}")

        emotion ="情绪：" + extract_second_tag(response)
        print(emotion)  # 输出: ANGRY    

        response =re.sub(r"<[^>]*>", "", response) 

        print("Transcription result:", response)

        if not save_transcription_messages:
            save_transcription_messages = ''
        save_transcription_messages += response

        return emotion +","+ response
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
def TTS_stream(context: str):
    global tts_playing
    tts_playing = True  # 开始播放TTS

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
        tts_playing = False  # 播放结束

# TTS openvoice API
def TTS_play_audio_stream(context: str):
    global tts_playing
    tts_playing = True  # 开始播放TTS

    import requests
    import io
    import pygame

    # 发送POST请求到TTS接口
    url = "http://127.0.0.1:5059/api/tts"
    data = {
        "prompt": context,
        "speaker_id": "HuTao",
        "language": "zh",
        "style": "default"
    }
    
    response = requests.post(url, json=data)
    
    # 将响应内容转换为字节流
    audio_stream = io.BytesIO(response.content)
    
    # 初始化pygame混音器
    pygame.mixer.init()
    
    # 加载并播放音频
    pygame.mixer.music.load(audio_stream)
    pygame.mixer.music.play()
    
    # 等待播放完成
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    tts_playing = False  # 播放结束    
    # 清理
    pygame.mixer.quit() 

def process_audio():
    """处理音频的独立线程函数"""
    global detected_wake_word, audio_queue, last_speech_time, is_recording
    
    # 确保audio目录存在
    temp_wake_word_path = os.path.join("audio", "temp_wake_word.wav")
    os.makedirs("audio", exist_ok=True)
    
    conversation_buffer = []  # 用于存储当前对话的音频数据
    all_transcriptions = []  # 存储所有对话的转录结果
    
    while is_recording:
        if len(audio_queue) > 0:
            temp_audio = audio_queue.pop(0)
            
            # 检查是否需要重置唤醒状态
            if detected_wake_word and last_speech_time and time.time() - last_speech_time > MAX_IDLE_TIME:
                print("超过1分钟没有对话，需要重新唤醒")
                detected_wake_word = False
                conversation_buffer = []
                last_speech_time = None
            
            if not detected_wake_word:
                # 唤醒词检测逻辑保持不变
                write(temp_wake_word_path, sample_rate, temp_audio)
                try:
                    transcription = create_Transcription(temp_wake_word_path, "zh")
                    print(f"唤醒词检测中... 当前音频长度: {len(temp_audio)/sample_rate}秒")
                    
                    if check_wake_word(transcription):
                        print("检测到唤醒词！准备开始对话...")
                        detected_wake_word = True
                        last_speech_time = time.time()
                        audio_queue.clear()  # 清空之前的音频
                except Exception as e:
                    print(f"唤醒词检测出错: {e}")
            else:
                # 检测到唤醒词后的对话处理
                write(temp_wake_word_path, sample_rate, temp_audio)
                try:
                    transcription = create_Transcription(temp_wake_word_path, "zh")
                    if transcription.strip():  # 如果有识别到文字
                        print(f"检测到语音输入: {transcription}")
                        last_speech_time = time.time()
                        conversation_buffer.append(temp_audio)
                    elif time.time() - last_speech_time > SILENCE_THRESHOLD:
                        # 检测到静音，处理当前对话
                        if conversation_buffer and len(conversation_buffer) > 0:  # 确保有实际的对话内容
                            audio_data = np.concatenate(conversation_buffer, axis=0)
                            temp_conversation_path = os.path.join("audio", f"conversation_{len(all_transcriptions)}.wav")
                            write(temp_conversation_path, sample_rate, audio_data)
                            
                            # 转录当前对话
                            current_transcription = transcribe_audio(temp_conversation_path)
                            if current_transcription and current_transcription.strip():  # 确保转录结果不为空
                                all_transcriptions.append(current_transcription)
                                print(f"当前对话转录完成: {current_transcription}")

                                # 这里可以添加对话处理逻辑，比如调用 LLM 或 TTS
                                # 调用Mem0_LLM_TTS
                                Mem0_LLM_TTS(current_transcription,"john")

                            # 清空当前对话缓冲区，准备下一轮对话
                            conversation_buffer = []
                            last_speech_time = time.time()
                            
                except Exception as e:
                    print(f"对话处理出错: {e}")
        
        time.sleep(0.1)
    
    # 返回所有对话的转录结果
    return all_transcriptions


def Mem0_LLM_TTS(current_transcription:str, user_id:str):
    '''
    1.检索上下文
    2.生成LLM响应并流式传输到TTS
    3.调用TTS
    '''
    global is_tts_running, tts_thread, tts_queue
    
    # 检索上下文
    context = mu.retrieve_context_with_timing(current_transcription, user_id)

    # 保存当前对话信息
    mu.save_interaction_timing(user_id, current_transcription, "")
    
    # 启动TTS处理线程
    if tts_thread is None or not tts_thread.is_alive():
        is_tts_running = True
        tts_thread = threading.Thread(target=process_tts_queue)
        tts_thread.start()
    
    # 用于临时存储当前正在构建的句子
    current_sentence = ""
    
    # 收集完整响应
    start_time_generate = time.time()
    
    for chunk in mu.generate_response(current_transcription, context):
        print(chunk, end="", flush=True)  # 实时打印
        
        current_sentence += chunk
        
        # 检查是否有完整的句子
        if any(punct in chunk for punct in ['。', '！', '？', '.', '!', '?']):
            # 将当前句子加入TTS队列
            if current_sentence.strip():
                tts_queue.put(current_sentence.strip())
            current_sentence = ""
    
    # 处理最后可能剩余的不完整句子
    if current_sentence.strip():
        tts_queue.put(current_sentence.strip())
    
    end_time_generate = time.time()
    print(f"\nollama LLM耗时: {end_time_generate - start_time_generate} 秒")

def process_tts_queue():
    """处理TTS队列的线程函数"""
    global is_tts_running, tts_playing
    
    while is_tts_running:
        try:
            # 从队列中获取文本，如果队列为空会等待
            text = tts_queue.get(timeout=1)
            if text:
                # 执行TTS
                # TTS_stream(text)
                TTS_play_audio_stream(text)
                # 标记任务完成
                tts_queue.task_done()
        except Empty:
            continue
        except Exception as e:
            print(f"TTS处理错误: {e}")
            continue

# 启动录音
@app.post("/start_recording/")
def start_recording():
    global is_recording, recording_thread, processing_thread, save_transcription_messages, detected_wake_word, audio_queue
    
    save_transcription_messages = ''
    detected_wake_word = False
    audio_queue = []  # 重置音频队列

    if os.path.exists(mic_output_file):
        os.remove(mic_output_file)

    if not is_recording:
        is_recording = True
        # 启动录音线程
        recording_thread = threading.Thread(target=record_microphone)
        recording_thread.start()
        
        # 使用ThreadPoolExecutor来执行process_audio
        result = None  # 初始化result变量
        with ThreadPoolExecutor() as executor:
            future = executor.submit(process_audio)
            try:
                # 获取所有对话的转录结果
                conversations = future.result()
                print(f"对话记录: {conversations}")
                
                # 这里可以添加对所有对话的后处理逻辑
                # process_all_conversations(conversations)
                
                return {
                    "message": "Recording completed",
                    "conversations": conversations or []
                }
            except Exception as e:
                print(f"处理音频时出错: {e}")
                return {"message": "Error processing audio", "error": str(e)}
    else:
        return {"message": "Recording is already in progress."}



# 停止录音
@app.post("/stop_recording/")
def stop_recording(request_data: Dict = Body(...)):
    global is_recording, recording_thread, is_tts_running, tts_thread

    # 获取参数user_id
    user_id = request_data.get("user_id", "john")

    # 保存所有的对话信息到向量数据库中
    mu.save_interaction_to_vector_store_timing()

    if is_recording:
        is_recording = False
        recording_thread.join()  # 等待录音线程结束
        
        # 停止TTS处理线程
        if tts_thread and tts_thread.is_alive():
            is_tts_running = False
            tts_thread.join()
            
        return {"message": "Recording stopped."}
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
