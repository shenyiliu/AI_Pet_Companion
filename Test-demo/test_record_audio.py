import pyaudio
import wave
import threading
import time
from queue import Queue, Empty

# 添加全局变量
audio_path_queue = Queue()

# 配置
FORMAT = pyaudio.paInt16  # 采样格式
CHANNELS = 1              # 单声道
RATE = 16000              # 采样率（Hz）
CHUNK = 1024              # 每个缓冲区的大小
RECORD_SECONDS = 5        # 默认录制时长，若没有提供flage时，会录制5秒
OUTPUT_FILENAME = "output.wav"  # 输出音频文件名
# 传入转录的录音线程的控制变量
flage = True

def record_audio():
    global flage
    # 设置pyaudio录音对象
    p = pyaudio.PyAudio()
    
    # 打开麦克风
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("开始录制音频...")
    
    frames = []
    
    # 开始录音，直到flage变为False
    while flage:
        data = stream.read(CHUNK)
        frames.append(data)
    
    # 停止录音
    print("录音结束，保存音频...")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # 保存音频到文件
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    print(f"音频文件保存为: {OUTPUT_FILENAME}")
    # 将文件路径放入队列
    audio_path_queue.put(OUTPUT_FILENAME)

# 启动录音线程
def voice_start_recording():
    global flage
    flage = True
    record_thread = threading.Thread(target=record_audio)
    record_thread.start()
    return record_thread  # 返回线程对象以便后续使用

# 停止录音并获取音频文件路径
def voice_stop_recording():
    global flage
    flage = False
    print("停止录音")
    # 等待获取音频文件路径（设置超时时间为5秒）
    try:
        audio_path = audio_path_queue.get(timeout=2)
        return audio_path
    except Empty:
        print("获取音频文件路径超时")
        return None

if __name__ == "__main__":
    voice_start_recording()
    
    # 设置录制时长或使用外部事件控制
    time.sleep(10)  # 假设录音5秒后停止
    
    voice_stop_recording()
