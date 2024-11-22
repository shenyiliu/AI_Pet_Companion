# TTS openvoice API
def play_audio_stream(context: str):
    
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
        
    # 清理
    pygame.mixer.quit() 


play_audio_stream("今天天气真好呀，我想和你一起出去玩")