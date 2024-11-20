'''
1.调整音量
2.调整亮度
3.打开/关闭摄像头
4.打开/关闭实时字幕、
5.打开/关闭任务管理器
6.截图当前窗口并保存到桌面
7.获取系统基本信息
比如 CPU、内存、磁盘使用情况。
8.检测电池状态
适用于笔记本电脑，查询电池电量和剩余使用时间。
'''

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def Set_volume(volume_level:str):
    """
    设置 Windows 系统音量
    :param volume_level: 音量大小(1-100的整数)
    :return: None
    """
    try:
        volume_level = int(volume_level)
        # 验证输入值是否在有效范围内
        if not isinstance(volume_level, int) or volume_level < 0 or volume_level > 100:
            raise ValueError("音量必须是0-100之间的整数")
        
        # 获取音频设备
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        
        # 将0-100的值转换为-65.25到0的值(pycaw使用的范围)
        volume_scalar = volume_level / 100.0
        # 设置音量
        volume.SetMasterVolumeLevelScalar(volume_scalar, None)
        
        print(f"系统音量已设置为: {volume_level}%")
        
    except Exception as e:
        print(f"设置音量时出错: {str(e)}")

if __name__ == "__main__":
    # 1.测试控制音量函数
    # Set_volume("100")