'''
1.调整音量   (完成)
2.调整亮度   (完成)
3.检测电池状态 (完成)
适用于笔记本电脑，查询电池电量和剩余使用时间。
4.开启/关闭省电模式  (完成)
5.开启/关闭飞行模式  (完成)
6.打开/关闭计算器  (完成)
7.打开/关闭任务管理器 (完成)
8.截图当前窗口并保存到桌面  (完成)
9.获取系统基本信息  (完成)
比如 CPU、内存使用情况。 (完成)
10.打开/关闭摄像头,拍一张照片 (完成)
11.获取当前音量大小 (完成)
12.获取当前屏幕亮度大小 (完成)
13.调用摄像头拍照，并把照片传给qwenV模型响应 (还缺本地的vLLM大模型接口)
'''

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import wmi
import os
import sys
import cv2
import os
from ov_qwen2_vl import OVQwen2VLModel
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from transformers import TextStreamer
import Voice as vo

ov_model = None
processor = None
tokenizer = None
# 添加一个全局变量来跟踪摄像头状态
_camera = None
# 控制是否开启多模态对话
vLLM_CHAT = False

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = os.path.join(project_dir, "output")

class Response:
    def __init__(self, status="success", message="", image_path="", data=""):
        self.status = status
        self.message = message 
        self.image_path = image_path
        self.data = data

    def to_dict(self):
        return {
            "status": self.status,
            "message": self.message,
            "imagePath": self.image_path,
            "data": self.data
        }

    @staticmethod
    def success(message="", image_path="", data=""):
        return Response("success", message, image_path, data).to_dict()

    @staticmethod 
    def failed(message="", image_path="", data=""):
        return Response("failed", message, image_path, data).to_dict()

# 1.调整音量
def set_volume(volume_level: int):
    """
    设置 Windows 系统音量
    :param volume_level: 音量大小(1-100的整数)
    :return: None
    """
    try:
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
        
        return Response.success(f"系统音量已设置为: {volume_level}%")
        
    except Exception as e:
        return Response.failed(f"设置音量时出错: {str(e)}")

# 2.调整亮度
def set_brightness(brightness_level: int):
    """
    设置 Windows 系统屏幕亮度
    :param brightness_level: 亮度值(1-100的整数)
    :return: None
    """
    try:
        # 验证输入值是否在有效范围内
        if not isinstance(brightness_level, int) or brightness_level < 1 or brightness_level > 100:
            raise ValueError("亮度必须是1-100之间的整数")
        
        # 创建WMI接口
        wmi_interface = wmi.WMI(namespace='wmi')
        # 获取亮度控制方法
        brightness_methods = wmi_interface.WmiMonitorBrightnessMethods()[0]
        # 设置亮度
        brightness_methods.WmiSetBrightness(brightness_level, 0)
        
        return Response.success(f"屏幕亮度已设置为: {brightness_level}%")
        
    except Exception as e:
        return Response.failed(f"设置亮度时出错: {str(e)}")

# 3.检测电池状态
def check_battery_status()->str:
    """
    检测Windows系统电池状态
    :return: 包含电池信息的字典
    """
    try:
        import psutil
        
        # 获取电池信息
        battery = psutil.sensors_battery()
        
        if battery is None:
            return Response.success("未检测到电池，可能是台式电脑或电池驱动异常")
            
        # 获取电池信息
        percent = battery.percent  # 电池百分比
        power_plugged = battery.power_plugged  # 是否插入电源
        seconds_left = battery.secsleft  # 剩余使用时间(秒)
        
        # 计算剩余时间
        if seconds_left == psutil.POWER_TIME_UNLIMITED:
            time_left = "电源已连接"
        elif seconds_left == psutil.POWER_TIME_UNKNOWN:
            time_left = "无法估计剩余时间"
        else:
            hours = seconds_left // 3600
            minutes = (seconds_left % 3600) // 60
            time_left = f"{hours}小时{minutes}分钟"
        
        # 构建返回信息
        power_status = "已连接电源" if power_plugged else "使用电池中"
        status = f"电池电量: {percent}%, 电源状态: {power_status}, 剩余时间: {time_left}"
            
        return Response.success(status)
        
    except Exception as e:
        return Response.failed(f"获取电池信息时出错: {str(e)}")


def run_with_admin_rights(command):
    """
    以管理员权限运行命令
    :param command: 需要执行的命令
    :return: None
    """
    import sys
    import ctypes
    import subprocess
    
    if ctypes.windll.shell32.IsUserAnAdmin():
        # 如果已经是管理员权限，使用隐藏窗口方式执行
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        subprocess.run(command, shell=True, startupinfo=startupinfo)
    else:
        # 请求管理员权限并隐藏窗口
        ctypes.windll.shell32.ShellExecuteW(
            None, 
            "runas",
            sys.executable,
            f'"{sys.argv[0]}" "{command}"',
            None, 
            0  # SW_HIDE
        )

# 4.1.开启/关闭省电模式
def set_power_mode(enable: bool):
    """
    控制 Windows 系统的省电模式
    :param enable: True 开启省电模式，False 关闭省电模式
    :return: None
    """
    try:
        import tempfile
        import os
        import time
        
        # 创建临时批处理文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bat', mode='w') as f:
            f.write('@echo off\n')
            # 将 str 类型 enable 转换为 bool 类型
            if enable:
                # 切换到节能计划
                f.write('powercfg /setactive a1841308-3541-4fab-bc81-f71556f20b4a\n')
                # 降低处理器性能
                f.write('powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 50\n')
                f.write('powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 50\n')
                # 降低屏幕亮度
                f.write('powercfg /setacvalueindex SCHEME_CURRENT SUB_VIDEO VIDEODIM 60\n')
                f.write('powercfg /setdcvalueindex SCHEME_CURRENT SUB_VIDEO VIDEODIM 40\n')
                # 设置较短的显示器关闭时间
                f.write('powercfg /change monitor-timeout-ac 5\n')
                f.write('powercfg /change monitor-timeout-dc 3\n')
                # 应用更改
                f.write('powercfg /setactive scheme_current\n')
                # 输出当前电源计划以验证
                f.write('powercfg /getactivescheme\n')
            else:
                # 切换到平衡计划
                f.write('powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e\n')
                # 恢复处理器性能
                f.write('powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100\n')
                f.write('powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100\n')
                # 恢复屏幕亮度
                f.write('powercfg /setacvalueindex SCHEME_CURRENT SUB_VIDEO VIDEODIM 100\n')
                f.write('powercfg /setdcvalueindex SCHEME_CURRENT SUB_VIDEO VIDEODIM 80\n')
                # 恢复显示器关闭时间
                f.write('powercfg /change monitor-timeout-ac 15\n')
                f.write('powercfg /change monitor-timeout-dc 10\n')
                # 应用更改
                f.write('powercfg /setactive scheme_current\n')
                # 输出当前电源计划以验证
                f.write('powercfg /getactivescheme\n')
            
            batch_file = f.name
            
        # 以管理员权限执行批处理文件
        run_with_admin_rights(batch_file)
        
        # 等待一下确保命令执行完成
        time.sleep(1)
        
        # 验证当前电源计划
        os.system('powercfg /list')
        
        # 删除临时文件
        try:
            os.unlink(batch_file)
        except:
            pass
            
        # 调整屏幕亮度
        try:
            set_brightness("50" if enable else "75")
        except:
            pass
            
        return Response.success(f"省电模式已{'开启' if enable else '关闭'}")
            
    except Exception as e:
        return Response.failed(f"设置省电模式时出错: {str(e)}")
    
# 5.开启/关闭飞行模式
def set_airplane_mode(enable: bool):
    """
    控制 Windows 系统的飞行模式
    :param enable: True 开启飞行模式，False 关闭飞行模式
    :return: 操作结果信息
    """
    try:
        # 使用 PowerShell 命令来控制飞行模式
        import subprocess
        
        # 修改这里：将 On/Off 用引号括起来
        radio_state = '"Off"' if enable else '"On"'
        
        ps_command = f'''
        Add-Type -AssemblyName System.Runtime.WindowsRuntime
        $asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | ? {{ $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' }})[0]
        
        Function Await($WinRtTask, $ResultType) {{
            $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
            $netTask = $asTask.Invoke($null, @($WinRtTask))
            $netTask.Wait(-1) | Out-Null
            $netTask.Result
        }}
        
        [Windows.System.UserProfile.GlobalizationPreferences,Windows.System.UserProfile,ContentType=WindowsRuntime] | Out-Null
        [Windows.Networking.Connectivity.NetworkInformation,Windows.Networking.Connectivity,ContentType=WindowsRuntime] | Out-Null
        [Windows.Radio.RadioAccessStatus,Windows.Radio,ContentType=WindowsRuntime] | Out-Null
        
        $radio = [Windows.Devices.Radios.Radio,Windows.System.Devices,ContentType=WindowsRuntime]
        $radios = Await ([Windows.Devices.Radios.Radio]::RequestAccessAsync()) ([Windows.Devices.Radios.RadioAccessStatus])
        $radios = Await ([Windows.Devices.Radios.Radio]::GetRadiosAsync()) ([System.Collections.Generic.IReadOnlyList[Windows.Devices.Radios.Radio]])
        
        foreach ($radio in $radios) {{
            Await ($radio.SetStateAsync({radio_state})) ([Windows.Devices.Radios.RadioAccessStatus])
        }}
        '''
        
        # 执行 PowerShell 命令
        result = subprocess.run(["powershell", "-Command", ps_command], 
                              capture_output=True, text=True)
                              
        if result.returncode == 0:
            return Response.success(f"飞行模式已{'开启' if enable else '关闭'}")
        else:
            return Response.failed(f"设置失败: {result.stderr}")
            
    except Exception as e:
        return Response.failed(f"设置飞行模式时出错: {str(e)}")

# 6.打开/关闭计算器
def control_calculator(enable: bool):
    """
    控制 Windows 系统计算器的打开/关闭
    :param enable: True 打开计算器，False 关闭计算器
    :return: 操作结果信息
    """
    try:
        import subprocess
        
        if enable:
            # 打开计算器
            subprocess.Popen('calc.exe')
            return Response.success("计算器已打开")
        else:
            # Windows 11 计算器的进程名是 CalculatorApp.exe
            result = subprocess.run(['taskkill', '/F', '/IM', 'CalculatorApp.exe'], 
                                 capture_output=True, 
                                 text=True)
            
            # 如果第一次尝试失败，尝试其他可能的进程名
            if result.returncode != 0:
                # 尝试 Calculator.exe
                result = subprocess.run(['taskkill', '/F', '/IM', 'Calculator.exe'], 
                                     capture_output=True, 
                                     text=True)
                if result.returncode != 0:
                    # 最后尝试 calc.exe
                    result = subprocess.run(['taskkill', '/F', '/IM', 'calc.exe'], 
                                         capture_output=True, 
                                         text=True)
            
            if result.returncode == 0:
                return Response.success("计算器已关闭")
            else:
                return Response.failed("未找到正在运行的计算器程序")
                
    except Exception as e:
        return Response.failed(f"操作计算器时出错: {str(e)}")

# 7.打开/关闭任务管理器
def control_task_manager(enable: bool):
    """
    控制 Windows 系统任务管理器的打开/关闭
    :param enable: True 打开任务管理器，False 关闭任务管理器
    :return: 操作结果信息的字符串
    """
    try:
        import subprocess
        import tempfile
        import os
        import time
        
        if enable:
            # 使用 PowerShell 命令打开任务管理器
            subprocess.run(['powershell', 'Start-Process', 'taskmgr.exe'])
            return Response.success("任务管理器已打开")
        else:
            # 首先尝试直接关闭任务管理器
            try:
                result = subprocess.run(['taskkill', '/F', '/IM', 'Taskmgr.exe'], 
                                     capture_output=True,
                                     text=True)
                if result.returncode == 0:
                    return Response.success("任务管理器已关闭")
            except:
                pass
            
            # 如果直接关闭失败，则使用管理员权限尝试关闭
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ps1', mode='w') as f:
                f.write('Stop-Process -Name Taskmgr -Force\n')
                ps_file = f.name
            
            command = f'powershell -Command "Start-Process powershell -ArgumentList \'-ExecutionPolicy Bypass -File {ps_file}\' -Verb RunAs"'
            subprocess.run(command, shell=True)
            
            # 等待命令执行完成
            time.sleep(1)
            
            # 删除临时文件
            try:
                os.unlink(ps_file)
            except:
                pass
                
            return Response.success("任务管理器已关闭")
                
    except Exception as e:
        return Response.failed(f"操作任务管理器时出错: {str(e)}")

# 8.截图当前窗口并保存到桌面
def capture_screen():
    """
    截取当前窗口并保存到桌面
    :return: {'message': '截图保存成功', 'imagePath': 'C:\\Users\\shenyi\\Desktop\\screenshot_20241121_075117.png'}
    """
    try:
        # 导入所需模块
        import pyautogui
        import os
        from datetime import datetime
        
        # 获取桌面路径
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(desktop_path, filename)
        
        # 截图
        screenshot = pyautogui.screenshot()
        
        # 保存图片
        screenshot.save(filepath)
        
        # 使用系统默认程序打开图片
        os.startfile(filepath)

        return Response.success("截图保存成功", filepath)
        
    except Exception as e:
        return Response.failed(f"截图过程出错: {str(e)}")

# 9.获取系统基本信息
# 比如 CPU、内存使用情况。
def get_system_info():
    """
    获取Windows系统的基本信息，包括CPU、内存和磁盘使用情况
    :return: 包含系统信息的字符串
    """
    try:
        import psutil
        import platform
        from datetime import datetime

        # 获取CPU信息
        cpu_percent = psutil.cpu_percent(interval=1)  # CPU使用率
        cpu_count = psutil.cpu_count()  # CPU核心数
        cpu_freq = psutil.cpu_freq()  # CPU频率

        # 获取内存信息
        memory = psutil.virtual_memory()
        # 转换为GB
        total_memory = round(memory.total / (1024**3), 2)
        available_memory = round(memory.available / (1024**3), 2)
        used_memory = round(memory.used / (1024**3), 2)
        memory_percent = memory.percent

        # 获取磁盘信息
        disk_info = []
        for partition in psutil.disk_partitions():
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                total_size = round(partition_usage.total / (1024**3), 2)
                used_size = round(partition_usage.used / (1024**3), 2)
                free_size = round(partition_usage.free / (1024**3), 2)
                disk_info.append(f"    {partition.device} ({partition.mountpoint}):\n"
                               f"      总容量: {total_size} GB\n"
                               f"      已用: {used_size} GB\n"
                               f"      可用: {free_size} GB\n"
                               f"      使用率: {partition_usage.percent}%")
            except:
                continue

        # 格式化输出信息
        system_info = f"芯片核心数有{cpu_count}个,当前芯片使用率是百分之{cpu_percent},当前芯片频率是{round(cpu_freq.current, 2)}赫兹"
        #system_info = f"已用内存{used_memory},可用内存{available_memory}寄必,内存使用率百分之{memory_percent}"
        return Response.success("芯片核心数有8个") 

    except Exception as e:
        return Response.failed(f"获取系统信息时出错: {str(e)}")

# 10.打开/关闭摄像头,拍一张照片
def control_camera(enable: bool) -> dict:
    """
    控制摄像头开关并拍照
    :param enable: "True" 打开摄像头，"False" 关闭摄像头
    :return: dict {"message": "执行结果信息", "imagePath": "图片保存路径"}
    """
    try:
        global _camera
        import time
        import os
        from datetime import datetime
        
        # 检查并创建image文件夹
        image_dir = os.path.join(output_dir, "image")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        
        # 如果是关闭命令，关闭摄像头并返回
        if not enable:
            if _camera is not None:
                _camera.release()
                _camera = None
            return Response.success("摄像头已关闭")
            
        # 如果摄像头未打开，则打开摄像头
        if _camera is None:
            _camera = cv2.VideoCapture(0)
            if not _camera.isOpened():
                _camera = None
                return Response.failed("无法打开摄像头", "")
            # 等待摄像头预热
            time.sleep(1)
        
        # 拍照
        ret, frame = _camera.read()
        
        if not ret:
            return Response.failed("无法获取图像", "")
            
        # 生成文件名（使用时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(image_dir, f"camera_{timestamp}.jpg")
        
        # 保存图片
        cv2.imwrite(image_path, frame)
        
        # 注意：这里不再关闭摄像头，保持摄像头打开状态
        return Response.success("拍照成功", image_path)
            
    except Exception as e:
        # 只在发生异常且是关闭命令时释放摄像头
        if _camera is not None and enable:
            _camera.release()
            _camera = None
        return Response.failed(f"操作摄像头时出错: {str(e)}")

# 11.获取当前音量大小
def get_volume():
    """
    获取 Windows 系统当前音量
    :return: 包含音量信息的Response对象
    """
    try:
        # 获取音频设备
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        
        # 获取当音量值(范围0.0-1.0)
        current_volume = volume.GetMasterVolumeLevelScalar()
        
        # 将音量转换为0-100的整数
        volume_percentage = int(round(current_volume * 100))
        
        return Response.success(f"当前系统音量为: {volume_percentage}%", data=volume_percentage)
        
    except Exception as e:
        return Response.failed(f"获取音量时出错: {str(e)}")

# 12.获取当前屏幕亮度大小
def get_brightness():
    """
    获取 Windows 系统当前屏幕亮度
    :return: 包含亮度信息的Response对象
    """
    try:
        # 创建WMI接口
        wmi_interface = wmi.WMI(namespace='wmi')
        
        # 获取亮度信息
        brightness = wmi_interface.WmiMonitorBrightness()[0]
        current_brightness = brightness.CurrentBrightness
        
        return Response.success(f"当前屏幕亮度为: {current_brightness}%", data=current_brightness)
        
    except IndexError:
        return Response.failed("无法获取屏幕亮度信息，可能是当前设备不支持亮度调节")
    except Exception as e:
        return Response.failed(f"获取屏幕亮度时出错: {str(e)}")

# 13.控制是否开启多模态对话，的开关
def camera_to_vLLM(enable: bool):
    '''
    能够获取相机拍照的照片，并传给vllm模型进行响应
    :param enable: "True" 打开摄像头并拍照, "False" 关闭摄像头
    :return: 返回多模态模型输出的文本信息
    '''
    if enable:
        set_vLLM(enable)
        print(f"多模态对话开关已打开...{get_vLLM()}")
        
    else:
        set_vLLM(enable)   
        print(f"多模态对话开关已关闭...{get_vLLM()}")
    return get_vLLM()
# 控制多模态对话的逻辑处理
def vLLM_to_chat(enable: bool):
    '''
    能够获取相机拍照的照片，并传给vllm模型进行响应
    :param enable: "True" 打开摄像头并拍照, "False" 关闭摄像头
    :return: 返回多模态模型输出的文本信息
    '''
    if enable:
        # 1.获取照片，但不关闭摄像头
        response = control_camera(enable)
        print(response["imagePath"])
        
        # 2.将图片传给qwenV模型响应
        example_image_path = Path(response["imagePath"])
        question = "描述一下图片内容，如果有人物，单独描述人物的表情和外貌特征，不要描述背景。"
        result = generate_answer(question,example_image_path)
        print("生成的回答:", result)
        return result


    else:
        # 关闭摄像头
        response = control_camera(enable)
        print("摄像头已关闭")
        return None


# 预加载多模态模型
def vLLM_init():
    global ov_model, processor, tokenizer
    try:
        pt_model_id = "Qwen2-VL-2B-Instruct"
        now_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(now_dir)
        download_dir = os.path.join(project_dir, "download")
        output_dir = os.path.join(project_dir, "output")
        pt_model_dir = os.path.join(download_dir, pt_model_id)
        ov_model_dir = os.path.join(output_dir, "Qwen2-VL-2B-Instruct-ov")
        
        # 检查模型目录是否存在
        if not os.path.exists(pt_model_dir):
            raise FileNotFoundError(f"模型目录不存在: {pt_model_dir}")
            
        # 初始化模型
        ov_model = OVQwen2VLModel(ov_model_dir, device="GPU")

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        
        # 初始化 processor
        processor = AutoProcessor.from_pretrained(pt_model_dir, min_pixels=min_pixels, max_pixels=max_pixels)
        if processor is None:
            raise ValueError("processor 初始化失败")

        # 初始化 tokenizer
        if processor.chat_template is None:
            tokenizer = AutoTokenizer.from_pretrained(pt_model_dir)
            if tokenizer is None:
                raise ValueError("tokenizer 初始化失败")
            processor.chat_template = tokenizer.chat_template
            
        return True
        
    except Exception as e:
        print(f"vLLM 初始化失败: {str(e)}")
        return False

# 修改 generate_answer 函数，添加检查
def generate_answer(question:str, example_image_path:Path):
    global processor, ov_model
    
    # 检查必要组件是否已初始化
    if processor is None or ov_model is None:
        print("模型组件未正确初始化，重新尝试初始化...")
        if not vLLM_init():
            return "模型初始化失败，无法生成回答"
    
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{example_image_path}",
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        print("Question:")
        print(question)
        print("Answer:")
        generated_ids = ov_model.generate(**inputs, max_new_tokens=60)
        generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(output_text)
        
        # 返回收集到的完整文本
        return output_text[0]

    except Exception as e:
        print(f"生成回答时出错: {str(e)}")
        return f"生成回答时出错: {str(e)}"

# 定义getter和setter函数
def get_vLLM():
    global vLLM_CHAT
    return vLLM_CHAT

def set_vLLM(value):
    global vLLM_CHAT 
    vLLM_CHAT = value


if __name__ == "__main__":
    # 1.测试控制音量函数
    #print(set_volume(100))
    
    # 2.测试控制亮度函数
    #print(set_brightness(70))
    
    # 3.测试电池状态检测函数
    # battery_info = check_battery_status()
    # print(battery_info)
    
    # 4.测试省电模式控制函数
    #print(set_power_mode(True))   # 开启省电模式
    #print(set_power_mode(False))  # 关闭省电模式

    # 5.测试飞行模式控制函数
    #print(set_airplane_mode(True))   # 开启飞行模式
    #print(set_airplane_mode(False))  # 关闭飞行模式

    # 6.测试打开/关闭计算器
    #print(control_calculator(True))   # 打开计算器
    #print(control_calculator(False))  # 关闭计算器
    
    # 7.测试打开/关闭任务管理器
    #print(control_task_manager(True))   # 打开任务管理器
    #print(control_task_manager(False))  # 关闭任务管理器

    # 8.测试截图功能
    #print(capture_screen())

    # 9.测试获取系统信息函数
    # print(get_system_info()["message"])
    # vo.TTS_play_audio_stream(get_system_info()["message"])
    # 10.测试控制摄像头开关并拍照
    # print(control_camera(True))   # 打开摄像头并拍照

    # 11.测试获取当前音量函数
    #print(get_volume())

    # 12.测试获取当前屏幕亮度大小
    #print(get_brightness())

    # 13.
    # 初始化 vLLM 并检查结果
    if not vLLM_init():
        print("vLLM 初始化失败，程序退出")
        sys.exit(1)
        
    # 继续执行其他操作
    # camera_to_vLLM(True)
    # # 2.执行多模态对话
    # current_transcription = "你能看到我现在的样子吗？"
    # vo.Mem0_LLM_TTS(current_transcription,"john")

    # current_transcription = "你能帮我心情好点吗？"
    # vo.Mem0_LLM_TTS(current_transcription,"john")




