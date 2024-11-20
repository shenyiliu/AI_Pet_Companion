'''
1.调整音量   (完成)
2.调整亮度   (完成)
3.检测电池状态 (完成)
适用于笔记本电脑，查询电池电量和剩余使用时间。
4.开启/关闭省电模式  (完成)
5.开启/关闭飞行模式  (完成)
6.打开/关闭计算器  (完成)
7.打开/关闭任务管理器 (完成)
8.截图当前窗口并保存到桌面
9.获取系统基本信息
比如 CPU、内存、磁盘使用情况。
10.打开/关闭摄像头,拍一张照片
'''

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import wmi
import os
import sys

# 1.调整音量
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

# 2.调整亮度
def Set_brightness(brightness_level: str):
    """
    设置 Windows 系统屏幕亮度
    :param brightness_level: 亮度值(1-100的整数)
    :return: None
    """
    try:
        brightness_level = int(brightness_level)
        # 验证输入值是否在有效范围内
        if not isinstance(brightness_level, int) or brightness_level < 1 or brightness_level > 100:
            raise ValueError("亮度必须是1-100之间的整数")
        
        # 创建WMI接口
        wmi_interface = wmi.WMI(namespace='wmi')
        # 获取亮度控制方法
        brightness_methods = wmi_interface.WmiMonitorBrightnessMethods()[0]
        # 设置亮度
        brightness_methods.WmiSetBrightness(brightness_level, 0)
        
        print(f"屏幕亮度已设置为: {brightness_level}%")
        
    except Exception as e:
        print(f"设置亮度时出错: {str(e)}")

# 3.检测电池状态
def check_battery_status():


    """
    检测Windows系统电池状态
    :return: 包含电池信息的字典
    """
    try:
        import psutil
        
        # 获取电池信息
        battery = psutil.sensors_battery()
        
        if battery is None:
            return "未检测到电池，可能是台式电脑或电池驱动异常"
            
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
        status = {
            "电池电量": f"{percent}%",
            "电源状态": "已连接电源" if power_plugged else "使用电池中",
        }
        
        return status
        
    except Exception as e:
        return f"获取电池信息时出错: {str(e)}"

# 4.启/关闭省电模式
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
        # 如果已经是管理员权限，直接执行
        subprocess.run(command, shell=True)
    else:
        # 请求管理员权限
        ctypes.windll.shell32.ShellExecuteW(
            None, 
            "runas",  # 请求管理员权限
            sys.executable,  # Python解释器路径
            f'"{sys.argv[0]}" "{command}"',  # 要执行的脚本和命令
            None, 
            1  # SW_SHOWNORMAL
        )

# 4.1.开启/关闭省电模式
def set_power_mode(enable: str):
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
            if enable.lower() == 'true':
                enable = True
            elif enable.lower() == 'false':
                enable = False
            else:
                return "输入错误，请输入 True 或 False"

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
            Set_brightness("50" if enable else "75")
        except:
            pass
            
        return f"省电模式已{'开启' if enable else '关闭'}"
            
    except Exception as e:
        return f"设置省电模式时出错: {str(e)}"
    
# 5.开启/关闭飞行模式
def set_airplane_mode(enable: str):
    """
    控制 Windows 系统的飞行模式
    :param enable: "True" 开启飞行模式，"False" 关闭飞行模式
    :return: 操作结果信息
    """
    try:
        # 将字符串转换为布尔值
        if enable.lower() == 'true':
            enable = True
        elif enable.lower() == 'false':
            enable = False
        else:
            return "输入错误，请输入 True 或 False"
            
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
            return f"飞行模式已{'开启' if enable else '关闭'}"
        else:
            return f"设置失败: {result.stderr}"
            
    except Exception as e:
        return f"设置飞行模式时出错: {str(e)}"

# 6.打开/关闭计算器
def control_calculator(enable: str):
    """
    控制 Windows 系统计算器的打开/关闭
    :param enable: "True" 打开计算器，"False" 关闭计算器
    :return: 操作结果信息
    """
    try:
        # 将字符串转换为布尔值
        if enable.lower() == 'true':
            enable = True
        elif enable.lower() == 'false':
            enable = False
        else:
            return "输入错误，请输入 True 或 False"
            
        import subprocess
        
        if enable:
            # 打开计算器
            subprocess.Popen('calc.exe')
            return "计算器已打开"
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
                return "计算器已关闭"
            else:
                return "未找到正在运行的计算器程序"
                
    except Exception as e:
        return f"操作计算器时出错: {str(e)}"

# 7.打开/关闭任务管理器
def control_task_manager(enable: str):
    """
    控制 Windows 系统任务管理器的打开/关闭
    :param enable: "True" 打开任务管理器，"False" 关闭任务管理器
    :return: 操作结果信息的字符串
    """
    try:
        # 将字符串转换为布尔值
        if enable.lower() == 'true':
            enable = True
        elif enable.lower() == 'false':
            enable = False
        else:
            return "输入错误，请输入 True 或 False"
            
        import subprocess
        import tempfile
        import os
        import time
        
        if enable:
            # 使用 PowerShell 命令打开任务管理器
            subprocess.run(['powershell', 'Start-Process', 'taskmgr.exe'])
            return "任务管理器已打开"
        else:
            # 创建临时 PowerShell 脚本文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ps1', mode='w') as f:
                f.write('Stop-Process -Name Taskmgr -Force\n')
                ps_file = f.name
            
            # 使用 PowerShell 以管理员权限执行脚本
            command = f'powershell -Command "Start-Process powershell -ArgumentList \'-ExecutionPolicy Bypass -File {ps_file}\' -Verb RunAs"'
            subprocess.run(command, shell=True)
            
            # 等待命令执行完成
            time.sleep(1)
            
            # 删除临时文件
            try:
                os.unlink(ps_file)
            except:
                pass
                
            return "任务管理器已关闭"
                
    except Exception as e:
        return f"操作任务管理器时出错: {str(e)}"

# 8.截图当前窗口并保存到桌面
def capture_screen() -> str:
    """
    截取当前窗口并保存到桌面
    :param enable: "True" 开始截图，其他值返回使用说明
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

        return {"message":"截图保存成功","imagePath":filepath}
        
    except Exception as e:
        return f"截图过程出错: {str(e)}"

if __name__ == "__main__":
    # 1.测试控制音量函数
    # Set_volume("100")
    
    # 2.测试控制亮度函数
    # Set_brightness("75")
    
    # 3.测试电池状态检测函数
    # battery_info = check_battery_status()
    # print(battery_info)
    
    # 4.测试省电模式控制函数
    #print(set_power_mode("True"))   # 开启省电模式
    #print(set_power_mode("False"))  # 关闭省电模式

    # 5.测试飞行模式控制函数
    #print(set_airplane_mode("True"))   # 开启飞行模式
    #print(set_airplane_mode("False"))  # 关闭飞行模式

    # 6.测试打开/关闭计算器
    #print(control_calculator("True"))   # 打开计算器
    #print(control_calculator("False"))  # 关闭计算器
    
    # 7.测试打开/关闭任务管理器
    #print(control_task_manager("True"))   # 打开任务管理器
    #print(control_task_manager("False"))  # 关闭任务管理器

    # 8.测试截图功能
    #print(capture_screen())

    



