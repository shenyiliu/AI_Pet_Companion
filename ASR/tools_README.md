# 安装环境
```
conda create -n tools_utils python=3.10 -y
activate tools_utils

pip install comtypes
pip install pycaw
pip install wmi
pip install pyautogui
pip install opencv-python

```

环境配置完毕之后即可运行tools_utils.py这个文件，该文件是对windows系统操作的函数，目前支持的功能有：

1. 调整音量   (完成)
2. 调整亮度   (完成)
3. 检测电池状态，适用于笔记本电脑，查询电池电量和剩余使用时间。 (完成)
4. 开启/关闭省电模式  (完成)
5. 开启/关闭飞行模式  (完成)
6. 打开/关闭计算器  (完成)
7. 打开/关闭任务管理器 (完成)
8. 截图当前窗口并保存到桌面  (完成)
9. 获取系统基本信息，比如 CPU、内存使用情况。  (完成)
10. 打开/关闭摄像头,拍一张照片 (完成)