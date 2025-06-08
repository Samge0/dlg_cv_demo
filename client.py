import os
import subprocess
import base64
import requests
import time
from PIL import Image

# API服务地址
API_URL = os.getenv("API_URL", 'http://localhost:9720')

# 手机IP地址和端口
PHONE_IP_PORT = os.getenv("PHONE_IP_PORT", "") # 192.168.xx.xx:5555

# 自定义字典 - 可减轻翻译api耗时
CUSTOM_DICT = {
    "不是": "아니요",
    "朋友": "친구",
    "灰色": "회색",
    "岁": "살",
    "猫": "고양이"
}

# 自定义字典 - 正反组装
custom_dict_all = {**CUSTOM_DICT, **{v: k for k, v in CUSTOM_DICT.items()}}

# 确认按钮过滤文本 - 可选，置为空则表示不过滤
# commit_filter_texts = ["继续", "提交", "开玩", "检查", "我能做到!", "知道了", "不。谢谢", "退出", "下次再说", "立即开始"]
commit_filter_texts = []

# 当前跳过按钮信息
current_jump_info = None

# 当前确认按钮信息
current_commit_info = None

# 是否每次都点跳过按钮
always_click_jump = False

def run_adb_command(command, capture_output=False, text=False, check=True, stdout=None):
    """
    执行adb命令
    
    Args:
        command: adb命令列表
        capture_output: 是否捕获输出
        text: 是否返回文本
        check: 是否检查命令执行状态
        stdout: 输出重定向目标
    
    Returns:
        命令执行结果
    """
    if PHONE_IP_PORT:
        command = ['adb', '-s', PHONE_IP_PORT] + command
    else:
        command = ['adb'] + command
    
    return subprocess.run(command, capture_output=capture_output, text=text, check=check, stdout=stdout)

def connect_adb():
    """连接手机设备"""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # 启动adb服务
            run_adb_command(['start-server'])
            
            if not PHONE_IP_PORT:
                # 检查是否有任何已连接的设备
                result = run_adb_command(['devices'], capture_output=True, text=True)
                devices = [line.split('\t')[0] for line in result.stdout.strip().split('\n')[1:] if line.strip()]
                if devices:
                    print(f"发现已连接的设备: {', '.join(devices)}")
                    return True
                else:
                    print("未发现任何已连接的设备")
                    return False
            
            # 连接指定设备
            run_adb_command(['connect', f'{PHONE_IP_PORT}'])
            # 等待连接建立
            time.sleep(2)
            
            # 检查设备连接状态
            result = run_adb_command(['devices'], capture_output=True, text=True, check=True)
            if f'{PHONE_IP_PORT}' in result.stdout:
                # 检查设备是否已授权
                if 'unauthorized' in result.stdout:
                    print(f"设备 {PHONE_IP_PORT} 需要授权，请在设备上确认授权请求")
                    time.sleep(5)  # 等待用户确认授权
                    retry_count += 1
                    continue
                print(f"成功连接到设备 {PHONE_IP_PORT}")
                return True
            else:
                print(f"设备连接失败: {result.stdout}")
                return False
            
        except subprocess.CalledProcessError as e:
            print(f"ADB连接错误: {e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"将在5秒后重试... (第{retry_count}次)")
                time.sleep(5)
            else:
                print("达到最大重试次数，连接失败")
                return False
    
    return False

def click_coordinates(x, y):
    """使用adb点击指定坐标"""
    run_adb_command(['shell', 'input', 'tap', str(x), str(y)])
    time.sleep(0.03)  # 等待点击动作完成

def get_center_point(v: dict):
    """读取中心坐标点"""
    return v.get("center") if v else None

def need_to_jump(txt: str) -> bool:
    """是否需要跳过"""
    return '听力' in txt

def need_to_commit(txt: str) -> bool:
    """是否需要提交"""
    if not commit_filter_texts:
        return True
    return txt in commit_filter_texts

def handle_api_response(response_data):
    """处理API返回结果，如果有center坐标则点击"""
    
    global current_jump_info, current_commit_info
    
    if response_data['code'] == 200:
        data = response_data['data']
        
        # 检查 答案选项
        answers = data.get('answers') or []
        for _item in answers[:1]:
            answer_item = _item.get("answer")
            
            txt = answer_item.get('txt') or ''
            similarity = answer_item.get('similarity') or ''
            center = get_center_point(answer_item)
            if not center:
                print(f"答案选项不符合要求，跳过：{center} | {txt}")
                continue
            
            print(f"点击 答案选项 坐标: {center} | {txt} | {similarity}")
            click_coordinates(center[0], center[1])
        
        # 检查 jump按钮
        jump_data = current_jump_info or data.get('jump') or {}
        if jump_data and get_center_point(jump_data):
            
            txt = jump_data.get('txt') or ''
            if need_to_jump(txt) is False:
                print(f"jump按钮文字不符合要求，跳过：{txt}")
            elif always_click_jump or current_jump_info is None:
                current_jump_info = jump_data
                center = get_center_point(jump_data)
                print(f"点击 jump按钮 坐标: {center} | {txt}")
                click_coordinates(center[0], center[1])
            
        # 检查 commit按钮
        commit_data = current_commit_info or data.get('commit') or {}
        if commit_data and get_center_point(commit_data):
            current_commit_info = commit_data
            
            txt = commit_data.get('txt') or ''
            if need_to_commit(txt) is False:
                print(f"commit按钮文字不符合要求，跳过：{txt}")
                return False
            
            center = get_center_point(commit_data)
            for i in range(1, 6):
                time.sleep(0.03)
                print(f"【{i}】点击 commit按钮 坐标: {center} | {txt}")
                click_coordinates(center[0], center[1])
            
            return True
            
    return False

def capture_screen_fast():
    """更快地捕获设备屏幕并保存到本地缓存目录"""
    os.makedirs('output/screens', exist_ok=True)
    _t = int(time.time())
    screencap_local_path = f'output/screen_{_t}.png'
    with open(screencap_local_path, 'wb') as f:
        run_adb_command(['exec-out', 'screencap', '-p'], stdout=f, check=True)
    return screencap_local_path

def screen_and_call():
    """abd截屏并调用API进行识别"""
    
    global current_jump_info, current_commit_info
    
    # 捕获屏幕
    screencap_local_path = capture_screen_fast()

    # 读取图片并转为base64
    with open(screencap_local_path, 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # 发送HTTP请求到test_api服务
    response = requests.post(
        f'{API_URL}/process_image',
        json={
            'image_base64': img_base64,
            'need_jump_info': current_jump_info is None,
            'need_commit_info': current_commit_info is None,
            'custom_dict': custom_dict_all
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print('API返回结果:', result)
        handle_api_response(result)
    else:
        print('请求失败:', response.status_code, response.text)

def main(loop: bool = True, interval: float = 0.1):
    """
    主函数，用于执行屏幕捕获和API调用
    
    Args:
        loop: 是否循环执行，默认为True
        interval: 循环执行时的间隔时间（秒），默认为0.1秒
    """
    # 首先连接设备
    if not connect_adb():
        print("无法连接到设备，程序退出")
        return

    if not loop:
        screen_and_call()
        return

    count = 1
    while True:
        print(f"第{count}次运行")
        screen_and_call()
        time.sleep(interval)
        count += 1

if __name__ == '__main__':
    main(loop=True) 