import os
import subprocess
import base64
import requests
import time
from PIL import Image

# API服务地址
API_URL = os.getenv("API_URL", 'http://localhost:9720')

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

# 确认按钮文本
commit_texts = ["继续", "提交", "开玩", "检查", "我能做到!", "知道了", "不。谢谢", "退出", "下次再说", "立即开始"]

# 当前跳过按钮信息
current_jump_info = None

# 当前确认按钮信息
current_commit_info = None

# 是否每次都点跳过按钮
always_click_jump = False

def click_coordinates(x, y):
    """使用adb点击指定坐标"""
    subprocess.run(['adb', 'shell', 'input', 'tap', str(x), str(y)], check=True)
    time.sleep(0.03)  # 等待点击动作完成

def get_center_point(v: dict):
    """读取中心坐标点"""
    return v.get("center") if v else None

def need_to_jump(txt: str) -> bool:
    """是否需要跳过"""
    return '听力' in txt

def need_to_commit(txt: str) -> bool:
    """是否需要提交"""
    return txt in commit_texts

def handle_api_response(response_data):
    """处理API返回结果，如果有center坐标则点击"""
    
    global current_jump_info, current_commit_info
    
    if response_data['code'] == 200:
        data = response_data['data']
        
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
        subprocess.run(['adb', 'exec-out', 'screencap', '-p'], stdout=f, check=True)
    return screencap_local_path

def compress_png_lossless(input_path, output_path):
    """对PNG图片进行无损压缩"""
    img = Image.open(input_path)
    img.save(output_path, optimize=True)

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