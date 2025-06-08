import os
import re
import cv2
import time
import json
import torch
import base64
import uvicorn
import easyocr
import numpy as np
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.tmt.v20180321 import tmt_client, models
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

app = FastAPI()

# 加载环境变量
load_dotenv()

# 是否调试模式
debug_mode: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"

# 是否调试模式
use_similarity_detection: bool = os.getenv("USE_SIMILARITY_DETECTION", "False").lower() == "true"

# 区域高度配置
QUESTION_START_Y = float(os.getenv("QUESTION_START_Y", "0.18"))  # 问题区域起始Y坐标（距离顶部）- 小于等于1时表示屏幕高度的百分比
QUESTION_END_Y = float(os.getenv("QUESTION_END_Y", "0.5"))  # 问题区域结束Y坐标（距离底部）- 小于等于1时表示屏幕高度的百分比
ANSWER_END_Y = float(os.getenv("ANSWER_END_Y", "0.17"))  # 答案区域结束Y坐标（距离底部）- 小于等于1时表示屏幕高度的百分比
JUMP_END_Y = float(os.getenv("JUMP_END_Y", "0.11"))  # 跳过按钮区域结束Y坐标（距离底部）- 小于等于1时表示屏幕高度的百分比

# 初始化语义相似度模型 - 【非必要】如果使用自定义字典，可以移除语义相似度模型
similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') if use_similarity_detection else None

# 腾讯云翻译配置 - 【非必要】如果使用自定义字典，可以移除翻译配置
TX_SECRET_ID = os.getenv("TX_SECRET_ID", "")
TX_SECRET_KEY = os.getenv("TX_SECRET_KEY", "")

# 初始化OCR读取器
gpu_available = torch.cuda.is_available()
reader_cn = easyocr.Reader(['ch_sim'], gpu=gpu_available)
reader_ko = easyocr.Reader(['ko'], gpu=gpu_available)

class ImageRequest(BaseModel):
    image_base64: str
    custom_dict: dict = None
    need_jump_info: bool = True
    need_commit_info: bool = True
    commit_texts: List[str] = []

class BboxItem(BaseModel):
    txt: str
    bbox: List[List[int]]
    center: List[int]

class AnswerItem(BaseModel):
    question: str | None
    answer: BboxItem | None
    similarity: float | None

class ResponseItem(BaseModel):
    jump: BboxItem | None
    commit: BboxItem | None
    answers: List[AnswerItem]

class APIResponse(BaseModel):
    code: int
    msg: str
    data: ResponseItem
    elapsed_ms: int | None = None

def generate_bbox_item(v: Optional[tuple[str, List[List[int]]]]) -> BboxItem | None:
    """
    根据OCR结果生成边界框对象
    
    Args:
        v: OCR结果元组，包含文本和边界框坐标
        
    Returns:
        BboxItem对象或None
    """
    if not v:
        return None
    txt, bbox = v
    return BboxItem(txt=txt, bbox=bbox, center=get_center_from_bbox(bbox))

def is_korean(text: str) -> bool:
    """
    判断文本是否包含韩文字符
    
    Args:
        text: 待检查的文本
        
    Returns:
        是否包含韩文字符
    """
    korean_pattern = re.compile('[가-힣]')
    return bool(korean_pattern.search(text))

def get_similarity(text1: str, text2: str) -> float:
    """
    计算两段文本的语义相似度
    
    Args:
        text1: 第一段文本
        text2: 第二段文本
        
    Returns:
        相似度分数(0-1之间)
    """
    if not similarity_model:
        return 0.0
    
    embeddings1 = similarity_model.encode(text1)
    embeddings2 = similarity_model.encode(text2)
    similarity = np.dot(embeddings1, embeddings2) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))
    return similarity

def tx_translate_text(text: str) -> str:
    """
    使用腾讯云翻译API进行文本翻译
    
    Args:
        text: 待翻译的文本
        
    Returns:
        翻译后的文本
        
    Raises:
        HTTPException: 翻译服务出错时抛出异常
    """
    # 未启用腾讯翻译服务，直接返回原文
    if not TX_SECRET_ID or not TX_SECRET_KEY:
        return text
    
    try:
        cred = credential.Credential(TX_SECRET_ID, TX_SECRET_KEY)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "tmt.tencentcloudapi.com"
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = tmt_client.TmtClient(cred, "ap-guangzhou", clientProfile)
        req = models.TextTranslateRequest()
        
        if is_korean(text):
            params = {
                "SourceText": text,
                "Source": "ko",
                "Target": "zh",
                "ProjectId": 0
            }
        else:
            params = {
                "SourceText": text,
                "Source": "zh",
                "Target": "ko",
                "ProjectId": 0
            }
            
        req.from_json_string(json.dumps(params))
        resp = client.TextTranslate(req)
        return resp.TargetText
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

def ocr_dual(img: np.ndarray, offset_y: int = 0, conf_thres: float = 0.35) -> List[tuple]:
    """
    对图像进行中韩双语OCR识别
    
    Args:
        img: 输入图像
        offset_y: Y轴偏移量
        conf_thres: 置信度阈值
        
    Returns:
        包含文本和边界框的元组列表
    """
    # 串行调用OCR，先中文后韩文
    result_cn = reader_cn.readtext(img, detail=1, batch_size=2)
    result_ko = reader_ko.readtext(img, detail=1, batch_size=2)
    
    results = []
    seen_texts = set()
    
    for r in result_cn + result_ko:
        text = r[1].strip()
        conf = r[2]
        bbox = r[0]
        x_coords = [int(p[0]) for p in bbox]
        y_coords = [int(p[1] + offset_y) for p in bbox]
        top_left = [min(x_coords), min(y_coords)]
        bottom_right = [max(x_coords), max(y_coords)]
        
        if conf >= conf_thres and len(text) > 0:
            if text not in seen_texts:
                seen_texts.add(text)
                results.append((text, [top_left, bottom_right]))
    return results

def get_percentage(y1: int, y2: int) -> str:
    """
    计算y1和y2的百分比
    
    Args:
        y1: 分子
        y2: 分母
        
    Returns:
        百分比字符串
    """
    percentage = int(round((y1 / y2) * 100))
    return f"{percentage}%"

def visualize_debug_results(img: np.ndarray, results: List[dict]) -> None:
    """
    在图片上绘制检测框并保存为调试图片
    
    Args:
        img: 原始图片
        results: 答案列表
    """
    debug_img = img.copy()
    h, w = img.shape[:2]
    
    # 定义区域颜色
    colors = {
        'default': (60, 20, 220),   # 默认颜色
        'question': (255, 0, 0),    # 问题区
        'answer': (130, 0, 75),     # 答案区
        'jump': (0, 0, 255),        # 跳过按钮区
        'commit': (128, 0, 128)     # 提交按钮区
    }
    
    # 显示宽高信息
    cv2.putText(debug_img, f"w={w}, h={h}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, colors['default'], 2)
    
    # 绘制区域边界
    # 问题区
    question_start_y = get_real_y(QUESTION_START_Y, h=h)
    question_end_y = get_real_y(QUESTION_END_Y, h=h)
    cv2.rectangle(debug_img, (0, question_start_y), (w, question_end_y), colors['question'], 2)
    cv2.putText(debug_img, f"question y={question_start_y} {get_percentage(question_start_y, h)}", (10, question_start_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, colors['question'], 2)
    
    # 答案区
    answer_end_y = get_real_y(ANSWER_END_Y, h=h)
    cv2.rectangle(debug_img, (0, question_end_y), (w, h-answer_end_y), colors['answer'], 2)
    cv2.putText(debug_img, f"answer y={question_end_y} {get_percentage(question_end_y, h)}", (10, question_end_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, colors['answer'], 2)
    
    # 跳过按钮区
    jump_end_y = get_real_y(JUMP_END_Y, h=h)
    cv2.rectangle(debug_img, (0, h-answer_end_y), (w, h-jump_end_y), colors['jump'], 2)
    cv2.putText(debug_img, f"jump y={h-answer_end_y} (h-{answer_end_y}) -{get_percentage(answer_end_y, h)}", (10, h-answer_end_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, colors['jump'], 2)
    
    # 提交按钮区
    cv2.rectangle(debug_img, (0, h-jump_end_y), (w, h), colors['commit'], 2)
    cv2.putText(debug_img, f"commit y={h-jump_end_y} (h-{jump_end_y}) -{get_percentage(jump_end_y, h)}", (10, h-jump_end_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, colors['commit'], 2)
    
    # 画答案框和打印坐标
    print("\n=== 坐标信息 ===")
    for idx, result in enumerate(results, 1):
        if result['answer']:
            bbox = result['answer'].bbox
            center = result['answer'].center
            txt = result['answer'].txt
            
            # 绘制答案框
            cv2.rectangle(debug_img, 
                        (bbox[0][0], bbox[0][1]), 
                        (bbox[1][0], bbox[1][1]), 
                        colors['answer'], 2)
            
            # 绘制中心点
            cv2.circle(debug_img, (center[0], center[1]), 5, (0, 0, 255), -1)
            
            # 添加文本标注
            cv2.putText(debug_img, f"Answer {idx}", 
                       (bbox[0][0], bbox[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['answer'], 2)
            
            # 打印坐标信息
            print(f"\n答案 {idx}:")
            print(f"文本: {txt}")
            print(f"左上角: {bbox[0]}")
            print(f"右下角: {bbox[1]}")
            print(f"中心点: {center}")
    
    # 确保results目录存在
    os.makedirs('output/results', exist_ok=True)
    
    # 保存调试图片
    _t = int(time.time())
    output_path = f'output/results/api_result_{_t}.jpg'
    cv2.imwrite(output_path, debug_img)
    print(f"\n调试图片已保存至: {output_path}")

def remove_special_chars(text):
    """
    移除文本中的特殊字符（数字、字母和标点符号）
    
    Args:
        text: 输入文本
        
    Returns:
        清理后的文本
    """
    pattern = r'[0-9a-zA-Z\s\.,;:!?\'"()\[\]{}<>@#$%^&*\-_+=|\\/~`]'
    return re.sub(pattern, '', text)

def clean_base64(base64_str: str) -> str:
    """
    清理base64字符串，移除可能存在的data URI scheme
    
    Args:
        base64_str: 原始base64字符串
        
    Returns:
        清理后的base64字符串
    """
    if ',' in base64_str:
        return base64_str.split(',', 1)[1]
    return base64_str

def is_percentage(v: float) -> bool:
    """判断是否为百分比"""
    return 0 <= v <= 1

def get_real_y(v: float, h: int) -> int:
    """获取真实y高度值"""
    return int(h * v if is_percentage(v) else v)

def get_block_info(img, y1: int, y2: int) -> List[tuple[str, List[List[int]]]]:
    """
    获取图像指定区域的OCR识别结果
    
    Args:
        img: 输入图像
        y1: 区域起始Y坐标
        y2: 区域结束Y坐标
        
    Returns:
        包含文本和边界框的元组列表
    """
    h, w = img.shape[:2]
    print(f"w={w}, h={h}, y1={y1}, y2={y2}")
    
    crop = img[y1:y2, 0:w]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    return ocr_dual(crop_rgb, offset_y=y1) or []

def get_question_info(img) -> List[tuple[str, List[List[int]]]]:
    """
    获取问题区域的文本和位置信息
    
    Args:
        img: 输入图像
        
    Returns:
        包含问题文本和位置信息的列表
    """
    h = img.shape[0]
    
    y1 = get_real_y(QUESTION_START_Y, h=h)
    y2 = get_real_y(QUESTION_END_Y, h=h)
    return get_block_info(img=img, y1=y1, y2=y2)

def get_answer_info(img) -> List[tuple[str, List[List[int]]]]:
    """
    获取答案区域的文本和位置信息
    
    Args:
        img: 输入图像
        
    Returns:
        包含答案文本和位置信息的列表
    """
    h = img.shape[0]
    y1 = get_real_y(QUESTION_END_Y, h=h)
    y2 = h - get_real_y(ANSWER_END_Y, h=h)
    return get_block_info(img=img, y1=y1, y2=y2)

def get_jump_button_info(img) -> tuple[str, List[List[int]]]:
    """
    获取跳过按钮的位置和文本信息
    
    Args:
        img: 输入图像
        
    Returns:
        包含按钮文本和位置信息的元组
    """
    h = img.shape[0]
    
    y1 = h - get_real_y(ANSWER_END_Y, h=h)
    y2 = h - get_real_y(JUMP_END_Y, h=h)
    results = get_block_info(img=img, y1=y1, y2=y2)
    return results[0] if results else None

def get_commit_button_info(img) -> tuple[str, List[List[int]]]:
    """
    获取提交按钮的位置和文本信息
    
    Args:
        img: 输入图像
        
    Returns:
        包含按钮文本和位置信息的元组
    """
    h = img.shape[0]
    y1 = h - get_real_y(JUMP_END_Y, h=h)
    results = get_block_info(img=img, y1=y1, y2=h)
    return results[0] if results else None

def get_center_from_bbox(bbox: List[List[int]]) -> List[int]:
    """
    计算边界框的中心点坐标
    
    Args:
        bbox: 边界框坐标 [[x1, y1], [x2, y2]]
        
    Returns:
        中心点坐标 [center_x, center_y]
    """
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return [center_x, center_y]

def process_image(request: ImageRequest) -> dict:
    """
    处理图像请求，进行OCR识别和文本匹配
    
    Args:
        request: 包含图像和配置信息的请求对象
        
    Returns:
        处理结果字典，包含答案、提交按钮和跳过按钮信息
        
    Raises:
        HTTPException: 处理过程中出现错误时抛出异常
    """
    try:
        
        t0 = time.time()
        image_base64 = request.image_base64
        image_base64 = clean_base64(image_base64)
        t1 = time.time()
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        t2 = time.time()
        
        # 解析跳转按钮区域信息
        jump_button_info = get_jump_button_info(img=img) if request.need_jump_info else None
        t3 = time.time()
        
        # 解析提交按钮区域信息
        commit_button_info = get_commit_button_info(img=img) if request.need_commit_info else None
        t4 = time.time()
        
        # 解析问题区域信息
        question_infos = get_question_info(img=img)
        t5 = time.time()
        
        # 解析答案区域信息
        answer_infos = get_answer_info(img=img)
        t6 = time.time()
        
        # 提取问题和答案的文本列表
        questions = [text for text, _ in question_infos]
        answers = [text for text, _ in answer_infos]
        
        # 翻译问题内容 - 优先使用自定义字典
        translated_questions = [request.custom_dict.get(remove_special_chars(q), q) for q in questions] if request.custom_dict else []
        print(f"request.custom_dict... {request.custom_dict}") if debug_mode else None
        print(f"Translating questions... {translated_questions}")
        
        # 如果自定义字典无结果，尝试使用翻译api服务进行翻译
        if not translated_questions:
            translated_questions = [tx_translate_text(q) for q in questions]
        
        t7 = time.time()
        
        results = []
        for q_trans, q_orig, q_bbox in zip(translated_questions, questions, [bbox for _, bbox in question_infos]):
            q_trans = remove_special_chars(q_trans)
            if not q_trans:
                continue
            
            best_match = None
            best_similarity = -1
            best_answer_bbox = None
            
            for a_txt, a_bbox in zip(answers, [bbox for _, bbox in answer_infos]):
                # 优先判断全文匹配
                if q_trans == remove_special_chars(a_txt):
                    best_match = a_txt
                    best_similarity = 1
                    best_answer_bbox = a_bbox
                    break
            
            if not best_match and use_similarity_detection:
                for a_txt, a_bbox in zip(answers, [bbox for _, bbox in answer_infos]):
                    # 如果不相等再尝试进行相似度判断
                    similarity = get_similarity(q_trans, a_txt)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = a_txt
                        best_answer_bbox = a_bbox
            
            if best_match:
                results.append({
                    "question": q_orig,
                    "answer": generate_bbox_item((best_match, best_answer_bbox)),
                    "similarity": best_similarity,
                })
                
        t8 = time.time()
        
        if debug_mode:
            visualize_debug_results(img, results)
        
        print(f"base64处理耗时: {(t1-t0)*1000:.2f} ms")
        print(f"图片解码耗时: {(t2-t1)*1000:.2f} ms")
        print(f"跳过按钮检测耗时: {(t3-t2)*1000:.2f} ms")
        print(f"提交按钮检测耗时: {(t4-t3)*1000:.2f} ms")
        print(f"问题区OCR耗时: {(t5-t4)*1000:.2f} ms")
        print(f"答案区OCR耗时: {(t6-t5)*1000:.2f} ms")
        print(f"问题翻译耗时: {(t7-t6)*1000:.2f} ms")
        print(f"相似度匹配耗时: {(t8-t7)*1000:.2f} ms")
        print(f"总耗时: {(t8-t0)*1000:.2f} ms")
        
        return {
            "answers": results,
            "commit": generate_bbox_item(commit_button_info) if commit_button_info else None,
            "jump": generate_bbox_item(jump_button_info)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/process_image", response_model=APIResponse)
async def process_image_endpoint(request: ImageRequest):
    start_time = time.time()
    try:
        result = process_image(request)
        elapsed_ms = (time.time() - start_time) * 1000  # 毫秒
        result["elapsed_ms"] = elapsed_ms
        print(f"处理时间: {elapsed_ms:.2f} ms")
        return {
            "code": 200,
            "msg": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    
    print(f"gpu_available={gpu_available}")
    if not gpu_available:
        print("当前正在使用CPU运行程序，推荐安装CUDA版pytorch加速推理：https://pytorch.org/get-started/locally/")
        
    uvicorn.run(app, host="0.0.0.0", port=9720) 