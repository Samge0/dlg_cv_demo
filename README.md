# 多邻国APP自动答题Demo

一个基于`FastAPI`+`adb`+`easyocr`的多邻国APP自动答题Demo。

本Demo主要测试其中某个韩语课程，使用`easyocr`进行OCR识别，使用`adb`进行屏幕截图&模拟点击，使用`fastapi`进行接口封装，完成自动答题。

https://github.com/user-attachments/assets/259e1f2f-f7a1-4968-963c-a30277b55ab7

## 功能特点

- 支持中文和韩文OCR识别
- 支持语义相似度匹配 [可选]
- 支持中韩互译（需配置腾讯云翻译服务）[可选]
- 支持自定义词典匹配
- 支持GPU加速（需安装CUDA版本PyTorch）

## 环境要求

- Python 3.10+
- CUDA支持（可选，用于GPU加速）

## 安装说明

1. 克隆项目并安装依赖：
    ```bash
    git clone https://github.com/Samge0/dlg_cv_demo.git

    cd dlg_cv_demo

    conda create -n dlg_vc_demo python=3.10.13 -y

    conda activate dlg_vc_demo

    pip install -r requirements.txt
    ```

2. [可选]安装CUDA版本PyTorch（根据您的CUDA版本选择合适的安装命令）：
- 访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 选择适合您系统的安装命令
- 例如，对于CUDA 12.8：
    ```bash
    # 先卸载cup版的pytorch
    pip uninstall torch torchvision

    # 安装CUDA版本PyTorch
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    ```

## 配置说明

1. 创建`.env`文件并配置以下环境变量：
    ```bash
    cp .env.example .env
    ```

    ```env
    # 是否调试模式
    DEBUG_MODE=False

    # [可选] 是否启用语义相似度检测
    USE_SIMILARITY_DETECTION=False

    # [可选] 腾讯云翻译配置
    TX_SECRET_ID=你的腾讯云SecretId
    TX_SECRET_KEY=你的腾讯云SecretKey
    ```

## 使用方法

1. 启动服务：
    ```bash
    python server.py
    ```

2. 服务默认运行在 `http://localhost:9720`

3. API接口：
- POST `/process_image`：处理图像识别请求
  - 请求体格式：
    ```json
    {
        "image_base64": "base64编码的图像数据",
        "need_jump_info": true, # 是否需要跳转按钮信息
        "need_commit_info": true, # 是否需要提交按钮信息
        "custom_dict": {} # 可选，自定义词典，如果不提供自定义字典，则需要配置腾旭翻译
    }
    ```
4. 用`adb`连接手机并打开目标课程页面后，执行自动答题脚本：
    ```bash
    adb connect <phone_ip:port>

    python client.py
    ```

## 加速推理

- 如果您的设备支持CUDA，建议安装对应版本的PyTorch以提升识别速度

## 注意事项

- 首次运行时会自动下载OCR模型文件
- 使用语义相似度功能需要额外下载模型文件
- 使用翻译功能需要配置腾讯云API密钥

### docker方式运行api

> [可选]映射`/app/.env`是为了自定义项目的配置文件
> 
> [可选]映射`/app/output/`可在debug模式时查看输出图片
> 
> [可选]映射`/root/.cache/`目录是为了复用`huggingface`的模型缓存文件

- CPU版本
    ```shell
    docker run -itd \
    -p 9720:9720 \
    --name dlg_cv_demo \
    -v ~/dlg_cv_demo/.env:/app/.env \
    -v ~/dlg_cv_demo/output/:/app/output/ \
    -v ~/.cache/:/root/.cache/ \
    --restart always \
    samge/dlg_cv_demo
    ```

- GPU版本
    ```shell
    docker run -itd \
    --gpus all \
    -p 9720:9720 \
    --name dlg_cv_demo \
    -v ~/dlg_cv_demo/.env:/app/.env \
    -v ~/dlg_cv_demo/output/:/app/output/ \
    -v ~/.cache/:/root/.cache/ \
    --restart always \
    samge/dlg_cv_demo
    ```
