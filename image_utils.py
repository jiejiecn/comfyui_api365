"""
图像处理公共工具模块
"""

import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch


def pil2tensor(image: Image.Image) -> torch.Tensor:
    """将PIL图像转换为ComfyUI tensor格式 [1, H, W, 3]"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    np_image = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_image)
    tensor = tensor.unsqueeze(0)
    return tensor


def tensor2pil(tensor: torch.Tensor) -> list:
    """将ComfyUI tensor转换为PIL图像列表"""
    if len(tensor.shape) == 4:
        return [Image.fromarray((t.cpu().numpy() * 255).astype(np.uint8)) for t in tensor]
    else:
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return [Image.fromarray(np_image)]


def decode_image(image_url: str) -> torch.Tensor:
    """下载或解码图片"""
    try:
        if image_url.startswith('data:image/'):
            base64_data = image_url.split(',', 1)[1]
            image_data = base64.b64decode(base64_data)
            pil_image = Image.open(BytesIO(image_data))
        else:
            session = requests.Session()
            session.trust_env = True
            try:
                response = session.get(image_url, timeout=60)
                response.raise_for_status()
                pil_image = Image.open(BytesIO(response.content))
            finally:
                session.close()
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        print(f"图片解码成功: {pil_image.size}")
        return pil2tensor(pil_image)
        
    except Exception as e:
        print(f"图片解码失败: {str(e)}")
        raise


def calculate_dimensions(aspect_ratio: str, image_size: str) -> tuple:
    """计算图像尺寸"""
    # 宽高比映射
    ratio_map = {
        "1:1": (1, 1), "2:3": (2, 3), "3:2": (3, 2),
        "3:4": (3, 4), "4:3": (4, 3), "4:5": (4, 5),
        "5:4": (5, 4), "9:16": (9, 16), "16:9": (16, 9),
        "21:9": (21, 9)
    }
    
    # 分辨率映射
    size_map = {"1K": 1024, "2K": 2048, "4K": 4096}
    
    w_ratio, h_ratio = ratio_map.get(aspect_ratio, (1, 1))
    base_size = size_map.get(image_size, 1024)
    
    # 计算实际尺寸
    if w_ratio >= h_ratio:
        width = base_size
        height = int(base_size * h_ratio / w_ratio)
    else:
        height = base_size
        width = int(base_size * w_ratio / h_ratio)
        
    return width, height


def create_default_image(size_spec, aspect_ratio=None, image_size=None) -> torch.Tensor:
    """
    创建默认占位图，支持多种参数格式
    
    Args:
        size_spec: 尺寸规格，可以是 "1024x1024" 格式的字符串，或者用于兼容的参数
        aspect_ratio: 宽高比 (如 "1:1", "16:9")，当 size_spec 不是 "WxH" 格式时使用
        image_size: 图像大小 (如 "1K", "2K")，当 size_spec 不是 "WxH" 格式时使用
    
    Returns:
        torch.Tensor: 默认白色图像的tensor
    """
    try:
        # 尝试解析 "1024x1024" 格式
        if isinstance(size_spec, str) and 'x' in size_spec:
            width, height = map(int, size_spec.split('x'))
        else:
            # 使用 aspect_ratio 和 image_size 计算尺寸
            if aspect_ratio and image_size:
                width, height = calculate_dimensions(aspect_ratio, image_size)
            else:
                # 默认尺寸
                width, height = 1024, 1024
    except (ValueError, TypeError):
        # 如果解析失败，使用默认尺寸
        width, height = 1024, 1024
    
    img = Image.new('RGB', (width, height), color='white')
    return pil2tensor(img)