"""
Imagen4 节点实现
"""

import requests
import json
import base64
import random
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


class Imagen4Node:
    """Imagen4节点 - Google Imagen4 图像生成"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": ("STRING", {"forceInput": True}),

                "model": (["imagen-4.0-generate-001", "imagen-4.0-fast-generate-001", "imagen-4.0-ultra-generate-001"], {
                    "default": "imagen-4.0-generate-001"
                }),
                "size": (["1K", "2K"], {
                    "default": "1K"
                }),
                "ratio": (["1:1", "4:3", "3:4", "16:9", "9:16"], {
                    "default": "1:1"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "result", "raw_json")
    FUNCTION = "generate"
    CATEGORY = "API365"

    def build_api365_payload(self, negative_prompt, user_prompt, aspect_ratio, image_size, seed):
        """构建谷歌官方 Gemini Imagen4 格式的请求"""

        payload = {
            "instances": [
                {
                "prompt": user_prompt
                }
            ],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": aspect_ratio,
                "imageSize": image_size,
                "seed": seed,
                "addWatermark": False
            }
        }
        
        # 构建完整的payload
        if negative_prompt and negative_prompt.strip():
            payload["parameters"]["negative_prompt"] = negative_prompt

        return payload


    def calculate_dimensions(self, aspect_ratio, image_size):
        """计算图像尺寸"""
        ratio_map = {
            "1:1": (1, 1), "3:4": (3, 4), 
            "4:3": (4, 3), "9:16": (9, 16), "16:9": (16, 9)
        }
        
        size_map = {"1K": 1024, "2K": 2048}
        
        w_ratio, h_ratio = ratio_map.get(aspect_ratio, (1, 1))
        base_size = size_map.get(image_size, 1024)
        
        if w_ratio >= h_ratio:
            width = base_size
            height = int(base_size * h_ratio / w_ratio)
        else:
            height = base_size
            width = int(base_size * w_ratio / h_ratio)
            
        return width, height

    def create_default_image(self, aspect_ratio, image_size):
        """创建默认占位图"""
        width, height = self.calculate_dimensions(aspect_ratio, image_size)
        img = Image.new('RGB', (width, height), color='white')
        return pil2tensor(img)

    def decode_image(self, image_url):
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

    def generate(self, **kwargs):
        """生成图像"""
        negative_prompt = kwargs.get("negative_prompt", None)
        user_prompt = kwargs.get("user_prompt", None)
        model = kwargs.get("model", "")
        image_size = kwargs.get("size", "1K")
        aspect_ratio = kwargs.get("ratio", "1:1")
        api_key = kwargs.get("api_key", "")

        session = None
        try:
            effective_seed = random.randint(0, 2147483647)
            # 构建请求payload
            payload = self.build_api365_payload(negative_prompt, user_prompt, aspect_ratio, image_size, effective_seed)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            url = f"https://gemini.api365.cloud/v1/models/{model}:predict"
            
            session = requests.Session()
            response = session.post(url, headers=headers, json=payload, timeout=180)

            if response.status_code != 200:
                error_msg = f"API请求失败: {response.status_code} - {response.text}"
                print(f"Error: {error_msg}")
                return (self.create_default_image(aspect_ratio, image_size), error_msg, "None")
            
            resp_json = response.json()
            
            # 解析响应
            if "predictions" in resp_json and resp_json["predictions"]:
                image_data = resp_json["predictions"][0]
                if "bytesBase64Encoded" in image_data:
                    # Base64格式
                    image_url = f"data:image/png;base64,{image_data['bytesBase64Encoded']}"
                    output_image = self.decode_image(image_url)
                    return (output_image, "图片生成成功", json.dumps(resp_json, ensure_ascii=False))
                else:
                    return (self.create_default_image(aspect_ratio, image_size), "API未返回图片数据", json.dumps(resp_json, ensure_ascii=False))
            else:
                return (self.create_default_image(aspect_ratio, image_size), "API未返回图片", json.dumps(resp_json, ensure_ascii=False))
                
        except Exception as e:
            error_msg = f"处理过程中发生错误: {str(e)}"
            print(f"Error: {error_msg}")
            return (self.create_default_image(aspect_ratio, image_size), error_msg, "None")
        
        finally:
            if session:
                session.close()