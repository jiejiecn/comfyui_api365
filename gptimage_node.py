"""
GPTImage 节点实现
"""

import requests
import json
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


class GPTImageNode:
    """GPTImage节点 - OpenAI DALL-E 图像生成"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "在这里输入你的提示词..."
                }),
                "model": (["dall-e-3", "dall-e-2"], {
                    "default": "dall-e-3"
                }),
                "size": (["1024x1024", "1024x1792", "1792x1024"], {
                    "default": "1024x1024"
                }),
                "quality": (["standard", "hd"], {
                    "default": "standard"
                }),
                "style": (["vivid", "natural"], {
                    "default": "vivid"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            },
            "optional": {
                "n": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "result", "raw_json")
    FUNCTION = "generate"
    CATEGORY = "API365"

    def create_default_image(self, size):
        """创建默认占位图"""
        width, height = map(int, size.split('x'))
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

    def generate(self, prompt, model, size, quality, style, api_key, n=1):
        """生成图像"""
        session = None
        try:
            # 构建请求payload
            payload = {
                "model": model,
                "prompt": prompt,
                "n": n,
                "size": size,
                "response_format": "url"
            }
            
            # DALL-E 3 特有参数
            if model == "dall-e-3":
                payload["quality"] = quality
                payload["style"] = style

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            url = "https://openai.api365.cloud/v1/images/generations"
            
            session = requests.Session()
            response = session.post(url, headers=headers, json=payload, timeout=180)

            if response.status_code != 200:
                error_msg = f"API请求失败: {response.status_code} - {response.text}"
                print(f"Error: {error_msg}")
                return (self.create_default_image(size), error_msg, "None")
            
            resp_json = response.json()
            
            # 解析响应
            if "data" in resp_json and resp_json["data"]:
                image_data = resp_json["data"][0]
                if "url" in image_data:
                    output_image = self.decode_image(image_data["url"])
                    return (output_image, "图片生成成功", json.dumps(resp_json, ensure_ascii=False))
                else:
                    return (self.create_default_image(size), "API未返回图片URL", json.dumps(resp_json, ensure_ascii=False))
            else:
                return (self.create_default_image(size), "API未返回图片", json.dumps(resp_json, ensure_ascii=False))
                
        except Exception as e:
            error_msg = f"处理过程中发生错误: {str(e)}"
            print(f"Error: {error_msg}")
            return (self.create_default_image(size), error_msg, "None")
        
        finally:
            if session:
                session.close()