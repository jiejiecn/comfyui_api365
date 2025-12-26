"""
Imagen4 节点实现
"""

import requests
import ujson as json
import random
from image_utils import decode_image, calculate_dimensions, create_default_image


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
                    "default": "tk-xxxxxx",
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
                return (create_default_image(None, aspect_ratio, image_size), error_msg, "None")
            
            resp_json = response.json()
            
            # 解析响应
            if "predictions" in resp_json and resp_json["predictions"]:
                image_data = resp_json["predictions"][0]
                if "bytesBase64Encoded" in image_data:
                    # Base64格式
                    image_url = f"data:image/png;base64,{image_data['bytesBase64Encoded']}"
                    output_image = decode_image(image_url)
                    return (output_image, "图片生成成功", resp_json)
                else:
                    return (create_default_image(None, aspect_ratio, image_size), "API未返回图片内容", resp_json)
            else:
                return (create_default_image(None, aspect_ratio, image_size), "API未返回图片内容", resp_json)
                
        except Exception as e:
            error_msg = f"处理过程中发生错误: {str(e)}"
            print(f"Error: {error_msg}")
            return (create_default_image(None, aspect_ratio, image_size), error_msg, "None")
        
        finally:
            if session:
                session.close()