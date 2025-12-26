"""
GPTImage 节点实现
"""

import requests
import ujson as json
from image_utils import decode_image, create_default_image 


class GPTImageNode:
    """GPTImage节点 - OpenAI DALL-E 图像生成"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": ("STRING", {"forceInput": True}),

                "model": (["gpt-image-1.5"], {
                    "default": "gpt-image-1.5"
                }),
                "size": (["1024x1024", "1536x1024", "1024x1536"], {
                    "default": "1024x1024"
                }),
                "quality": (["auto", "high", "medium", "low"], {
                    "default": "auto"
                }),
                "background": (["transparent", "opaque", "auto"], {
                    "default": "auto"
                }),
                "api_key": ("STRING", {
                    "default": "tk-xxxxxx",
                    "multiline": False
                }),
            },
            "optional": {

            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "result", "raw_json")
    FUNCTION = "generate"
    CATEGORY = "API365"

    def build_api365_payload(self, user_prompt, model, image_size, image_quality, image_background):
        payload = {
            "model": model,
            "prompt": user_prompt,
            "size": image_size,
            "quality": image_quality,
            "background": image_background,
            "n": 1
        }

        return payload




    def generate(self, **kwargs):
        """生成图像"""

        user_prompt = kwargs.get("user_prompt", None)
        model = kwargs.get("model", "")
        image_size = kwargs.get("size", "1024x1024")
        image_quality = kwargs.get("quality", "auto")
        image_background = kwargs.get("background", "auto")
        api_key = kwargs.get("api_key", "")


        session = None
        try:
            # 构建请求payload
            payload = self.build_api365_payload(user_prompt, model, image_size, image_quality, image_background)

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
                return (create_default_image(image_size), error_msg, "None")
            
            resp_json = response.json()
            
            # 解析响应
            if "data" in resp_json and resp_json["data"]:
                image_data = resp_json["data"][0]
                if "b64_json" in image_data:
                    image_url = f"data:image/png;base64,{image_data['b64_json']}"
                    output_image = decode_image(image_url)
                    return (output_image, "图片生成成功", resp_json)
                else:
                    return (create_default_image(image_size), "API未返回图片内容", resp_json)
            else:
                return (create_default_image(image_size), "API未返回图片内容", resp_json)
                
        except Exception as e:
            error_msg = f"处理过程中发生错误: {str(e)}"
            print(f"Error: {error_msg}")
            return (create_default_image(image_size), error_msg, "None")
        
        finally:
            if session:
                session.close()