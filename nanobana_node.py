"""
Nano Banana 节点实现
"""

import requests
import ujson as json
import base64
import time, os, random, re
from io import BytesIO
from image_utils import pil2tensor, tensor2pil, decode_image, calculate_dimensions, create_default_image



class NanoBanana2Node:
    """nano banana2节点 - 接受提示词和8个图片输入，调用后端接口"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                
                "user_prompt": ("STRING", {"forceInput": True}),
                
                "model": (["gemini-3-pro-image-preview", "gemini-2.5-flash-image"], {
                    "default": "gemini-3-pro-image-preview"
                }),
                "size": (["1K", "2K", "4K"], {
                    "default": "1K"
                }),
                "ratio": (["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3"], {
                    "default": "1:1"
                }),
                "api_key": ("STRING", {
                    "default": "tk-xxxxxx",
                    "multiline": False
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {"forceInput": True}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "result", "raw_json")
    FUNCTION = "generate"
    CATEGORY = "API365"

    
    def add_random_variation(self, prompt, seed=0):
        rng = random.Random(seed)
        random_id = rng.randint(10000, 99999)
        
        return f"{prompt} [variation-{random_id}]"

    def build_api365_payload(self, system_prompt, user_prompt, input_images, aspect_ratio, image_size, seed):
        """构建谷歌官方 Gemini API 格式的请求"""
        # 添加随机变化因子
        varied_prompt = self.add_random_variation(user_prompt, seed)
        
        # 构建端口号到数组索引的映射
        port_to_array_map = {}  # 端口号 -> 数组索引
        array_idx = 0
        for port_idx, img in enumerate(input_images, 1):
            if img is not None:
                array_idx += 1
                port_to_array_map[port_idx] = array_idx
        
        # 自动转换提示词中的图片引用（端口号 -> 数组索引）
        for port_num, array_num in port_to_array_map.items():
            # 替换各种可能的引用格式
            patterns = [
                (rf'图{port_num}(?![0-9])', f'图{array_num}'),  # 图2 -> 图1
                (rf'图片{port_num}(?![0-9])', f'图片{array_num}'),  # 图片2 -> 图片1
                (rf'第{port_num}张图', f'第{array_num}张图'),  # 第2张图 -> 第1张图
                (rf'第{port_num}个图', f'第{array_num}个图'),  # 第2个图 -> 第1个图
            ]
            for pattern, replacement in patterns:
                varied_prompt = re.sub(pattern, replacement, varied_prompt)

        # 构建 contents 数组（Google官方格式）
        parts = []
        
        # 添加所有输入图片 - 保持原始索引位置
        for i in range(len(input_images)):
            img_tensor = input_images[i]
            if img_tensor is not None:
                # 转换为PIL图片
                pil_image = tensor2pil(img_tensor)[0]
                
                # 转换为base64
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG", optimize=True, quality=95)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # 添加图片到parts
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_base64
                    }
                })

                print(f"已添加输入端口 {i+1} 的图片, Base64大小: {len(img_base64)} 字符")
        
        # 添加文本提示词
        parts.append({
            "text": varied_prompt
        })
        
        # 构建完整的payload
        if system_prompt and system_prompt.strip():
            payload = {
                "system_instruction": {
                    "parts": [{
                        "text": system_prompt,
                    }]
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": parts
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {
                        "aspectRatio": aspect_ratio,
                        "imageSize": image_size
                    }
                }
            }
        else:
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": parts
                }],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {
                        "aspectRatio": aspect_ratio,
                        "imageSize": image_size
                    }
                }
            }

        return payload


    def parse_response(self, response_json):
        try:
            if "candidates" not in response_json or not response_json["candidates"]:
                raise Exception("响应中没有candidates数据")
            
            candidate = response_json["candidates"][0]
            if "content" not in candidate or "parts" not in candidate["content"]:
                raise Exception("响应格式错误")
            
            parts = candidate["content"]["parts"]
            images = []
            text_parts = []
            
            for part in parts:
                # 跳过thought部分
                if part.get("thought", False):
                    continue
                    
                if "inlineData" in part:
                    # 图片数据
                    inline_data = part["inlineData"]
                    if "data" in inline_data:
                        # Base64格式
                        image_url = f"data:{inline_data.get('mimeType', 'image/png')};base64,{inline_data['data']}"
                        images.append(image_url)
                elif "text" in part:
                    # 文本数据
                    text_parts.append(part["text"])
            
            print(f"解析到 {len(images)} 张图片, {len(text_parts)} 段文本")
            
            return {
                'images': images,
                'text': '\n'.join(text_parts),
                'success': len(images) > 0
            }
            
        except Exception as e:
            print(f"响应解析错误: {str(e)}")
            print(f"响应内容: {json.dumps(response_json)[:500]}")
            raise Exception(f"响应解析失败: {str(e)}")

    def generate(self, **kwargs):
        """处理输入并调用后端接口"""

        system_prompt = kwargs.get("system_prompt", None)
        user_prompt = kwargs.get("user_prompt", None)
        model = kwargs.get("model", "")
        image_size = kwargs.get("size", "1K")
        aspect_ratio = kwargs.get("ratio", "1:1")
        api_key = kwargs.get("api_key", "")

        effective_seed = random.randint(0, 2147483647)

        input_images = []
        for i in range(1, 9):
            img = kwargs.get(f"image{i}")
            input_images.append(img)     

        session = None
        try:
            # post body
            payload = self.build_api365_payload(system_prompt, user_prompt, input_images, aspect_ratio, image_size, effective_seed)
            
            # post headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            url = f"https://gemini.api365.cloud/v1beta/models/{model}:generateContent"
            # 发送POST请求
            session = requests.Session()
            response = session.post(url, headers=headers, json=payload, timeout=180)

            if response.status_code != 200:
                error_msg = f"API请求失败: {response.status_code} - {response.text}"
                print(f"Error: {error_msg}")
                return (create_default_image(None, aspect_ratio, image_size), error_msg, "None")
            
            # 解析响应
            resp_json = response.json()
            result = self.parse_response(resp_json)
            
            if result['success']:
                # 解码第一张图片（如果有）
                output_image = decode_image(result['images'][0])
                return (output_image, "图片生成成功", resp_json)
            else:
                return (create_default_image(None, aspect_ratio, image_size), "API未返回图片", resp_json)
                
                
        except Exception as e:
            error_msg = f"处理过程中发生错误: {str(e)}"
            print(f"Error: {error_msg}")
            return (create_default_image(None, aspect_ratio, image_size), error_msg, "None")
        
        finally:
            if session:
                session.close()