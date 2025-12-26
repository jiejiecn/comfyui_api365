"""
API365 ComfyUI 自定义节点实现
"""

import requests
import json
import base64
import time, os, random, re
from io import BytesIO
from PIL import Image
import numpy as np
import torch


# 获取当前目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(CURRENT_DIR, 'api365_config.json')

def get_config():
    """获取配置文件"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        return {}
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        return {}

def save_config(config):
    """保存配置文件 - 已禁用"""
    # print(f"[BananaIntegrated] 提示：配置文件保存功能已禁用，API密钥不会保存到本地")
    pass


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


class PromptNode:
    """提示词节点 - 允许用户输入文字作为输出"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "在这里输入你的提示词..."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "process"
    CATEGORY = "API365"
    
    def process(self, text):
        """处理输入的文本并返回"""
        return (text,)


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
                    "default": "",
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
        """
        解析谷歌官方 Gemini API 响应
        """
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
            print(f"响应内容: {json.dumps(response_json, indent=2, ensure_ascii=False)[:500]}")
            raise Exception(f"响应解析失败: {str(e)}")


    def decode_image(self, image_url):
        """下载或解码图片"""
        try:
            if image_url.startswith('data:image/'):
                # Base64图片
                base64_data = image_url.split(',', 1)[1]
                image_data = base64.b64decode(base64_data)
                pil_image = Image.open(BytesIO(image_data))
            else:
                # HTTP URL图片 - 使用独立session避免代理连接复用问题
                session = requests.Session()
                session.trust_env = True
                try:
                    response = session.get(image_url, timeout=60)
                    response.raise_for_status()
                    pil_image = Image.open(BytesIO(response.content))
                finally:
                    session.close()
            
            # 转换为RGB模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            print(f"[BananaIntegrated] 图片解码成功: {pil_image.size}")
            return pil2tensor(pil_image)
            
        except Exception as e:
            print(f"[BananaIntegrated] 图片解码失败: {str(e)}")
            raise
    
    def calculate_dimensions(self, aspect_ratio, image_size):
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

    def create_default_image(self, aspect_ratio, image_size):
        """创建默认占位图"""
        width, height = self.calculate_dimensions(aspect_ratio, image_size)
        
        # 创建白色图片
        img = Image.new('RGB', (width, height), color='white')
        return pil2tensor(img)

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
                return (self.create_default_image(aspect_ratio, image_size), error_msg)
            
            # 解析响应
            resp_json = response.json()
            result = self.parse_response(resp_json)
            
            if result['success']:
                # 解码第一张图片（如果有）
                output_image = self.decode_image(result['images'][0])
                return (output_image, result['text'] if result['text'] else "图片生成成功", resp_json)
            else:
                return (self.create_default_image(aspect_ratio, image_size), "API未返回图片", resp_json)
                
                
        except Exception as e:
            error_msg = f"处理过程中发生错误: {str(e)}"
            print(f"Error: {error_msg}")
            return (self.create_default_image(aspect_ratio, image_size), error_msg, "None")
        
        finally:
            if session:
                session.close()


# 节点映射
NODE_CLASS_MAPPINGS = {
    "API365_Prompt": PromptNode,
    "API365_NanoBanana2": NanoBanana2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "API365_Prompt": "提示词",
    "API365_NanoBanana2": "Nano Banana 图像生成&编辑",
}