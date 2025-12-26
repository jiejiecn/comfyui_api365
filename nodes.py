"""
API365 ComfyUI 自定义节点实现
"""

# 导入所有节点
from .prompt_node import PromptNode
from .nanobana_node import NanoBanana2Node
from .imagen4_node import Imagen4Node
from .gptimage_node import GPTImageNode


# 节点映射
NODE_CLASS_MAPPINGS = {
    "API365_Prompt": PromptNode,
    "API365_NanoBanana2": NanoBanana2Node,
    "API365_Imagen4": Imagen4Node,
    "API365_GPTImage": GPTImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "API365_Prompt": "提示词",
    "API365_NanoBanana2": "Nano Banana 图像生成&编辑",
    "API365_Imagen4": "Imagen4 图像生成",
    "API365_GPTImage": "GPT Image图像生成",
}