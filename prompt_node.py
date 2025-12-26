"""
Prompt 节点实现
"""


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