# API365 ComfyUI 自定义节点

API365是一个为ComfyUI设计的自定义节点包，提供两个实用的节点功能。

## 功能特性

### 1. 提示词节点 (API365_Prompt)
- 允许用户输入多行文本作为提示词
- 输出类型：STRING
- 用途：为其他节点提供文本输入

### 2. nano banana2节点 (API365_NanoBanana2)
- 接受系统提示词和用户提示词文本输入
- 接受八个图片输入
- 支持模型选择（model1, model2, model3）
- 支持尺寸选择（1024x1024, 2048x2048, 4096x4096）
- 支持比例选择（1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3）
- 支持种子值设置（0-2147483647）
- 将数据打包后调用后端API接口
- 输出API调用的返回结果
- 支持自定义API端点和API密钥

## 安装方法

1. 将此项目文件夹复制到ComfyUI的`custom_nodes`目录下
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```
3. 重启ComfyUI

## 使用方法

### 提示词节点
1. 在ComfyUI中添加"提示词"节点
2. 在文本框中输入你的提示词内容
3. 连接到其他需要文本输入的节点

### nano banana2节点
1. 添加"nano banana2"节点
2. 连接系统提示词和用户提示词输入
3. 连接八个图片输入
4. 选择合适的模型
5. 设置输出尺寸（1024x1024, 2048x2048, 4096x4096）
6. 选择图片比例（1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3）
7. 设置种子值（用于结果的可重现性）
8. 配置API端点URL
9. 如需要，配置API密钥
10. 运行工作流，查看API返回结果

## API接口规范

nano banana2节点发送的请求格式：
```json
{
  "system_prompt": "系统提示词",
  "user_prompt": "用户输入的提示词",
  "images": ["base64编码的图片1", "base64编码的图片2", "base64编码的图片3", "base64编码的图片4", "base64编码的图片5", "base64编码的图片6", "base64编码的图片7", "base64编码的图片8"],
  "model": "选择的模型",
  "size": "输出尺寸",
  "ratio": "图片比例",
  "seed": 种子值,
  "timestamp": "时间戳"
}
```

请求头：
- Content-Type: application/json
- User-Agent: API365-ComfyUI/1.0
- Authorization: Bearer {api_key} (如果提供了API密钥)

## 注意事项

- 确保API端点可以正常访问
- 图片会被转换为PNG格式的base64编码
- 请求超时时间为30秒
- 支持中文输入输出

## 版本信息

- 版本：1.0.0
- 兼容ComfyUI版本：最新版本
- Python版本要求：3.8+