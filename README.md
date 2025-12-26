# API365 ComfyUI 自定义节点

API365是一个为ComfyUI设计的自定义节点包，提供直接调用api365 API的能力。

## 功能特性

### 1. 提示词节点 (API365_Prompt)
- 允许用户输入多行文本作为提示词
- 输出类型：STRING
- 用途：为其他节点提供文本输入

### 2. Nano Banana2节点 (API365_NanoBanana2)
- 接受系统提示词和用户提示词文本输入
- 接受八个图片输入
- 支持模型选择（model1, model2, model3）
- 支持尺寸选择（1024x1024, 2048x2048, 4096x4096）
- 支持比例选择（1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3）
- 将数据打包后调用后端API接口
- 输出API调用的返回结果

### 3. Imagen4 节点
- 接受用户提示词文本输入，排除提示词（可选）
- 支持模型选择（"imagen-4.0-generate-001", "imagen-4.0-fast-generate-001", "imagen-4.0-ultra-generate-001"）
- 支持尺寸选择（1024x1024, 2048x2048）
- 支持比例选择（1:1, 4:3, 3:4, 16:9, 9:16）
- 将数据打包后调用后端API接口
- 输出API调用的返回结果


## 安装方法

1. 将此项目文件夹复制到ComfyUI的`custom_nodes`目录下
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```
3. 重启ComfyUI


