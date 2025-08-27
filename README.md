# 本地智聊机器人

基于Ollama和Streamlit开发的本地聊天机器人应用。

## 功能特点

1. **普通聊天**: 与AI进行自然语言对话
2. **文本扩写**: 输入开头，AI自动续写内容
3. **代码生成**: 根据描述生成Python代码
4. **情感分析**: 分析用户反馈的情感倾向

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行应用

bash

```
streamlit run app.py
```

## 使用前提

1. 安装并运行Ollama

2. 下载所需模型，例如:

   bash

   ```
   ollama pull qwen2:0.5b
   ollama pull deepseek-r1:8b
   ollama pull qwen2.5:7b
   ```

## 项目结构

- `app.py`: 主应用入口
- `ollama_client.py`: Ollama客户端类，处理与模型的交互
- `frontend.py`: Streamlit前端界面类
- `utils.py`: 工具函数
- `requirements.txt`: 项目依赖列表

## 注意事项

1. 确保Ollama服务正在运行
2. 首次使用需要下载模型，可能需要较长时间
3. 对话历史限制为最近50条消息，避免超出模型上下文限制

## 运行说明

1. 确保已安装Ollama并下载所需模型
2. 安装所有依赖: `pip install -r requirements.txt`
3. 运行应用: `streamlit run app.py`
4. 在浏览器中打开显示的URL(通常是http://localhost:8501)

这个实现采用了面向对象的设计，将前端和后端逻辑分离，使得代码更加模块化和可维护。前端提供了基础聊天功能和高级功能(文本扩写、代码生成、情感分析)，用户可以通过侧边栏选择不同的功能和模型。