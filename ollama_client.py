import ollama
from typing import List, Dict


class OllamaClient:
    def __init__(self, model_name: str = "qwen2:0.5b"):
        """
        初始化Ollama客户端

        Args:
            model_name: 模型名称，默认为qwen2:0.5b
        """
        self.model_name = model_name

    def get_response(self, messages: List[Dict]) -> str:
        """
        获取模型回复

        Args:
            messages: 消息列表，格式为[{'role': 'user/assistant', 'content': '消息内容'}]

        Returns:
            str: 模型生成的回复内容
        """
        try:
            # 限制历史消息长度，避免超出上下文限制
            truncated_messages = messages[-50:]
            response = ollama.chat(model=self.model_name, messages=truncated_messages)
            return response['message']['content']
        except Exception as e:
            return f"抱歉，发生错误: {str(e)}"

    def text_completion(self, prompt: str) -> str:
        """
        文本续写功能

        Args:
            prompt: 提示文本

        Returns:
            str: 续写后的文本
        """
        messages = [{'role': 'user', 'content': prompt}]
        return self.get_response(messages)

    def code_generation(self, requirement: str) -> str:
        """
        代码生成功能

        Args:
            requirement: 代码需求描述

        Returns:
            str: 生成的代码
        """
        prompt = f"请为以下功能生成一段Python代码：\n{requirement}"
        return self.get_response([{'role': 'user', 'content': prompt}])

    def sentiment_analysis(self, feedback: str) -> str:
        """
        情感分析功能

        Args:
            feedback: 用户反馈文本

        Returns:
            str: 情感分类结果
        """
        prompt = f"""
        你需要对用户的反馈进行原因分类。
        分类包括：价格过高、售后支持不足、产品使用体验不佳、其他。
        回答格式为：分类结果：xx。
        用户的问题是：{feedback}
        """
        return self.get_response([{'role': 'user', 'content': prompt}])
