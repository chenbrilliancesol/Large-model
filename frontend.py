import streamlit as st
from typing import List, Dict
from ollama_client import OllamaClient


class ChatFrontend:
    def __init__(self, title: str = "黑马智聊机器人"):
        """
        初始化前端界面

        Args:
            title: 应用标题
        """
        self.title = title
        self.ollama_client = OllamaClient()
        self.setup_page()

    def setup_page(self):
        """设置页面配置"""
        st.set_page_config(page_title=self.title, page_icon="🤖", layout="wide")
        st.title(self.title)

    def initialize_session_state(self):
        """初始化会话状态"""
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "你好，我是黑马智聊机器人，有什么可以帮助你的么？"}
            ]

    def display_chat_history(self):
        """显示聊天历史"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_user_input(self):
        """处理用户输入"""
        prompt = st.chat_input("请输入您要咨询的问题：")
        if prompt:
            # 添加用户消息到历史
            st.session_state.messages.append({"role": "user", "content": prompt})

            # 显示用户消息
            with st.chat_message("user"):
                st.markdown(prompt)

            # 获取并显示助手回复
            with st.chat_message("assistant"):
                with st.spinner("AI小助手正在思考中..."):
                    response = self.ollama_client.get_response(st.session_state.messages)
                st.markdown(response)

            # 添加助手回复到历史
            st.session_state.messages.append({"role": "assistant", "content": response})

    def run(self):
        """运行应用"""
        self.initialize_session_state()
        self.display_chat_history()
        self.handle_user_input()


class AdvancedChatFrontend(ChatFrontend):
    """扩展功能的前端界面"""

    def __init__(self, title: str = "黑马智聊机器人-高级版"):
        super().__init__(title)
        self.setup_sidebar()

    def setup_sidebar(self):
        """设置侧边栏"""
        with st.sidebar:
            st.header("功能选择")
            self.function_choice = st.radio(
                "选择功能:",
                ["普通聊天", "文本扩写", "代码生成", "情感分析"]
            )

            st.divider()
            st.header("模型设置")
            model_options = ["qwen2:0.5b", "deepseek-r1:8b", "qwen2.5:7b"]
            self.selected_model = st.selectbox("选择模型:", model_options)
            self.ollama_client = OllamaClient(self.selected_model)

            if st.button("清空聊天记录"):
                st.session_state.messages = [
                    {"role": "assistant", "content": "你好，我是黑马智聊机器人，有什么可以帮助你的么？"}
                ]
                st.rerun()

    def handle_special_functions(self):
        """处理特殊功能"""
        if self.function_choice != "普通聊天":
            st.header(f"{self.function_choice}功能")

            if self.function_choice == "文本扩写":
                text_input = st.text_area("请输入要续写的文本:", "从前有座山，山里有个庙，")
                if st.button("生成续写"):
                    with st.spinner("正在生成续写内容..."):
                        result = self.ollama_client.text_completion(text_input)
                    st.write("续写结果:")
                    st.success(result)

            elif self.function_choice == "代码生成":
                code_request = st.text_area("请输入代码需求:", "求两个数的最大公约数")
                if st.button("生成代码"):
                    with st.spinner("正在生成代码..."):
                        result = self.ollama_client.code_generation(code_request)
                    st.write("生成的代码:")
                    st.code(result, language="python")

            elif self.function_choice == "情感分析":
                feedback = st.text_area("请输入用户反馈:", "性价比不高，我觉得不值这个价钱。")
                if st.button("分析情感"):
                    with st.spinner("正在分析情感..."):
                        result = self.ollama_client.sentiment_analysis(feedback)
                    st.write("分析结果:")
                    st.info(result)

    def run(self):
        """运行高级应用"""
        self.initialize_session_state()

        # 根据功能选择显示不同内容
        if self.function_choice == "普通聊天":
            self.display_chat_history()
            self.handle_user_input()
        else:
            self.handle_special_functions()