from frontend import ChatFrontend, AdvancedChatFrontend
import streamlit as st

def main():
    # 创建前端实例并运行
    # frontend = ChatFrontend()  # 基础版本
    frontend = AdvancedChatFrontend()  # 高级版本
    frontend.run()

if __name__ == "__main__":
    main()