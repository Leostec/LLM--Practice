from langchain_core.messages import HumanMessage, SystemMessage
# 导入所需的库
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
# 设置 OpenAI API 密钥（请确保将 'your_openai_api_key' 替换为您的实际密钥）
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# 定义一个 Robot 类，包含模型和 RAG 工具
class Robot():
    def __init__(self):
        # 初始化 ChatOpenAI 模型
        self.model = ChatOpenAI(model="gpt-4o-mini")  # 或者使用您实际的模型名称
        self.store = {}

        # 初始化属性和生肖的检索器
        self.shuxing_retriever = self.init_shuxing_retriever()
        self.shengxiao_retriever = self.init_shengxiao_retriever()

    # 初始化属性检索器
    def init_shuxing_retriever(self):
        vectorstore = Chroma(persist_directory="/Users/leo/研究生/实习/北大青鸟/项目制作/配对小精灵/shuxing_chroma_save", embedding_function=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        return retriever

    # 初始化生肖检索器
    def init_shengxiao_retriever(self):
        vectorstore = Chroma(persist_directory="/Users/leo/研究生/实习/北大青鸟/项目制作/配对小精灵/shengxiao_chroma_save", embedding_function=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        return retriever

    # 定义主方法 run，用于运行整个流程
    def run(self):
        # 高级系统提示，传递给模型
        advanced_system_prompt = """
        你是一个名为“配对小精灵”的AI助手，专注于生肖和属性配对相关问题。
        你的任务是：

        1. 根据用户的问题，调用RAG工具，提供最准确的生肖、属性和配对信息。

        2. 在回答时：
            - 保持语气温和、友好，并提供有用的建议。
            - 请分步骤说明复杂的查询结果。
            - 确保答案简洁、清晰，但必要时可以详细解释背景知识。

        3. 处理特殊情况：
            - 如果数据不可用或无法确定结果，请礼貌地告知用户原因，并建议用户提供更多信息（如具体出生时间）。
            - 如果用户输入的格式不正确，请友好地提醒并提供正确的输入格式。

        在每次回答中，请根据用户的问题选择合适的工具，并以用户可以理解的方式解释结果。
        """

        # 初始化对话历史
        conversation_history = [SystemMessage(content=advanced_system_prompt)]

        input_text = ""
        while input_text.lower() != "exit":
            input_text = input("请输入（输入 'exit' 退出）：")
            if input_text.lower() == "exit":
                break

            # 使用属性和生肖的检索器获取相关文档
            shuxing_docs = self.shuxing_retriever.get_relevant_documents(input_text)
            shengxiao_docs = self.shengxiao_retriever.get_relevant_documents(input_text)

            # 合并检索到的文档内容
            retrieved_texts = "\n".join([doc.page_content for doc in shuxing_docs + shengxiao_docs])

            # 构建完整的提示，包括用户输入和检索到的文档
            full_prompt = f"用户的问题：{input_text}\n\n参考资料：\n{retrieved_texts}\n\n请根据以上信息回答用户的问题。"

            # 将提示添加到对话历史
            conversation_history.append(HumanMessage(content=full_prompt))

            # 调用模型生成回复
            response = self.model(conversation_history)

            # 获取助手的回复
            assistant_reply = response.content

            # 输出回复
            print(f"配对小精灵：{assistant_reply}")

            # 将助手的回复添加到对话历史
            conversation_history.append(SystemMessage(content=assistant_reply))

# 如果该脚本是主脚本，则创建 Robot 实例并调用 run 方法
if __name__ == '__main__':
    robot = Robot()
    robot.run()
