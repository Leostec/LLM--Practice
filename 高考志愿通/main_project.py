from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import openai
import os
import sqlite3  # 新增
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置 OpenAI API 密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

# 使用 OpenAI 的嵌入模型
embeddings = OpenAIEmbeddings()

# 调整持久化目录为之前构建向量库的路径
persist_directory = "/Users/leo/研究生/实习/北大青鸟/项目制作/高考政策通/chromadb_rag"

# 加载持久化的 Chroma 向量库
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_directory
)

def get_embedding(input_text):
    # 使用 OpenAI 的嵌入模型生成文本的嵌入
    return embeddings.embed_query(input_text)

class Robot():
    def __init__(self):
        self.embedding = ""
        # 连接到 SQLite 数据库
        db_path = "/Users/leo/研究生/实习/北大青鸟/项目制作/高考政策通/database.db"
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def __del__(self):
        # 关闭数据库连接
        self.conn.close()

    def RAG(self, input_text):
        # Step 1: 使用向量库检索与用户输入最相关的内容
        search_embedding = get_embedding(input_text)
        # 执行向量相似度搜索，获取最相关的文档
        first_results = vectorstore.similarity_search_by_vector(
            embedding=search_embedding,
            k=5  # 您可以根据需要调整返回结果的数量
        )

        result_all = ""
        retrieved_info = set()  # 用于避免重复信息

        for doc in first_results:
            metadata = doc.metadata
            # 从元数据中获取相关信息，例如学校名称、专业等
            school_name = metadata.get('学校名称', '未知学校')
            major = metadata.get('专业名称', '未知专业')
            # 组合结果
            key = f"{school_name}-{major}"
            if key in retrieved_info:
                continue  # 跳过已处理的信息
            retrieved_info.add(key)
            result_documents = doc.page_content
            # 您可以根据需要从元数据中获取更多信息
            result_all += f"【学校】：{school_name}\n【专业】：{major}\n{result_documents}\n\n"

        # 如果没有找到相关信息，提示用户
        if not result_all:
            result_all = "抱歉，没有找到任何相关的参考资料。"

        return result_all

    def run(self):
        history_list = [{
            "role": "system",
            "content": (
                "你是一个高考志愿通，对于用户提问的问题，你需要按照给出的【参考资料】对问题进行回答。"
                "你的回答需要按照以下两个步骤："
                "1.分析用户问题和参考资料，判断是否有【参考资料】可以解答用户的问题，如果有则说明【参考资料】的名称，"
                "如果没有，则首先告知用户没有任何可参考的资料，需要注意答案的准确性。"
                "2.根据资料内容对问题进行解答，若用户希望根据高考分数得到志愿推荐，那么首先关注学校的投档分，越接近越好，从好的学校推荐，并结合【参考资料】中的学校信息来说明。"
            )
        }]
        input_text = ""
        while input_text != "exit":
            input_text = input("请输入：")
            if input_text == "exit":
                break
            result_rag = self.RAG(input_text)
            # 将参考资料添加到用户输入中，供后续对话模型使用
            input_with_context = f"{input_text}\n参考资料：\n{result_rag}"
            result = chat_with_openai(input_with_context, history_list)
            print(result)
            message_input = {"role": "user", "content": input_with_context}
            history_list.append(message_input)
            message_output = {"role": "assistant", "content": result}
            history_list.append(message_output)

def chat_with_openai(input_text, history_list):
    # 将用户输入加入对话历史
    messages = history_list.copy()
    messages.append({"role": "user", "content": input_text})

    # 使用 OpenAI 的 ChatCompletion 进行对话
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # 建议使用已知的模型名称
        messages=messages
    )

    # 提取完整的回复内容
    result = response.choices[0].message.content
    return result

if __name__ == '__main__':
    robot = Robot()
    robot.run()
