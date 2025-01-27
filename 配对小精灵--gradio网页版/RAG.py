# 导入所需的库，PDFMinerLoader用于加载PDF文件，RecursiveCharacterTextSplitter用于将文本分割为小块，
# Chroma用于存储和检索向量化的文本，OpenAIEmbeddings用于生成文本的嵌入向量。



# 设置API密钥的环境变量，用于访问OpenAI API
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["OPENAI_API_KEY"] = "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"

def file_to_chroma(file_path):

    # 获取文件夹中所有的 Excel 文件
    excel_files = [f for f in os.listdir(file_path) if f.endswith('.xlsx') or f.endswith('.xls')]
    # 遍历文件夹中的每个 Excel 文件
    for file_name in excel_files:
        fold_path = os.path.join(file_path, file_name)

        # 读取 Excel 文件
        df = pd.read_excel(fold_path)
        if "属性" in file_name:
            # 初始化一个空列表来存储所有文档
            all_documents = []

            # 遍历数据框的每一行
            for index, row in df.iterrows():
                # 构建每一行的文档内容，包含男士属性、女士属性和配对结果
                document_content = f"男士属性: {row['male']}\n女士属性: {row['female']}\n配对结果: {row['result']}"
                # 将文档内容添加到列表中
                all_documents.append(document_content)

            # # 使用 RecursiveCharacterTextSplitter 将文档拆分为较小的文本块（如果文档较长）
            # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            # documents = text_splitter.split_documents(all_documents)

            # 使用 OpenAI 的嵌入模型将分割后的文档转换为向量，并存储到 Chroma 向量数据库中
            vectorstore = Chroma.from_texts(
                all_documents,  # 被拆分的文档
                embedding=OpenAIEmbeddings(),  # 使用 OpenAI 的嵌入向量模型
                persist_directory="/Users/leo/研究生/实习/北大青鸟/项目制作/配对小精灵/shuxing_chroma_save"  # 指定向量数据库保存的目录
            )
            vectorstore.persist()  # 确保数据持久化保存到磁盘中

        elif "生肖" in file_name:
            # 初始化一个空列表来存储所有文档
            all_documents = []

            # 遍历数据框的每一行
            for index, row in df.iterrows():
                # 构建每一行的文档内容，包含男士生肖、女士生肖和配对结果
                document_content = f"男士生肖: {row['male']}\n女士生肖: {row['female']}\n配对结果: {row['result']}"
                # 将文档内容添加到列表中
                all_documents.append(document_content)

            # # 使用 RecursiveCharacterTextSplitter 将文档拆分为较小的文本块（如果文档较长）
            # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            # documents = text_splitter.split_documents(all_documents)

            # 使用 OpenAI 的嵌入模型将分割后的文档转换为向量，并存储到 Chroma 向量数据库中
            vectorstore = Chroma.from_texts(
                all_documents,  # 被拆分的文档
                embedding=OpenAIEmbeddings(),  # 使用 OpenAI 的嵌入向量模型
                persist_directory="/Users/leo/研究生/实习/北大青鸟/项目制作/配对小精灵/shengxiao_chroma_save"  # 指定向量数据库保存的目录
            )
            vectorstore.persist()  # 确保数据持久化保存到磁盘中

    return "知识库导入完成！"

# 定义一个函数delete_chroma，用于删除Chroma向量数据库中的所有内容
def delete_chroma():
    # 打开Chroma数据库
    vectorstore = Chroma(persist_directory="./chroma_save")
    # 删除向量库中的所有集合
    vectorstore.delete_collection()
    # 返回一个操作完成的消息
    return "向量库删除完成！"

# 主程序入口
if __name__ == '__main__':
    # 如果需要删除向量库，取消注释以下代码：
    # result = delete_chroma()

    # 指定包含PDF文件的文件夹路径
    file_path = "/Users/leo/研究生/实习/北大青鸟/项目制作/配对小精灵/"
    # 将文件夹中的所有excel文件加载并存储到向量库
    result = file_to_chroma(file_path)
    # 输出结果
    print(result)
