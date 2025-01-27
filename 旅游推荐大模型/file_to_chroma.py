from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import os
os.environ["OPENAI_API_KEY"] = "<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>"

def file_to_chroma(file_path):
    file_names = os.listdir(file_path)
    for file in file_names:
        file_to_chroma_path = file_path + file
        print(file_to_chroma_path)
        # 导入需要向量化的文件
        loader = PDFMinerLoader(file_to_chroma_path)
        docs = loader.load()
        documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents,
            embedding=OpenAIEmbeddings(),
            persist_directory="./chroma_save"
        )
    return "向量库导入完成！"

def delete_chroma():
    vectorstore = Chroma(persist_directory="./chroma_save")
    vectorstore.delete_collection()
    return "向量库删除完成！"


if __name__ == '__main__':
    # result = delete_chroma()
    file_path = "./rag_file_path/"
    result = file_to_chroma(file_path)
    print(result)
