import requests
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain.agents import AgentExecutor

from langchain.agents import create_tool_calling_agent

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain import hub
from langchain.tools.retriever import create_retriever_tool

import os
os.environ['OPENAI_API_KEY'] = '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'
os.environ["TAVILY_API_KEY"] = "<你的apikey>"

class Robot():
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-3.5-turbo")

    def RAG(self):
        vectorstore = Chroma(persist_directory="./chroma_save", embedding_function=OpenAIEmbeddings())
        retriever = RunnableLambda(vectorstore.similarity_search).bind(k=3)  # 选择最佳门
        retriever_tool = create_retriever_tool(
            retriever,
            "travel_city",
            "搜索关于适合旅游的城市，任何有关了解旅游城市的信息，都可以使用该工具。",
        )

        return retriever_tool

    @tool
    def get_current_weather(city: str) -> str:
        """
        Returns weather ,use this for any questions related to knowing current weather.
        :param city:
        :return:
        """
        api_key = "<你的apikey>"  # 替换为你自己的API密钥
        citycodeurl = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=5&appid={api_key}"
        response = requests.get(citycodeurl)
        data = response.json()
        if response.status_code == 200:
            lat = data[0]["lat"]
            lon = data[0]["lon"]

        base_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"

        response = requests.get(base_url)
        data = response.json()

        if response.status_code == 200:
            weather = {
                'temperature': data['main']['temp'],
                'description': data['weather'][0]['description'],
                'city': data['name'],
                'country': data['sys']['country']
            }
            return weather
        else:
            return {"error": data.get("message", "An error occurred.")}



    def search(self):
        search_tool = TavilySearchResults(max_results=2)
        return search_tool


    def run(self):
        rag_tool = self.RAG()
        search_tool = self.search()
        weather_tool = self.get_current_weather

        tools = [rag_tool, weather_tool, search_tool]

        prompt = hub.pull("hwchase17/openai-functions-agent")

        # 初始化Agent
        agent = create_tool_calling_agent(self.model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent,
                                       tools=tools,
                                       handle_parsing_errors=True,
                                       verbose=True)
        input_text = ""
        while input_text != "exit":
            input_text = input("请输入问题：")
            if input_text == "exit":
                break

            # 执行AgentExecutor
            agent_executor.invoke({"input": f"{input_text}"})


if __name__ == '__main__':
    robot = Robot()
    robot.run()
