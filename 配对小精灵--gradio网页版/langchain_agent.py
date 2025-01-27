import urllib.parse
from langchain_core.messages import HumanMessage, SystemMessage
import requests
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import uuid
import gradio as gr
from dotenv import load_dotenv

# 设置API密钥的环境变量，用于访问OpenAI
import os
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')  # 设定OpenAI API密钥
# 定义一个Robot类，包含多个与模型和工具相关的功能
class Robot():
    def __init__(self):
        # 初始化时，定义一个使用gpt-4o-mini模型的ChatOpenAI实例
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.store = {}
        # 初始化工具和代理
        self.tools = self.initialize_tools()
        self.agent_executor = self.initialize_agent_executor()
    def initialize_tools(self):
        shuxing_RAG = self.shuxing_RAG()
        shengxiao_RAG = self.shengxiao_RAG()
        get_shengxiao = self.get_shengxiao
        get_shuxing = self.get_shuxing
        xzys_tool = self.get_xzys
        xzpd_tool = self.get_xzpd

        tools = [xzys_tool, xzpd_tool, shuxing_RAG, shengxiao_RAG, get_shengxiao, get_shuxing]
        return tools

    def initialize_agent_executor(self):
        prompt = hub.pull("hwchase17/openai-functions-agent")

        agent = create_tool_calling_agent(self.model, self.tools, prompt)
        agent_executor = AgentExecutor(agent=agent,
                                       tools=self.tools,
                                       handle_parsing_errors=True,
                                       verbose=True)
        return agent_executor


    def shuxing_RAG(self):
        # 设置Chroma作为向量数据库，保存文件路径为"./chroma_save"
        vectorstore = Chroma(persist_directory="/Users/leo/研究生/实习/北大青鸟/项目制作/配对小精灵/shuxing_chroma_save", embedding_function=OpenAIEmbeddings())
        # 定义一个检索器，使用Chroma的相似性搜索功能，并设置k=1，返回1个最相似的结果
        retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)
        # 创建一个检索工具，用于搜索二者的属性配对情况
        retriever_tool = create_retriever_tool(
            retriever,
            "shuxing_peidui",  # 工具的名称
            "查询关于两个人属性配对或搜索综合配对情况的，都可以使用该工具。"  # 工具的描述
        )

        return retriever_tool  # 返回该工具实例

    def shengxiao_RAG(self):
        # 设置Chroma作为向量数据库，保存文件路径为"./chroma_save"
        vectorstore = Chroma(persist_directory="/Users/leo/研究生/实习/北大青鸟/项目制作/配对小精灵/shengxiao_chroma_save", embedding_function=OpenAIEmbeddings())
        # 定义一个检索器，使用Chroma的相似性搜索功能，并设置k=1，返回1个最相似的结果
        retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)
        # 创建一个检索工具，用于搜索二者的生肖配对情况
        retriever_tool = create_retriever_tool(
            retriever,
            "shengxiao_peidui",  # 工具的名称
            "查询关于两个人生肖配对或搜索综合配对情况的，都可以使用该工具。"  # 工具的描述
        )

        return retriever_tool  # 返回该工具实例
    @tool
    def get_xzys(birth: str) -> str:
        """
        根据输入的生日获取该人的星座运势。
        :param birth: 输入的生日日期，格式为 'YYYY-MM-DD'
        :return: 星座运势
        """

        # 定义每个星座的日期范围
        def get_constellation(birth):
            zodiac_signs = [
                ("capricorn", (1, 1), (1, 19)),  # 摩羯座
                ("aquarius", (1, 20), (2, 18)),  # 水瓶座
                ("pisces", (2, 19), (3, 20)),  # 双鱼座
                ("aries", (3, 21), (4, 19)),  # 白羊座
                ("taurus", (4, 20), (5, 20)),  # 金牛座
                ("gemini", (5, 21), (6, 20)),  # 双子座
                ("cancer", (6, 21), (7, 22)),  # 巨蟹座
                ("leo", (7, 23), (8, 22)),  # 狮子座
                ("virgo", (8, 23), (9, 22)),  # 处女座
                ("libra", (9, 23), (10, 22)),  # 天秤座
                ("scorpio", (10, 23), (11, 21)),  # 天蝎座
                ("sagittarius", (11, 22), (12, 21)),  # 射手座
                ("capricorn", (12, 22), (12, 31))  # 摩羯座
            ]

            # 将生日字符串转换为日期
            year, month, day = map(int, birth.split('-'))

            for sign, start_date, end_date in zodiac_signs:
                start_month, start_day = start_date
                end_month, end_day = end_date

                # 判断生日是否在当前星座的范围内
                if (month == start_month and day >= start_day) or (month == end_month and day <= end_day):
                    # 根据星座名输出对应的拼音
                    if sign == "aries":
                        return "baiyang"
                    elif sign == "pisces":
                        return "shuangyu"
                    elif sign == "taurus":
                        return "jinniu"
                    elif sign == "gemini":
                        return "shuangzi"
                    elif sign == "cancer":
                        return "juxie"
                    elif sign == "leo":
                        return "shizi"
                    elif sign == "virgo":
                        return "chunv"
                    elif sign == "libra":
                        return "tiancheng"
                    elif sign == "scorpio":
                        return "tianxie"
                    elif sign == "sagittarius":
                        return "sheshou"
                    elif sign == "capricorn":
                        return "mojie"
                    elif sign == "aquarius":
                        return "shuiping"

            return "未知星座"

        # 计算出星座
        star = get_constellation(birth)

        if star == "未知星座":
            return "无法识别星座"

        # 提取日期的MMDD格式，用于API请求
        month, day = birth.split('-')[1], birth.split('-')[2]
        date = f"{month}{day}"

        # API密钥
        api_key = os.getenv('xzys_api_key')  # 替换为你的API密钥

        # 构建请求URL
        url = f"https://route.showapi.com/872-1?appKey={api_key}"

        # 准备请求数据
        data = {
            'star': star,
            'needTomorrow': 1,  # 是否需要明天的数据
            'date': date,
            'needWeek': 1,  # 是否需要本周数据
            'needYear': 1,  # 是否需要今年数据
            'needMonth': 1  # 是否需要本月数据
        }

        # 对数据进行URL编码
        encoded_data = urllib.parse.urlencode(data)

        # 设置请求头
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        try:
            # 发送POST请求
            response = requests.post(url, data=encoded_data, headers=headers)

            # 检查响应状态码
            if response.status_code == 200:
                # 解析并返回API响应
                result = response.json()
                return result  # 返回JSON结果
            else:
                return f"请求失败，状态码：{response.status_code}"
        except requests.RequestException as e:
            return f"请求异常：{e}"

    # 定义星座配对查询工具
    @tool
    def get_xzpd(birth_male: str, birth_female: str) -> str:
        """
            根据输入的男生和女生的生日，分别获取两个人的星座运势。
            :param birth_male: 男生的生日，格式为 'YYYY-MM-DD'
            :param birth_female: 女生的生日，格式为 'YYYY-MM-DD'
            :return: 男生和女生的星座运势
        """


        # 定义每个星座的日期范围
        def get_constellation(birth):
            zodiac_signs = [
                ("capricorn", (1, 1), (1, 19)),  # 摩羯座
                ("aquarius", (1, 20), (2, 18)),  # 水瓶座
                ("pisces", (2, 19), (3, 20)),  # 双鱼座
                ("aries", (3, 21), (4, 19)),  # 白羊座
                ("taurus", (4, 20), (5, 20)),  # 金牛座
                ("gemini", (5, 21), (6, 20)),  # 双子座
                ("cancer", (6, 21), (7, 22)),  # 巨蟹座
                ("leo", (7, 23), (8, 22)),  # 狮子座
                ("virgo", (8, 23), (9, 22)),  # 处女座
                ("libra", (9, 23), (10, 22)),  # 天秤座
                ("scorpio", (10, 23), (11, 21)),  # 天蝎座
                ("sagittarius", (11, 22), (12, 21)),  # 射手座
                ("capricorn", (12, 22), (12, 31))  # 摩羯座
            ]

            # 将生日字符串转换为日期
            year, month, day = map(int, birth.split('-'))

            for sign, start_date, end_date in zodiac_signs:
                start_month, start_day = start_date
                end_month, end_day = end_date

                # 判断生日是否在当前星座的范围内
                if (month == start_month and day >= start_day) or (month == end_month and day <= end_day):
                    if sign == "aries":
                        return "白羊"
                    elif sign == "pisces":
                        return "双鱼"
                    elif sign == "taurus":
                        return "金牛"
                    elif sign == "gemini":
                        return "双子"
                    elif sign == "cancer":
                        return "巨蟹"
                    elif sign == "leo":
                        return "狮子"
                    elif sign == "virgo":
                        return "处女"
                    elif sign == "libra":
                        return "天秤"
                    elif sign == "scorpio":
                        return "天蝎"
                    elif sign == "sagittarius":
                        return "射手"
                    elif sign == "capricorn":
                        return "摩羯"
                    elif sign == "aquarius":
                        return "水瓶"

            return "未知星座"

        # 计算出星座
        star_male = get_constellation(birth_male)
        star_female = get_constellation(birth_female)

        if star_male == "未知星座" or star_female == "未知星座":
            return "无法识别星座"

        # 基本参数配置
        apiUrl = 'http://apis.juhe.cn/xzpd/query'  # 接口请求URL
        apiKey = os.getenv('xzpd_api_key')  # 在个人中心->我的数据,接口名称上方查看

        # 接口请求入参配置
        requestParams = {
            'key': apiKey,
            'men': star_male,
            'women': star_female,
        }

        # 发起接口网络请求
        response = requests.get(apiUrl, params=requestParams)

        # 解析响应结果
        if response.status_code == 200:
            responseResult = response.json()
            # 网络请求成功。可依据业务逻辑和接口文档说明自行处理。
            return responseResult
        else:
            # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
            return f"请求失败"
    @tool
    def get_shengxiao(birth: str) -> str:
        """
            根据输入的男生或女生的生日信息，获取个人的生肖。查询需要得到出生年月日和小时，如果用户输入的生日信息不全，需要提醒用户输入完整信息
            :param birth: 待查询人的生日，格式为 'YYYY-MM-DD-HH'
            :return: 个人的生肖
        """
        # 基本参数配置
        apiUrl = 'http://apis.juhe.cn/birthEight/query'  # 接口请求URL
        apiKey = os.getenv('shengxiao_shuxing')  # 在个人中心->我的数据,接口名称上方查看
        year, month, day ,hour= map(int, birth.split('-'))
        # 接口请求入参配置
        requestParams = {
            'key': apiKey,
            'year': year,
            'month': month,
            'day': day,
            'hour': hour,
        }

        # 发起接口网络请求
        response = requests.get(apiUrl, params=requestParams)

        # 解析响应结果
        if response.status_code == 200:
            responseResult = response.json()
            # 将 JSON 字符串解析为 Python 字典

            # 获取 "Animal" 字段
            animal = responseResult['result']['Animal']
            # 网络请求成功。可依据业务逻辑和接口文档说明自行处理。
            return animal
        else:
            # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
            return f"请求失败"

    @tool
    def get_shuxing(birth: str) -> str:
        """
            根据输入的男生或女生的生日信息，获取个人的属性。查询需要用户输入出生年月日和小时，要具体到小时，如果用户输入的生日信息不全未具体到小时或若最后一位hour为0，需要提醒用户输入完整信息
            :param birth: 待查询人的生日，格式为 'YYYY-MM-DD-HH'
            :return: 个人的属性
        """
        # 基本参数配置
        apiUrl = 'http://apis.juhe.cn/birthEight/query'  # 接口请求URL
        apiKey = os.getenv('shengxiao_shuxing')  # 在个人中心->我的数据,接口名称上方查看
        year, month, day, hour = map(int, birth.split('-'))
        # 接口请求入参配置
        requestParams = {
            'key': apiKey,
            'year': year,
            'month': month,
            'day': day,
            'hour': hour,
        }

        # 发起接口网络请求
        response = requests.get(apiUrl, params=requestParams)

        # 解析响应结果
        if response.status_code == 200:
            responseResult = response.json()
            # 将 JSON 字符串解析为 Python 字典

            # 获取 "Animal" 字段
            shu = responseResult['result']['eightAll']['shu']
            # 网络请求成功。可依据业务逻辑和接口文档说明自行处理。
            return shu
        else:
            # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
            return f"请求失败"

    # 定义历史对话函数
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
            # 添加高级系统提示
            advanced_system_prompt = """
            你是一个名为“配对小精灵”的AI助手，专注于星座、生肖、运势和配对相关问题。
            你的任务是：

            1. 根据用户的问题，调用适当的工具（如RAG工具或API），提供最准确的星座、生肖、运势和配对信息。

            2. 在回答时：
                - 保持语气温和、友好，并提供有用的建议。
                - 请分步骤说明复杂的查询结果，例如当提供星座配对时，先解释每个星座的特点，然后给出配对的综合评价。
                - 确保答案简洁、清晰，但必要时可以详细解释背景知识。

            3. 处理特殊情况：
                - 如果数据不可用或无法确定结果，请礼貌地告知用户原因，并建议用户提供更多信息（如具体出生时间）。
                - 如果用户输入的格式不正确，请友好地提醒并提供正确的输入格式。

            在每次回答中，请根据用户的问题选择合适的工具，并以用户可以理解的方式解释结果。
            """
            self.store[session_id].add_message(SystemMessage(content=advanced_system_prompt))
        return self.store[session_id]

    # 定义聊天函数，调用代理
    def chat(self, user_input, session_id):
        # 获取会话历史
        history = self.get_session_history(session_id).messages

        # 将历史消息转换为对话格式
        conversation = [msg for msg in history]
        conversation.append(HumanMessage(content=user_input))

        # 使用代理执行
        response = self.agent_executor({"input": user_input, "chat_history": conversation})

        # 将代理的响应添加到会话历史中
        self.store[session_id].add_message(HumanMessage(content=user_input))
        self.store[session_id].add_message(SystemMessage(content=response['output']))

        return response['output']

    # 定义 Gradio 接口
    def run(self):
        # 定义 Gradio 接口逻辑
        def reset_state():
            return [], str(uuid.uuid4())

        def user_message(user_input, history, state):
            session_id = state
            history = history + [(user_input, None)]
            return "", history, state

        def bot_response(history, state):
            session_id = state
            user_input = history[-1][0]
            output = self.chat(user_input, session_id)
            history[-1] = (user_input, output)
            return history, state

        with gr.Blocks() as demo:
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            state = gr.State(value=str(uuid.uuid4()))
            clear = gr.Button("清除对话")

            msg.submit(user_message, inputs=[msg, chatbot, state], outputs=[msg, chatbot, state], queue=False).then(
                bot_response, inputs=[chatbot, state], outputs=[chatbot, state], queue=False
            )
            clear.click(fn=reset_state, outputs=[chatbot, state], queue=False)

        demo.launch(share=True)  # 如果仍有 localhost 访问问题，可以设置 share=True

# 如果该脚本是主脚本，则创建Robot实例并调用run方法
if __name__ == '__main__':
    robot = Robot()
    robot.run()


