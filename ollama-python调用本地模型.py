import ollama


# 流式输出
def api_generate(text: str):
    print(f'提问：{text}')

    stream = ollama.generate(
        stream=True,
        model='llama3.1:8b',  # 修改大模型名称1
        prompt=text,
    )

    print('-----------------------------------------')
    for chunk in stream:
        if not chunk['done']:
            print(chunk['response'], end='', flush=True)
        else:
            print('\n')
            print('-----------------------------------------')
            print(f'总耗时：{chunk["total_duration"]}')
            print('-----------------------------------------')


if __name__ == '__main__':
    # 流式输出
    api_generate(text='天空为什么是蓝色的？')

    # 非流式输出
    content = ollama.generate(model='llama3.1:8b', prompt='天空为什么是蓝色的？')  # 修改大模型名称2
    print(content)
