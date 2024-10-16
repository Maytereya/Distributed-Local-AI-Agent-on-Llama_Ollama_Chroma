import asyncio

from Deprecated import async_graph_operator
import config as c

# TODO 1. Добавить функциональность к строке ввода (чтобы ответ системы инициализировался клавишей ввод) 2. Понять
#  почему так долго происходит обращение к ollama и устранить причину либо иначе оформить доступ к vectorstore &
#  retriever. 3. Устранить ошибку вывода - убрать лишние знаки /n после ответа системы (причины: проверить
#  соответствие типов Document -> Str, а так же лишнюю подстановку или пустой вывод (что-то может не работать) ).


# ToDo 2. Check connection to services module before start the agent.


if __name__ == "__main__":
    print(f"ollama_url: {c.ollama_url}")
    print(f"ollama_host: {c.chroma_host}")
    print(f"model: {c.ll_model}")

    question = c.question1

    result = asyncio.run(
        async_graph_operator.compilation(question))  # Запускаем асинхронную функцию через asyncio.run()

    print(result)
