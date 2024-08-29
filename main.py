import asyncio
# import yappi

import async_graph_operator
import config as c

# TODO 1. Добавить функциональность к строке ввода (чтобы ответ системы инициализировался клавишей ввод) 2. Понять
#  почему так долго происходит обращение к ollama и устранить причину либо иначе оформить доступ к vectorstore &
#  retriever. 3. Устранить ошибку вывода - убрать лишние знаки /n после ответа системы (причины: проверить
#  соответствие типов Document -> Str, а так же лишнюю подстановку или пустой вывод (что-то может не работать) ).


# ToDo 2. Check connection to services module before start the agent.


if __name__ == "__main__":
    question = c.question1
    # yappi.start()
    result = asyncio.run(
        async_graph_operator.compilation(question))  # Запускаем асинхронную функцию через asyncio.run()
    # yappi.stop()
    print(result)

    # print("=== YAPPI ===")
    # print(yappi.get_func_stats().print_all())
