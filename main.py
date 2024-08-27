import graph_operator_01

# TODO 1. Добавить функциональность к строке ввода (чтобы ответ системы инициализировался клавишей ввод) 2. Понять
#  почему так долго происходит обращение к ollama и устранить причину либо иначе оформить доступ к vectorstore &
#  retriever. 3. Устранить ошибку вывода - убрать лишние знаки /n после ответа системы (причины: проверить
#  соответствие типов Document -> Str, а так же лишнюю подстановку или пустой вывод (что-то может не работать) ).


# ToDo 2. Check connection to services module first that start the agent.

q = input("Ваш вопрос: ")
print("=== AGENT ANSWER ===")
graph_operator_01.pretty_print_generation(graph_operator_01.compilation(q))
