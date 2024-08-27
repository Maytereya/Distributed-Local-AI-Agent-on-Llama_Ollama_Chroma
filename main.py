# This is a sample Python script.
from pprint import pprint

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# import hybrid
import graph_operator_01

# TODO 1. Добавить функциональность к строке ввода (чтобы ответ системы инициализировался клавишей ввод) 2. Понять
#  почему так долго происходит обращение к ollama и устранить причину либо иначе оформить доступ к vectorstore &
#  retriever. 3. Устранить ошибку вывода - убрать лишние знаки /n после ответа системы (причины: проверить
#  соответствие типов Document -> Str, а так же лишнюю подстановку или пустой вывод (что-то может не работать) ).

q = input("Ваш вопрос: ")
print("=== AGENT ANSWER ===")
graph_operator_01.pretty_print_generation(graph_operator_01.compilation(q))

# тестирование Retrieval Grader (ceйчас, через ollama agent он крайне медленный)
# Пример использования:А
# print(f"this is the {__name__} module")
# rg = Retrieving(embedding_model=emb_model, ollama_url=ollama_url_in, chroma_host=chroma_host_in,
#                      chroma_port=chroma_port, llm=ll_model, question=question1,
#                      collection_name=collect_name)
#
# result = asyncio.run(rg.process())
# print(result)

# Должен сообщить: {'score': 'yes'}

# Тестирование Routing
# question = "llm agent memory"
# # docs = index.retriever.get_relevant_documents(question)
# docs = index.retriever.invoke(question)
# doc_txt = docs[1].page_content
# print(question_router.invoke({"question": question}))
