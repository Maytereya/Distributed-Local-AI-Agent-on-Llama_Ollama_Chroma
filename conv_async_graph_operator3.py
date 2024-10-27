# Код lang_graph составлен как класс с учетом рекомендаций DeepLearning.ai (Harris)
# v 6.0 "long logic"
# Теперь это чат с логикой агента, расширенной на чат. Есть память в виде переменной history[]
# Но память langGraph в этой реализации не используется.
# Существенное изменение - добавление нативного ретривера поверх векторной базы данных Chroma DB
# ollama embeddings в данной конфигурации не используется.
# Пытаюсь удлинить логику работы узлов RAG с помощью i+


import asyncio
import copy
from pprint import pprint
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END
from typing import TypedDict, Optional, List
from langchain_core.documents import Document  # представляет документ.
import warnings

# R Project modules:
import generate2 as generate
import check
import routing2 as route
import aretrieve3 as retrieve
import search

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)

_ = load_dotenv()

import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"


# ToDo: Проверить соответствие типов, чтобы не было List[str], List[Document] и Document там, где все д.б. List[Document]

class AgentState(TypedDict):
    question: str
    generation: str
    web_search: str
    collection_name_1: str  # Пробуем в связи с необходимостью использовать разные модели и разные способы добавления данных (рус/eng)
    collection_name_2: str
    # collection_name_3: str # Резерв
    attempt_count: int  # счетчик попыток обращения к RAG до выхода на Web Search
    history: list
    documents: Optional[List[Document]]  # С поправкой на ошибку несоответствия типа в модуле web_search


# Входная функция !
async def route_question(state: AgentState):
    """
    Эта функция определяет, куда направить вопрос: на веб-поиск,
    векторное хранилище, в чат с памятью или окончание сессии (выход).

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print("Вопрос от пользователя: ", question)

    source = await route.route(question)

    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"

    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG CHAIN---")
        return "vectorstore"

    elif source["datasource"] == "chat":
        print("---ROUTE QUESTION TO CHAT WITH MEMORY---")
        return "chat"

    elif source["datasource"] == "exit":
        print("---ROUTE TO TERMINATE THE SESSION---")
        return "exit"


async def retrieve_vs_1(state: AgentState):
    """
    Retrieve documents from Chroma Vector Store
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """


    attempt_count = 1  # Инициализируем счетчик попыток обращения к RAG

    # state["collection_name"] = "txt-side-eff-cosine-distiluse-base-multilingual-cased-v1"

    print("Collection name: ", state["collection_name_1"])

    documents = retrieve.vs_query(existed_collection=state["collection_name_1"], question=state["question"],
                                  search_type="mmr", k=5)
    print("---RETRIEVE FROM CHROMA Vector Store---")
    question = state["question"]
    collection_name = state["collection_name_1"]
    print("Вопрос: ", state["question"])
    print("Collection name: ", collection_name)
    for document in documents:
        print(document.page_content[:100])

    return {"documents": documents, "question": question, "attempt_count": attempt_count}


async def retrieve_db_2(state: AgentState):
    """
    Retrieve documents from Chroma Database
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    # TODO: existed_collection=c.collect_name - исправить на правильное название коллекции!

    documents = retrieve.query_collection(existed_collection=state["collection_name_1"], question=state["question"],
                                          n_results=1)
    print("---RETRIEVE FROM CHROMA DB---")
    question = state["question"]
    attempt_count = 2
    print("Вопрос: ", state["question"])
    for document in documents:
        print(document.page_content)

    return {"documents": documents, "question": question, "attempt_count": attempt_count}


async def retrieve_db_3(state: AgentState):
    """
    Retrieve documents from Chroma Database using ____  model, chose for reserve attempt search
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    # TODO: existed_collection=c.collect_name - исправить на правильное название коллекции!

    documents = retrieve.query_collection(existed_collection=state["collection_name_3"], question=state["question"],
                                          model="distiluse", n_results=1)
    print("---RETRIEVE FROM CHROMA DB---")
    question = state["question"]
    attempt_count = 3
    print("Вопрос: ", state["question"])
    for document in documents:
        print(document.page_content)

    return {"documents": documents, "question": question, "attempt_count": attempt_count}


# not async converted
def web_search(state: AgentState):
    """
    Эта функция выполняет веб-поиск на основе вопроса и добавляет результаты к документам.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---TAVILY WEB SEARCH---")

    question = state["question"]
    documents = state["documents"]

    docs = search.web_search(question)

    web_results = Document(page_content=docs)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    print("Вопрос: ", state["question"])
    print("Ответ: ", docs)

    return {"documents": documents, "question": question}


# !
async def grade_documents(state: AgentState):
    """
    Determines whether the retrieved documents are relevant to the question.
    If any document is not relevant, we will set a flag to run web search.

    Args:
        state (dict): The current graph state.

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state.
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "yes"  # Устанавливаем значение "yes" по умолчанию

    if documents is not None:
        for d in documents:
            score = await check.grade(question=question, document=d.page_content)
            grade = score["score"]

            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
                # Как только найден релевантный документ, устанавливаем web_search в "no"
                web_search = "no"
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")

    else:
        print("Document was not given, maybe because of connection error")

    return {"documents": filtered_docs, "question": question, "web_search": web_search}


# ! sync
def decide_to_generate(state: AgentState):
    """
    Определяет какое векторное хранилище будет на следующей итерации до тех пор, пока не будет достигнуто условие
    наибольшего числа итераций.
    Например, векторных хранилища/коллекций в одном хранилище = 3. Итерационный индекс в таком случае = 2 (начиная с 0).
    При достижении максимального значения индекса, поиск переходит к WebSearch, что означает, что в векторных хранилищах
    релевантной информации не найдено.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    # ToDo: проверить логику работы

    print("---ASSESS GRADED DOCUMENTS---")

    web_search = state["web_search"]
    attempt_counter = state["attempt_count"]
    print("########################")
    print("decide_to_generate attempt_counter: ", attempt_counter)
    print("########################")

    if web_search == "yes" and attempt_counter == 1:  # Пока что захардкодим эту величину

        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            f"---DOCUMENTS ARE NOT RELEVANT TO QUESTION, "
            f"STARTING SEARCH NEXT COLLECTION. "
            f"Attempts: {attempt_counter}---"
        )
        return "retrieve_next_1"

    elif web_search == "yes" and attempt_counter == 2:  # Пока что захардкодим эту величину

        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            f"---DOCUMENTS ARE NOT RELEVANT TO QUESTION, "
            f"STARTING SEARCH NEXT COLLECTION. "
            f"Attempts: {attempt_counter}---"
        )
        # state["collection_name_3"] = "txt-side-eff-cosine-sbert_large_nlu_ru"
        return "retrieve_next_2"

    elif web_search == "yes" and attempt_counter == 3:

        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            f"---DOCUMENTS ARE NOT RELEVANT TO QUESTION, "
            f"STARTING WEB SEARCH. "
            f"Attempts: {attempt_counter}---"
        )
        return "web_search"

    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


# !
async def generate_final(state: AgentState):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE answer using RAG or WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    generation: list[Document] = []
    history = state["history"]

    # Потенциальное избавление от циклических ссылок
    history_copy = copy.deepcopy(history)

    try:
        # generation based on RAG or WEB_Search tool
        generation = await generate.generate_answer(question, documents, history_copy)
    except Exception as e:
        print("Ошибка получения данных: ", e)

    return {"documents": documents, "question": question, "generation": generation, "history": history}


async def chat(state: AgentState):
    """
    Generate chat

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---CHAT WITH U---")
    question = state["question"]
    history = state["history"]

    # Потенциальное избавление от циклических ссылок
    history_copy = copy.deepcopy(history)

    # chat generation
    generation = await generate.chat(question, history_copy)
    return {"question": question, "generation": generation, "history": history}


# !
async def grade_generation_v_documents_and_question(state: AgentState):
    """
    Эта функция проверяет, основан ли ответ на документах и отвечает ли он на вопрос.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    # history = state["history"]
    grade: str = "no"
    try:
        score = await check.hallucinations_checker(documents, generation)
        grade = score["score"]

        print("check.hallucinations_checker, score: ", score)
        print("check.hallucinations_checker, grade: ", grade)
    except Exception as e:
        print("Ошибка получения данных от check.hallucinations_checker: ", e)

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        try:
            score = await check.answer_grader(question, generation)
            grade = score["score"]
            print("check.answer_grader, score: ", score)
            print("check.answer_grader, grade: ", grade)

        except Exception as e:
            print("Ошибка получения данных от check.answer_grader: ", e)

        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---Nothing found, RE-TRY---")
        return "not supported"


class Agent:

    def __init__(self, system=""):
        self.system = system
        graph = StateGraph(AgentState)

        graph.add_node("websearch", web_search)  # web search
        graph.add_node("retrieve_vs_1", retrieve_vs_1)  # retrieve from Chroma vector store mmr only
        graph.add_node("retrieve_db_2", retrieve_db_2)  # retrieve from Chroma DB vector search
        graph.add_node("retrieve_db_3", retrieve_db_3)  # retrieve from Chroma DB vector search with sbert model
        graph.add_node("grade_documents", grade_documents)  # grade documents
        graph.add_node("generate", generate_final)  # generate
        graph.add_node("chat", chat)  # chat

        graph.add_conditional_edges(
            START,
            route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve_vs_1",
                "chat": "chat",
                "exit": END,
            },
        )

        graph.add_edge("retrieve_vs_1", "grade_documents")

        # добавим еще одну ветку, связанную с последовательным перебором коллекций перед переходом на
        # websearch
        graph.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
                "retrieve_next_1": "retrieve_db_2",
                "retrieve_next_2": "retrieve_db_3",
            },
        )
        graph.add_edge("websearch", "generate")

        # усложним логику дополнительным ветвлением для поиска разными способами с двумя моделями.
        graph.add_edge("retrieve_db_2", "grade_documents")
        graph.add_edge("retrieve_db_3", "grade_documents")

        graph.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
            },
        )
        graph.add_edge("chat", END)

        self.graph = graph.compile()  #


async def agent_conversation(agent: Agent):
    """Функция для взаимодействия с пользователем в виде чата."""

    # Инициализация пустой истории
    history = []
    # Первое название коллекции, которое следует передать в graph через окно загрузки
    name_of_collection_1: str = "25_10_2024_LaBSE-en-ru_pdf"
    name_of_collection_2: str = "23_10_2024_distiluse_txt"
    # attempt_counter: int = 0

    print("Начнем беседу с агентом. Введите ваш вопрос.")
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit", "e", "q", "выход", "в"]:
            print("Goodbye!")
            break
        # Добавляем новый вопрос в историю
        history.append({"role": "user", "content": question})

        # Для каждого нового вопроса создаем состояние агента
        inputs = {"question": question, "history": history, "collection_name_1": name_of_collection_1,
                  "collection_name_2": name_of_collection_2,
                  }

        # Запускаем цикл агента для каждого вопроса
        result = await run_agent(agent, inputs)

        # Добавляем ответ агента в историю
        history.append({"role": "assistant", "content": result})

        print(f"Agent: {result}")


async def run_agent(agent: Agent, inputs):
    """Запуск агента для обработки вопроса и получения результата."""
    app = agent.graph

    # Асинхронная обработка графа
    async for output in app.astream(inputs):
        for key, value in output.items():
            pprint(f"Finished running node: {key}")
    # Возвращаем сгенерированный ответ
    return value["generation"]


# Основная функция для запуска чат-бота
async def main():
    # Создаем агента
    agent = Agent()

    # Запускаем процесс взаимодействия с агентом
    await agent_conversation(agent)


# Запуск программы
if __name__ == "__main__":
    asyncio.run(main())
