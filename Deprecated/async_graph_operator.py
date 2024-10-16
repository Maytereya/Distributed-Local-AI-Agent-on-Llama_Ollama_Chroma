# Код lang_graph составлен как класс с учетом рекомендаций DeepLearning.ai (Harris)
# v 3.0
import json
import textwrap
from pprint import pprint
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END
from typing import TypedDict, Optional, List
from langchain_core.documents import Document  # представляет документ.

from Deprecated.aretrieve import QueryCollection
from Deprecated import agenerate
import arouting
import search
import config as c  # Here are all ip, llm names and other important things

_ = load_dotenv()


# ToDo: Проверить соответствие типов, чтобы не было List[str], List[Document] и Document там, где все д.б. List[Document]

class AgentState(TypedDict):
    # messages: Annotated[list[AnyMessage], operator.add]
    question: str
    generation: str
    web_search: str
    # documents: List[str]  #Возможно ошибочно
    documents: Optional[List[Document]]  # С поправкой на ошибку несоответствия типа в модуле web_search


# Входная функция !
async def route_question(state: AgentState):
    """
    Эта функция определяет, куда направить вопрос: на веб-поиск или векторное хранилище.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)

    source = await arouting.route(question)

    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


# !
async def retrieve(state: AgentState):
    """
    Retrieve documents from ChromaDB
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    qc = QueryCollection(ollama_url=c.ollama_url, chroma_host=c.chroma_host, chroma_port=c.chroma_port,
                         embedding_model=c.emb_model, )

    documents = await (qc.async_launcher(question=state["question"], collection_name=c.collect_name))
    print("---RETRIEVE FROM CHROMA DB---")
    question = state["question"]

    return {"documents": documents, "question": question}


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

    # Web search

    docs = search.web_search(question)
    # print("Тип документа поиска: ")
    # print(type(docs))
    # print(docs)
    # combined_results = "\n".join(
    #     [d["content"] for d in docs]
    # )  # docs — это список словарей, где каждый словарь имеет ключ "content"

    # combined_results = docs # Упростим задачу
    web_results = Document(page_content=docs)
    if documents is not None:
        documents.append(web_results)

    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


# !
async def grade_documents(state: AgentState):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc

    filtered_docs = []
    web_search = "No"
    if documents is not None:
        for d in documents:

            score = await agenerate.grade(question, d.page_content)

            grade = score["score"]
            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT, ALL IS OK---")
                filtered_docs.append(d)
            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                # We do not include the document in filtered_docs
                # We set a flag to indicate that we want to run web search
                web_search = "Yes"
                continue
    else:
        print("Document was not given, maybe because of connection error")
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


# ! sync
def decide_to_generate(state: AgentState):
    """
    Эта функция определяет, нужно ли выполнять веб-поиск или можно генерировать ответ.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")

    web_search = state["web_search"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, START WEB SEARCH---"
        )
        return "websearch"
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
    documents = agenerate.format_docs(documents)
    # RAG generation
    generation = await agenerate.generate_answer(documents, question)
    return {"documents": documents, "question": question, "generation": generation}


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

    score = await agenerate.hallucinations_checker(documents, generation)
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")

        score = await agenerate.answer_grader(question, generation)
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---Nothing found again, RE-TRY---")
        return "not supported"


class Agent:

    def __init__(self, system=""):
        self.system = system
        graph = StateGraph(AgentState)

        graph.add_node("websearch", web_search)  # web search
        graph.add_node("retrieve", retrieve)  # retrieve
        graph.add_node("grade_documents", grade_documents)  # grade documents
        graph.add_node("generate", generate_final)  # generate

        graph.add_conditional_edges(
            START,
            route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )

        graph.add_edge("retrieve", "grade_documents")

        graph.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        graph.add_edge("websearch", "generate")

        graph.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
            },
        )

        self.graph = graph.compile()  #


# Compile
async def compilation(question: str):
    """Compile and run agent answer, based on the question"""
    value = None
    agent = Agent()
    app = agent.graph
    inputs = {"question": question}
    async for output in app.astream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    return value["generation"]


def pretty_print_generation(generation_str):
    try:
        # Заменяем одинарные кавычки на двойные
        generation_str = generation_str.replace("'", '"').strip()

        # Парсим строку как JSON
        generation_dict = json.loads(generation_str)

        # Форматируем JSON строку с отступами
        formatted_json = json.dumps(generation_dict, indent=4)

        # Используем textwrap для переноса длинных строк
        wrapped_json = textwrap.fill(formatted_json, width=80, replace_whitespace=False)

        print(wrapped_json)

    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        print("Original string:")
        print(generation_str)
