# Код lang_graph составлен как класс с учетом рекомендаций DeepLearning.ai (Harris)
# v 2.0
import asyncio
import json
import textwrap

from pprint import pprint
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END
from typing import TypedDict, Optional, List
from langchain_core.documents import Document  # представляет документ.

from retrieve import QueryCollection
import generate
import routing
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


class Agent:

    def __init__(self, system=""):
        self.system = system
        graph = StateGraph(AgentState)

        graph.add_node("websearch", self.web_search)  # web search
        graph.add_node("retrieve", self.retrieve)  # retrieve
        graph.add_node("grade_documents", self.grade_documents)  # grade documents
        graph.add_node("generate", self.generate)  # generate

        graph.add_conditional_edges(
            START,
            self.route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )

        graph.add_edge("retrieve", "grade_documents")

        graph.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        graph.add_edge("websearch", "generate")

        graph.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
            },
        )

        self.graph = graph.compile()  #

    # Functions, that are called/invoked:

    def retrieve(self, state: AgentState):
        """
        Retrieve documents from Chroma VS
        Эта функция получает документы из векторного хранилища на основе вопроса.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """

        qc = QueryCollection(ollama_url=c.ollama_url, chroma_host=c.chroma_host, chroma_port=c.chroma_port,
                             embedding_model=c.emb_model, )

        documents = asyncio.run(qc.async_launcher(question=state["question"], collection_name=c.collect_name))

        print("---RETRIEVE FROM CHROMA---")
        question = state["question"]

        return {"documents": documents, "question": question}

    def generate(self, state: AgentState):
        """
        Generate answer using RAG on retrieved documents
        Эта функция генерирует ответ, используя ранее полученные документы.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE answer using RAG or WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]
        documents = generate_01.format_docs(documents)
        # RAG generation
        generation = asyncio.run(generate_01.generate_answer(documents, question))
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(self, state: AgentState):
        """
        Самая медленная процедура!!!

        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Эта функция проверяет релевантность документов вопросу и
        устанавливает флаг для веб-поиска, если документы нерелевантны.

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

                score = asyncio.run(generate_01.grade(question, d.page_content))

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

    def web_search(self, state: AgentState):
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
        # ToDo: try to make async!
        docs = search_01.web_search(question)
        combined_results = "\n".join(
            [d["content"] for d in docs])  # docs — это список словарей, где каждый словарь имеет ключ "content"
        # combined_results = docs # Упростим задачу
        web_results = Document(page_content=combined_results)
        if documents is not None:
            documents.append(web_results)

        else:
            documents = [web_results]
        return {"documents": documents, "question": question}

    # Условные переходы

    def route_question(self, state: AgentState):
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
        # source = routing_01.question_router.invoke({"question": question})
        source = asyncio.run(routing_01.route(question))
        print(source)
        print(source["datasource"])
        if source["datasource"] == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source["datasource"] == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"

    def decide_to_generate(self, state: AgentState):
        """
        Эта функция определяет, нужно ли выполнять веб-поиск или можно генерировать ответ.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        # print(state["question"])
        web_search = state["web_search"]
        # print(state["documents"])

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

    # Conditional edge

    def grade_generation_v_documents_and_question(self, state: AgentState):
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

        score = asyncio.run(generate_01.hallucinations_checker(documents, generation))
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")

            score = asyncio.run(generate_01.answer_grader(question, generation))
            grade = score["score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---ОПЯТЬ НИЧЕГО НЕ НАШЕЛ, RE-TRY---")
            return "not supported"


# Compile
def compilation(question: str):
    """Compile and run agent answer, based on the question"""
    agent = Agent()
    app = agent.graph
    inputs = {"question": question}
    for output in app.stream(inputs):
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
