# Код lang_graph составлен как класс с учетом рекомендаций DeepLearning.ai (Harris)
# v 2.0
import asyncio

from retrieval_grader_03 import RetrievalGrader
from pprint import pprint
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END
from typing import TypedDict, Annotated, Optional, List
# import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
# from typing import List  # используется для типизации списков.
from langchain_core.documents import Document  # представляет документ.

import generate_01
import routing_01
import search_01

_ = load_dotenv()

tool = TavilySearchResults(max_results=2)  # increased number of results

# Input variables:
collect_name: str = "rag-ollama"
#
ollama_url_out: str = "http://46.0.234.32:11434"
ollama_url_in: str = "http://192.168.1.57:11434"
chat_ollama_url_belgium: str = "http://46.183.187.205:11434"
ollama_url = ollama_url_out  # Итоговое значение
#
chroma_host_in: str = "192.168.1.57"
chroma_host_out: str = "46.0.234.32"
chroma_host = chroma_host_out  # Итоговое значение
chroma_port = 8000
#
emb_model = "llama3.1:8b-instruct-fp16"
ll_model = "llama3.1:8b-instruct-fp16"


# question1 = "What is agent memory?"
# local_llm = "llama3.1:8b-instruct-fp16"


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
        rg = RetrievalGrader(embedding_model=emb_model, ollama_url=ollama_url_in, chroma_host=chroma_host_in,
                             chroma_port=chroma_port, llm=ll_model, question=state["question"],
                             collection_name=collect_name)
        documents = asyncio.run(rg.retrieve_document())
        # ToDo: убрать question из конструктора в метод(-ы)

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
        generation = generate_01.generate_first_answer(documents, question)
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(self, state: AgentState):
        """
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
        rg = RetrievalGrader(embedding_model=emb_model, ollama_url=ollama_url_in, chroma_host=chroma_host_in,
                             chroma_port=chroma_port, llm=ll_model, question=state["question"],
                             collection_name=collect_name)

        # ToDo: убрать question из конструктора в метод(-ы)
        filtered_docs = []
        web_search = "No"
        for d in documents:
            # score = index.retrieval_grader.invoke(
            #     {"question": question, "document": d.page_content}
            # )
            score = rg.grade().invoke(
                {"question": question, "document": d.page_content}
            )

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
        docs = search_01.web_search_tool.invoke({"query": question})
        combined_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=combined_results)
        if documents is not None:
            documents.append(web_results)  # Expected type 'str' (matched generic type '_T'), got 'Document' instead

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
        source = routing_01.route(question)
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
        print(state["question"])
        web_search = state["web_search"]
        print(state["documents"])

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
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

        # score = main_generate.hallucination_grader.invoke(
        #     {"documents": documents, "generation": generation}
        # )

        score = generate_01.check_hallucinations(documents, generation)
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            # score = main_generate.answer_grader.invoke({"question": question, "generation": generation})
            score = generate_01.answer_grader(question, generation)
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


# Note, the query was modified to produce more consistent results.
# Results may vary per run and over time as search information and models change.
# На базе OPEN AI:
# query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
# What is the GDP of that state? Answer each question."
# messages = [HumanMessage(content=query)]
#
# model = ChatOpenAI(model="gpt-4o")  # requires more advanced model
# abot = Agent(model, [tool], system="prompt")
# result = abot.graph.invoke({"messages": messages})
#
# print(result['messages'][-1].content)

# На базе LLAMA3:

# Compile
def compilation(question: str):
    """Compile and run agent answer, based on the question"""
    agent = Agent()
    app = agent.graph
    inputs = {"question": question}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
            pprint(value["generation"])
