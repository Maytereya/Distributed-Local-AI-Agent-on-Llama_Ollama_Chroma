import asyncio
from xml.dom.minidom import Document

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

# import index
from retrieval_grader_03 import RetrievalGrader

collect_name: str = "rag-ollama"
ollama_url_out: str = "http://46.0.234.32:11434"
ollama_url_in: str = "http://192.168.1.57:11434"
chat_ollama_url_belgium: str = "http://46.183.187.205:11434"
ollama_url = ollama_url_out # Итоговое значение

chroma_host_in: str = "192.168.1.57"
chroma_host_out: str = "46.0.234.32"
chroma_host = chroma_host_out # Итоговое значение

chroma_port = 8000

emb_model = "llama3.1:8b-instruct-fp16"
ll_model = "llama3.1:8b-instruct-fp16"
question1 = "What is agent memory?"
local_llm = "llama3.1:8b-instruct-fp16"


# question = "agent memory"

# Generate
def generate_first_answer(question: str, documents: [Document]) -> [Document]:

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )

    llm = ChatOllama(model=ll_model, base_url=ollama_url, format="json", temperature=0)
    rag_chain = prompt | llm | StrOutputParser()


    # documents = "\n\n".join(doc.page_content for doc in documents)
    generation = rag_chain.invoke({"context": documents, "question": question})
    # print(generation)
    return generation


# Post-processing
def format_docs(docs):
    """Convert Document to string
        This option need for...
    """
    return "\n\n".join(doc.page_content for doc in docs)


def check_hallucinations(documents: [Document], generation: str):
    """Hallucination Grader"""
    llm = ChatOllama(model=ll_model, base_url=ollama_url, format="json", temperature=0)
    prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    return hallucination_grader.invoke({"documents": documents, "generation": generation})


def answer_grader(question: str, generation: str):
    """Answer Grader"""
    llm = ChatOllama(model=ll_model, base_url=ollama_url, format="json", temperature=0)

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )

    grade_answer = prompt | llm | JsonOutputParser()
    return grade_answer.invoke({"question": question, "generation": generation})
