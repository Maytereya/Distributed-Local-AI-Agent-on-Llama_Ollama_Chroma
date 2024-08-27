import asyncio
from langchain_community.llms.ollama import Ollama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
import config as c  # Here are all ip, llm names and other important things


async def generate_answer(question: str, documents) -> str:
    """
    Самая медленная процедура!!!

    Generate the answer if the agent

    """
    prompt = PromptTemplate(
        template=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question in string format. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        \n ------- \n
        Question: {question} 
        \n ------- \n
        Context: {documents} 
        \n ------- \n
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "documents"],
    )

    rag_chain = prompt | c.llm | StrOutputParser()
    return await rag_chain.ainvoke({"documents": documents, "question": question})


# Post-processing
def format_docs(docs):
    """Convert Document to string
        This option need for...
    """
    return "\n\n".join(doc.page_content for doc in docs)


async def grade(question: str, document: str):
    """
    Медленная процедура!!!

    Grade the relevance of the retrieved document.

    """

    system_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
                        of a retrieved document to a user question. If the document contains keywords related to the user question, 
                        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
                        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
                         <|eot_id|><|start_header_id|>user<|end_header_id|>
                        Here is the retrieved document: \n\n {document} \n\n
                        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                        """,
        input_variables=["question", "document"],
    )
    g = system_prompt | c.llm | JsonOutputParser()
    result = await g.ainvoke(
        {"question": question, "document": document}
    )
    return result


async def hallucinations_checker(documents, generation: str):
    """Hallucination Grader"""

    prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: 
        \n ------- \n
        {generation}  
        \n ------- \n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | c.llm | JsonOutputParser()
    return await hallucination_grader.ainvoke({"documents": documents, "generation": generation})


async def answer_grader(question: str, generation: str):
    """Answer Grader"""

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: 
        \n ------- \n
        {question} 
        \n ------- \n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )

    grade_answer = prompt | c.llm | JsonOutputParser()
    return await grade_answer.ainvoke({"question": question, "generation": generation})
