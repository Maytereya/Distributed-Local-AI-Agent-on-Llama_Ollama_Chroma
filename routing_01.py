# Router

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# Input variables:
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


def route(question: str):
    """"Routing Questions"""
    llm = ChatOllama(model=ll_model, base_url=ollama_url, format="json", temperature=0)

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
        prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
        or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
        no preamble or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    question_router = prompt | llm | JsonOutputParser()
    return question_router.invoke({"question": question})

# question = "llm agent memory"
# # docs = index.retriever.get_relevant_documents(question)
# docs = index.retriever.invoke(question)
# doc_txt = docs[1].page_content
# print(question_router.invoke({"question": question}))
