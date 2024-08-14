from typing import List

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ollama import Client

import uuid

import chromadb
from chromadb.config import Settings
import os

# Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"

# Index

urls_rus = [
    "https://neiro-psy.ru/blog/10-sovetov-dlya-poiska-lyubvi-i-znakomstv-pri-socialnoj-trevoge",
]

# Variables group for chroma collections & Ollama client
query_chroma_test_rus: str = "Что такое социальная тревога?"
collection_name: str = "rag-rus-ollama1"

ollama_url_home: str = "http://192.168.1.57:11434"
ollama_url_outdoor: str = "http://46.0.234.32:11434"
ollama_url_belgium: str = "http://46.183.187.205:11434"

chroma_client_host_home: str = "192.168.1.57"
chroma_client_host_outdoor: str = "46.0.234.32"
chroma_client_port: int = 8000

# Just for Ollama
emb_model = "llama3.1:8b-instruct-fp16"
# emb_model = "mxbai-embed-large"
# emb_model = "nextfire/paraphrase-multilingual-minilm"

# ll_model = "llama3.1:8b-instruct-fp16"
ll_model = "llama3.1:70b"
# ll_model = "llama3.1:70b-instruct-q6_K"


def txt_splitter(add_urls: List[str]) -> List[Document]:
    """Сплиттер для веб-документов"""
    docs = [WebBaseLoader(url).load() for url in add_urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=30
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print("Text splitting done.")
    return doc_splits


# ollama Embeddings

chroma_client = chromadb.HttpClient(host=chroma_client_host_home, port=chroma_client_port)
ollama_client = Client(host=ollama_url_home)

# STEP 0 Preparing the conditions
# Имя коллекции = target_name
target_name = collection_name
list_col = chroma_client.list_collections(10)
# Перебираем объекты в list_collections
found = False
for col in list_col:
    # Преобразуем объект в строку и находим значение name
    name_part = str(col).split(", name=")[1].rstrip(")")

    # Проверяем, совпадает ли name с искомым значением
    if name_part == target_name:
        found = True
        break  # Останавливаем цикл, если нашли совпадение

if found:
    print(f"Collection with name '{target_name}' exists, we'll NOT delete it")
    # chroma_client.delete_collection(collection_name)
else:
    print(f"Collection with name '{target_name}' does not exist, we'll create it.")

# STEP 1 :: store each document in a vector embedding database with Ollama
print(f"Создаем коллекцию (STEP 1): {collection_name}")

collection = chroma_client.create_collection(name=collection_name)
docs = txt_splitter(urls_rus)  # Подставляем правильную переменную в зависимости от источника данных.

i = 0
for doc in docs:
    i += 1
    print(f"Var response cycle # {i}")
    response = ollama_client.embeddings(model=emb_model, prompt=doc.page_content)
    embedding = response["embedding"]
    collection.add(
        ids=[str(uuid.uuid1())], embeddings=[embedding], metadatas=doc.metadata, documents=doc.page_content
    )

# an example prompt
prompt = query_chroma_test_rus

# STEP 2 :: generate an embedding for the prompt and retrieve the most relevant doc
print(f"Делаем запрос к коллекции (STEP 2): {collection_name}")
find_collection = chroma_client.get_collection(collection_name)
response = ollama_client.embeddings(
    prompt=prompt,
    model=emb_model
)
results = find_collection.query(
    query_embeddings=[response["embedding"]],
    n_results=2
)
# print(
#     "VALUES: ",
#     results.values(),
#     "ITEMS: ",
#     results.items(),
#     "KEYS: ",
#     results.keys(),
# )

data = results['documents'][0][0] + "\n\n\n" + results['documents'][0][1]

print(f"Ответ Ollama Embeddings {emb_model} на вопрос {prompt} (STEP 2):")
print(data)
print(" ###### ")

# STEP 3 :: generate a response combining the prompt and data we retrieved in step 2 (above)
print(f"Делаем запрос к языковой модели (STEP 3): {ll_model}")

first_prompt = (f"Ты - консультант по вопросам "
                f"психического здоровья психотерапевтической клиники Нейро-Пси. Твоя задача - отвечать на вопросы "
                f"человека на основании предоставленных тебе"
                f"документов из vectorstore retriever. Это "
                f"retrieved document: {data} Это вопрос пользователя: {prompt}")

first_cut_prompt = f"Это retrieved document: {data} Это вопрос пользователя: {prompt}"

system_prompt = ("Ты - ответственный консультант по вопросам психического здоровья психотерапевтической клиники "
                 "'Нейро-Пси' Твоя задача - отвечать на вопросы")
"человека на"
"основании предоставленных тебе документов."

output = ollama_client.generate(
    model=ll_model,
    prompt=first_prompt,
)

# STEP 3.1 :: generate a response trough ChatOllama

# llm = ChatOllama(model=ll_model, base_url=ollama_url_home, format="json", temperature=0)
# big_prompt = PromptTemplate(
#     template="""<|begin_of_text|> <|start_header_id|>system<|end_header_id|> Ты - консультант по вопросам
#     психического здоровья. Твоя задача - отвечать на вопросы человека на основании предоставленных тебе документов из
#     vectorstore retriever. <|eom_id|><|start_header_id|>user<|end_header_id|>
#     Here is the retrieved document:
#     {data}
#     Here is the user question:
#     {prompt}<|eom_id|>""",
#     input_variables=["data", "prompt"],
# )

print(" ****** ")
print(f"Ответ языковой модели {ll_model} на вопрос {prompt} (STEP 3) с использованием output['response']:")
print(output['response'])
print(" ****** ")
print(f"Ответ языковой модели {ll_model} на вопрос {prompt} (STEP 3.1) с использованием ollama_client.generate:")

# response = ollama_client.chat(
#     model=ll_model,
#
#     messages=[{'role': 'system', 'content': 'Ты - консультант по вопросам психического здоровья. Твоя задача - '
#                                             'отвечать на вопросы человека на основании предоставленных тебе '
#                                             'документов из vectorstore retriever.'},
#               {'role': 'user', 'content': prompt, 'data': data}],
#     keep_alive=-1,
#     stream=True,
# )
#
# print(response['message'])


gen = ollama_client.generate(
    model=ll_model,
    prompt=query_chroma_test_rus,
    system=system_prompt,
    template=first_cut_prompt,

)

print(gen['response'])

# Retrieval Grader

# llm = ChatOllama(model=ll_model, base_url=ollama_url_outdoor, format="json", temperature=0)
#
# prompt = PromptTemplate(
#     template="""
#     <|begin_of_text|>
#     <|start_header_id|>system<|end_header_id|>
#     You are a grader assessing relevance
#     of a retrieved document to a user question. If the document contains keywords related to the user question,
#     grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
#     Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
#     Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
#      <|eot_id|><|start_header_id|>user<|end_header_id|>
#     Here is the retrieved document: \n\n {document} \n\n
#     Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
#     """,
#     input_variables=["question", "document"],
# )
#
# retrieval_grader = prompt | llm | JsonOutputParser()
#
# question = query_chroma_test_rus
# docs = add_to_chromadb().invoke(question)
# doc_txt = docs[1].page_content
