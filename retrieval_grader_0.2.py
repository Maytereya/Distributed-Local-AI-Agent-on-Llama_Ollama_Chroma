# Ex index.py
# inherit FROM hybrid index v 2.0 copy because of bug in IDE.
# in memory rag changed onto chroma client rag
# Lets try to merge base retriever logic from lesson withs ollama and chroma clients...
# Done! It's working.

from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from bs4 import BeautifulSoup
from ollama import Client, embed
import ollama

import uuid

import chromadb
from chromadb.config import Settings
import os
from langchain_ollama import OllamaEmbeddings

# import ollama_embeddings

# Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"

# Index

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
]

urls_rus = [
    "https://neiro-psy.ru/blog/10-sovetov-dlya-poiska-lyubvi-i-znakomstv-pri-socialnoj-trevoge",
]

urls_reserve = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Variables group for chroma collections management

query_chroma_test: str = "What is the Adversarial attacks?"
query_chroma_test_rus: str = "Что такое социальная тревога?"
#
collection_name: str = "rag-ollama"
#
chat_ollama_url_home: str = "http://192.168.1.57:11434"
chat_ollama_url_outdoor: str = "http://46.0.234.32:11434"
chat_ollama_url_belgium: str = "http://46.183.187.205:11434"

chroma_client_host_home: str = "192.168.1.57"
chroma_client_host_outdoor: str = "46.0.234.32"
chroma_client_port: int = 8000

chroma_client = chromadb.HttpClient(host=chroma_client_host_outdoor, port=chroma_client_port)

# Variables group for ollama client management

ollama_client = Client(host=chat_ollama_url_outdoor)
emb_model = "llama3.1:8b-instruct-fp16"
ll_model = "llama3.1:8b-instruct-fp16"

# Start of Ollama Embeddings
question = "What is agent memory?"


# STEP 0 :: Preparing the conditions
def preconditioning(target_name: str):
    """Подготавливаем условия для создания и использования коллекций. В данном случае, удаляем коллекцию,
    если она уже существует"""
    # Имя коллекции = target_name
    list_col = chroma_client.list_collections(10)  # 10 - число выведенных коллекций
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
        print(f"Collection with name '{target_name}' exists, we'll delete it")
        chroma_client.delete_collection(collection_name)
    else:
        print(f"Collection with name '{target_name}' does not exist, we'll create it on the next step.")


# STEP 1 :: store each document in a vector embedding database with Ollama

def txt_splitter(add_urls: List[str]) -> List[Document]:
    """Split the web documents into list of Document"""
    docs = [WebBaseLoader(url).load() for url in add_urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print("Text Splitting Done.")
    return doc_splits


def ollama_create_collection(coll_name: str):
    """Создание коллекции с заданным именем"""
    ollama_collection = chroma_client.create_collection(name=coll_name)
    print(f"Создаем коллекцию (STEP 1): {coll_name}")
    return ollama_collection


def ollama_add_data_to_collection(coll_name: str, web_urls: List[str], embedding_model: str):
    """Добавляем встраивания в коллекцию"""

    # Передаем результаты работы предварительных процедур:
    docs = txt_splitter(web_urls)  # Подставляем правильную переменную в зависимости от источника данных.
    collection = chroma_client.get_collection(coll_name)

    print(f"Добавляем данные в коллекцию (STEP 1.1): {coll_name}")

    length = len(docs)
    print(f"Количество чанков для обработки: {length}")

    for doc in docs:
        length -= 1
        print(f"Var response cycle remain: {length}")
        response = ollama_client.embeddings(model=embedding_model, prompt=doc.page_content)
        embedding = response["embedding"]
        # print(str(embedding))
        collection.add(
            ids=[str(uuid.uuid1())], embeddings=[embedding], metadatas=doc.metadata, documents=doc.page_content
        )


# STEP 2 :: generate an embedding for the prompt and retrieve the most relevant doc

def ollama_query_to_collection(col_name: str, prompt: str, embedding_model: str) -> str:
    print(f"Делаем запрос к коллекции (STEP 2): {col_name}")
    collection = chroma_client.get_collection(col_name)
    response = ollama_client.embeddings(
        prompt=prompt,
        model=embedding_model
    )
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=1
    )

    data = results['documents'][0][0]
    # + "\n\n\n" + results['documents'][0][1])

    print(f"Ответ Ollama Embeddings {emb_model} на вопрос {question} (STEP 2):")
    print(data)
    print(" ###### ")
    return data


# Retrieval Grader
# 1, 2, 3 functions call
preconditioning(collection_name) # 1
ollama_create_collection(collection_name) # 2
ollama_add_data_to_collection(collection_name, urls, emb_model) # 3

llm = ChatOllama(model=ll_model, base_url=chat_ollama_url_outdoor, format="json", temperature=0)

prpt = PromptTemplate(
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

retrieval_grader = prpt | llm | JsonOutputParser()

doc_txt = ollama_query_to_collection(collection_name, question, emb_model)

print(f"Квалификация вопроса: {question}")

print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

# Должен сообщить: {'score': 'yes'}
