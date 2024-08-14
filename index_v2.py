# straight index v 2.0
# in memory rag changed onto chroma client rag

from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_nomic.embeddings import NomicEmbeddings
from chromadb.utils import embedding_functions  # Пытаемся установить all-MiniLM-L6-v2, который по умолчанию в Chroma
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from bs4 import BeautifulSoup
from ollama import Client
import ollama

import uuid

import chromadb
from chromadb.config import Settings
import os

import ollama_embeddings

# Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"

# Index

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
urls_rus = [
    "https://neiro-psy.ru/blog/10-sovetov-dlya-poiska-lyubvi-i-znakomstv-pri-socialnoj-trevoge",
]
# Variables group for chroma collections management
query_chroma_test: str = "What is the Adversarial attacks?"
query_chroma_test_rus: str = "Что такое социальная тревога?"
#
collection_name: str = "rag-rus-ollama"
#

chat_ollama_url_home: str = "http://192.168.1.57:11434"
chat_ollama_url_outdoor: str = "http://46.0.234.32:11434"
chat_ollama_url_belgium: str = "http://46.183.187.205:11434"

chroma_client_host_home: str = "192.168.1.57"
chroma_client_host_outdoor: str = "46.0.234.32"
chroma_client_port: int = 8000
local_llm: str = "llama3.1:8b-instruct-fp16"


def txt_splitter(add_urls: List[str]) -> List[Document]:
    docs = [WebBaseLoader(url).load() for url in add_urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=30
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print("Text Splitting Done.")
    return doc_splits


def create_chroma_client():
    """create the chroma client"""
    # client = chromadb.HttpClient(host=chroma_client_host_home, port=chroma_client_port,
    #                              settings=Settings(allow_reset=True))
    client = chromadb.HttpClient(host=chroma_client_host_home, port=chroma_client_port)
    return client


def create_collection(client):
    # client.reset()  # resets the database
    docs_ = txt_splitter(urls_rus)  # Подставляем правильную переменную в зависимости от источника данных.
    collection = client.create_collection(collection_name)
    for doc_ in docs_:
        collection.add(
            ids=[str(uuid.uuid1())], metadatas=doc_.metadata, documents=doc_.page_content
        )


# ollama Embeddings

chroma_client = chromadb.HttpClient(host=chroma_client_host_home, port=chroma_client_port)
ollama_client = Client(host=chat_ollama_url_home)
emb_model = "llama3.1:8b-instruct-fp16"
# emb_model = "mxbai-embed-large"
# emb_model = "nextfire/paraphrase-multilingual-minilm"
ll_model = "llama3.1:8b-instruct-fp16"


# STEP 0 :: Preparing the conditions
def preconditioning(target_name):
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

def ollama_create_collection(coll_name):
    """Создание коллекции с заданным именем"""
    ollama_collection = chroma_client.create_collection(name=coll_name)
    return ollama_collection


def ollama_add_data_to_collection():
    """Добавляем встраивания в коллекцию"""

    # Передаем результаты работы предварительных процедур:
    docs = txt_splitter(urls)  # Подставляем правильную переменную в зависимости от источника данных.
    collection = ollama_create_collection(collection_name)

    print(f"Создаем коллекцию (STEP 1): {collection_name}")
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
prompt = query_chroma_test


# STEP 2 :: generate an embedding for the prompt and retrieve the most relevant doc

def ollama_query_to_collection(col_name):
    print(f"Делаем запрос к коллекции (STEP 2): {col_name}")
    find_collection = chroma_client.get_collection(col_name)
    response = ollama_client.embeddings(
        prompt=prompt,
        model=emb_model
    )
    results = find_collection.query(
        query_embeddings=[response["embedding"]],
        n_results=2
    )

    data = results['documents'][0][0] + "\n\n\n" + results['documents'][0][1]

    print(f"Ответ Ollama Embeddings {emb_model} на вопрос {prompt} (STEP 2):")
    print(data)
    print(" ###### ")


# Retrieval Grader

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=txt_splitter(urls),
    collection_name=collection_name,
    embedding=ollama_embeddings
    # embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
)
retriever = vectorstore.as_retriever()

llm = ChatOllama(model=local_llm, base_url=chat_ollama_url_outdoor, format="json", temperature=0)

prompt = PromptTemplate(
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

retrieval_grader = prompt | llm | JsonOutputParser()

question = "agent memory"
docs = retriever.invoke(question)

doc_txt = docs[1].page_content

print(f"Квалификация вопроса: {question}")
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

# Должен сообщить: {'score': 'yes'}
