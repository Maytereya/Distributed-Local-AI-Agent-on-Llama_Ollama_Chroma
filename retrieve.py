# Ex index.py
# inherit FROM hybrid index v 2.0 copy because of bug in IDE.
# in memory rag changed onto chroma client rag
# Lets try to merge base retriever logic from lesson withs ollama and chroma clients...
# Done! It's working.
import json
from typing import List, Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
import chromadb
import os

from ollama import AsyncClient
import config as c  # Here are all ip, llm names and other important things

# Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"


class ChromaService:
    def __init__(self, host: str, port: int):
        self.chroma_client = chromadb.HttpClient(host=host, port=port)

    def info_chroma(self):
        print("Chroma current version: " + str(self.chroma_client.get_version()))
        print("Collections count: " + str(self.chroma_client.count_collections()))
        print("Chroma heartbeat: " + str(self.chroma_client.heartbeat()))

    def reset_chroma(self):
        self.chroma_client.reset()
        self.chroma_client.clear_system_cache()

    def preconditioning(self, target_name: str):
        """Подготавливаем условия для создания и использования коллекций. В данном случае, удаляем коллекцию,
        если она уже существует"""
        # Имя коллекции = target_name
        list_col = self.chroma_client.list_collections(10)
        found = False
        for col in list_col:
            # Преобразуем объект в строку и находим значение name
            name_part = str(col).split(", name=")[1].rstrip(")")

            if name_part == target_name:
                found = True
                break

        if found:
            print(f"Collection with name '{target_name}' exists, we'll delete it")
            self.chroma_client.delete_collection(target_name)
        else:
            print(f"Collection with name '{target_name}' does not exist, we'll create it on the next step.")


class CreateCollection:
    def __init__(self, ollama_url: str, chroma_host: str, chroma_port: int, embedding_model: str, collection: str,
                 add_urls: List[str] = None):
        self.add_urls = add_urls if add_urls is not None else []
        self.chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        self.ollama_aclient = AsyncClient(host=ollama_url)
        self.emb_model = embedding_model
        self.collection = collection

    def web_txt_splitter(self) -> List[Document]:
        """Split the web documents into list of Document"""
        doc_splits: List[Document] = []
        if self.add_urls:  # Check the list is not empty
            docs = []
            for url in self.add_urls:
                loaded_docs = WebBaseLoader(url).load()
                if loaded_docs:
                    docs.append(loaded_docs)

            docs_list = [item for sublist in docs for item in sublist]

            if docs_list:  # Check the docs_list is not empty
                text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=500, chunk_overlap=20
                )
                doc_splits = text_splitter.split_documents(docs_list)
                print("Web document splitting done.")
            else:
                print("Loaded documents are empty.")
        else:
            print("There are no web documents passed")

        return doc_splits

    def create_collection(self):
        """Создание коллекции с заданным именем"""
        try:
            ollama_collection = self.chroma_client.create_collection(name=self.collection)
            if ollama_collection:
                print(f"Creating collection (STEP 1): {self.collection}")
                return ollama_collection
            else:
                print(f"Failed to create collection: {self.collection}")
                return None
        except Exception as e:
            print(f"An error occurred while creating the collection: {e}")
            return None

    async def add_data(self):
        """Добавляем встраивания в коллекцию"""
        try:
            # Получаем документы
            docs = self.web_txt_splitter()
            if not docs:
                print("No documents to process.")
                return

            # Получаем коллекцию
            collection = self.chroma_client.get_collection(self.collection)
            if not collection:
                print(f"Collection {self.collection} not found.")
                return

            print(f"Adding data to collection (STEP 2): {self.collection}")

            total_chunks = len(docs)

            for i, doc in enumerate(docs, start=1):
                print(f"\rChunk cycles remain: {i}/{total_chunks}", end="", flush=True)

                try:
                    # Асинхронный запрос на создание эмбеддингов
                    response = await self.ollama_aclient.embeddings(model=self.emb_model, prompt=doc.page_content)
                    embedding = response["embedding"]

                    # Добавляем эмбеддинги в коллекцию
                    collection.add(
                        ids=[str(uuid.uuid1())],
                        embeddings=[embedding],
                        metadatas=doc.metadata,
                        documents=doc.page_content
                    )
                except Exception as e:
                    print(f"Failed to process document {i}/{total_chunks}: {e}")

            print()  # Печатаем пустую строку в конце чтобы счетчик не переносился.

        except Exception as e:
            print(f"An error occurred while adding data: {e}")


class QueryCollection:
    def __init__(self, ollama_url: str, chroma_host: str, chroma_port: int,
                 embedding_model: str):
        self.chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        self.ollama_aclient = AsyncClient(host=ollama_url)
        self.emb_model = embedding_model
        self.doc_txt = None

    # ... -> str:
    async def ollama_query_to_collection(self, col_name: str, prompt: str, embedding_model: str) -> Optional[
        List[Document]]:
        print(f"Query to collection: {col_name}")

        # Проверка наличия коллекции
        try:
            collection = self.chroma_client.get_collection(col_name)
            if not collection:
                print(f"Collection '{col_name}' not found.")
                return
        except Exception as e:
            print(f"Failed to get collection '{col_name}': {e}")
            return None

        # Вызов эмбеддингов и запрос к коллекции
        try:
            response = await self.ollama_aclient.embeddings(
                prompt=prompt,
                model=embedding_model,
                keep_alive=-1,
            )

            if "embedding" not in response:
                print("Embedding not found in the response.")
                return None

            results = collection.query(
                query_embeddings=[response["embedding"]],
                n_results=1,
            )

            if not results['documents']:
                print("No documents found in the query results.")
                return None

            # Преобразуем список списков в список объектов Document
            documents = [
                Document(page_content=doc, metadata={})
                for sublist in results['documents']
                for doc in sublist
            ]

            print(f"Ollama embeddings response '{embedding_model}' on question: '{prompt}':")
            return documents

        except Exception as e:
            print(f"An error occurred during the Ollama embeddings query: {e}")
            return None

    async def async_launcher(self, question: str, collection_name: str, ):
        """Запуск асинхронного запроса и обработка исключений"""
        try:
            # Выполняем запрос и ожидаем его завершения
            self.doc_txt = await self.ollama_query_to_collection(collection_name, question, self.emb_model)
            return self.doc_txt  # Явно возвращаем результат
        except Exception as e:
            print(f"An error occurred during the async task: {e}")
            return None  # Возвращаем None в случае ошибки

