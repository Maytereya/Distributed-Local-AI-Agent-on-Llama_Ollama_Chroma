from typing import Literal, Optional

from chromadb.api.models.Collection import Collection
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader, DirectoryLoader

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import uuid
from typing import List
import warnings
import config as c  # Here are all ip, llm names and other important things

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)

# Index

urls_rus = [
    "https://neiro-psy.ru/blog/monopobiya-kak-nazyvaetsya-strah-ostavatsya-odnomu-i-kak-s-nim-spravitsya",
    "https://neiro-psy.ru/blog/bipolyarnoe-rasstrojstvo-i-depressiya-ponimanie-razlichij",
    "https://neiro-psy.ru/blog/razdvoenie-lichnosti-kak-raspoznat-simptomy-i-obratitsya-za-pomoshchyu",
]

# PDF document to load pass
# file_path = "pdf/taking_guidelines.pdf"
file_path = "Upload/"

# Variables group for chroma collections management
# collection_name: str = "rag-chr-pdf-side-eff-cosine-LaBSE-en-ru"
# collection_name: str = "txt-side-eff-cosine-LaBSE-en-ru"
# collection_name: str = "txt-side-eff-cosine-MiniLM-L12-v2"
collection_name: str = "txt-side-eff-cosine-distiluse-base-multilingual-cased-v1"

# ini the chroma client
chroma_client = chromadb.HttpClient(host=c.chroma_host, port=c.chroma_port)

# Загрузка модели для эмбеддингов
#
# model_only = "cointegrated/LaBSE-en-ru"
# model_only = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' # эффективность под вопросом
# model_only = 'sentence-transformers/LaBSE'
model_only = "sentence-transformers/distiluse-base-multilingual-cased-v1"

# ==== Медицинские модели =====
# model_only = "dmis-lab/biobert-v1.1"
# model_only = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

# ==== Русские модели =====
# model_only = "DeepPavlov/rubert-base-cased"
# model_only = "ai-forever/sbert_large_nlu_ru"

sent_transform_model = SentenceTransformer(model_only)

class ChromaService:
    def __init__(self, host: str, port: int):
        self.chroma_client = chromadb.HttpClient(host=host, port=port)

    def info_chroma(self):
        print("Chroma current version: " + str(self.chroma_client.get_version()))
        print("Collections count: " + str(self.chroma_client.count_collections()))
        print("Chroma heartbeat: " + str(round(self.chroma_client.heartbeat() / 3_600_000_000_000, 2)), " hours")

    def reset_chroma(self):
        self.chroma_client.reset()
        self.chroma_client.clear_system_cache()

    def preconditioning(self, target_name: str):
        """Подготавливаем условия для создания и использования коллекций. В данном случае, удаляем коллекцию,
        если она уже существует"""
        # Имя коллекции = target_name
        list_col = self.chroma_client.list_collections()
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


class LocalHuggingFaceEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Only for Chroma server database
    For chroma vectorstore use model_only

    """

    def __call__(self, input: Documents) -> Embeddings:
        # Convert the numpy array to a Python list
        return sent_transform_model.encode(input, show_progress_bar=True).tolist()


def web_txt_splitter(add_urls) -> List[Document]:
    """
    Split the web documents into list of Document

    """
    doc_splits: List[Document] = []
    if add_urls:  # Check the list is not empty
        docs = []
        for url in add_urls:
            loaded_docs = WebBaseLoader(url).load()
            if loaded_docs:
                docs.append(loaded_docs)

        docs_list = [item for sublist in docs for item in sublist]

        if docs_list:  # Check the docs_list is not empty
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000, chunk_overlap=100
            )
            doc_splits = text_splitter.split_documents(docs_list)
            print("Web document splitting done.")
        else:
            print("Loaded documents are empty.")
    else:
        print("There are no web documents passed")

    return doc_splits


def txt_loader(path: str) -> List[Document]:
    """
    Load & Split the txt documents into list of Document

    """
    split_docs: List[Document] = []
    text_loader_kwargs = {"autodetect_encoding": True}

    loader = DirectoryLoader("Upload/", glob="**/*.txt", loader_cls=TextLoader,
                             show_progress=True,
                             loader_kwargs=text_loader_kwargs)
    docs = loader.load()

    if docs:  # Check the list is not empty
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=3500, chunk_overlap=800
        )
        split_docs = text_splitter.split_documents(docs)
        print("Web document splitting done.")
    else:
        print("There are no web documents passed")

    return split_docs


def pdf_loader(path: str) -> List[Document]:
    """
    Load and split PDF by page with page number and path metadata

    """
    docs = []
    loader = PyPDFLoader(path)
    docs_lazy = loader.lazy_load()

    # Инициализация счётчика
    step = 0

    for doc in docs_lazy:
        step += 1  # Увеличиваем счётчик на каждом шаге
        print(f"\rШаг {step}: обработка страницы...", end='', flush=True)
        docs.append(doc)

    print("\nВсе страницы обработаны!")

    return docs


def handle_collection(existed_collection):
    collection = chroma_client.get_collection(name=existed_collection,
                                              embedding_function=LocalHuggingFaceEmbeddingFunction())
    # peek = collection.peek()  # returns a list of the first 10 items in the collection
    count = collection.count()  # returns the number of items in the collection
    print(f'the number of items in the collection: {count}')
    # print(f'list of the first 10 items in the collection: {peek}')


def create_collection(exist_collection_name) -> Collection | None:
    """
    Создание коллекции с заданным именем
    Optional metadata argument which can be used to customize the distance method of the embedding
    space by setting the value of hnsw:space

    Valid options for hnsw:space are "l2", "ip", or "cosine".
    The default is "l2" which is the squared L2 norm.
    """
    try:
        chroma_collection = chroma_client.create_collection(name=exist_collection_name,
                                                            embedding_function=LocalHuggingFaceEmbeddingFunction(),
                                                            metadata={"hnsw:space": "cosine"})
        if chroma_collection:
            print(f"Creating collection: {exist_collection_name}")
            return chroma_collection
        else:
            print(f"Failed to create collection: {exist_collection_name}")
            return None
    except Exception as e:
        print(f"An error occurred while creating the collection: {e}")
        return None


def add_data(exist_collection_name, upload_type: Literal["URL", "PDF", "TXT"], add_urls: Optional[list] = None,
             add_path: Optional[str] = None):
    """

    Добавляет данные в существующую коллекцию Chroma DB в зависимости от типа загрузки (URL или PDF).

    :param exist_collection_name: Название существующей коллекции, в которую будут добавляться данные.
    :param upload_type: Тип загружаемых данных. Должен быть указан как "URL" или "PDF".
        - "URL": Загрузка текстовых данных по списку URL-адресов.
        - "PDF": Загрузка данных из PDF-файла по указанному пути.
    :param add_urls: Список URL-адресов для загрузки (только если upload_type = "URL").
    :param add_path: Путь к PDF-файлу для загрузки (только если upload_type = "PDF").
    :return: None. Функция выводит процесс загрузки и добавления данных в коллекцию на экран.
    :raises Exception: Если возникает ошибка при загрузке данных или взаимодействии с коллекцией.
    """

    print(f"Adding data to collection: {exist_collection_name}")
    docs = []
    try:
        # Получаем документы
        if upload_type == "URL" and add_urls is not None:
            print("Загружаем документ по URL")
            docs = web_txt_splitter(add_urls)
            if not docs:
                print("No URL documents to process.")
                return

        elif upload_type == "PDF" and add_path is not None:
            print("Загружаем PDF-документ")
            docs = pdf_loader(add_path)
            if not docs:
                print("No pdf documents to process.")
                return

        elif upload_type == "TXT" and add_path is not None:
            print("Загружаем TXT-документ")
            docs = txt_loader(add_path)
            if not docs:
                print("No txt documents to process.")
                return

        else:
            print("Не переданы необходимые данные для загрузки.")

        # Получаем коллекцию
        collection = chroma_client.get_collection(name=collection_name,
                                                  embedding_function=LocalHuggingFaceEmbeddingFunction())
        if not collection:
            print(f"Collection {collection_name} not found.")
            return

        total_chunks = len(docs)
        print(f"Total chunks: {total_chunks}")

        for i, doc in enumerate(docs, start=1):
            print(f"\rChunk cycles remain: {i}/{total_chunks}", end="", flush=True)
            print()
            print(f"ids: {str(uuid.uuid1())}")
            print(f"metadatas: {doc.metadata}")
            print(f"type: {doc.type}")
            print(f"page_content: {doc.page_content[:10]}")
            try:

                # Добавляем документы в коллекцию
                collection.add(
                    ids=[str(uuid.uuid1())],
                    metadatas=doc.metadata,
                    documents=doc.page_content
                )
            except Exception as e:
                print(f"Failed to process document {i}/{total_chunks}: {e}")

        print()  # Печатаем пустую строку в конце чтобы счетчик не переносился.
    except Exception as e:
        print(f"An error occurred while adding data: {e}")


def query_collection(existed_collection, question: str, n_results: int = 2) -> Optional[
    List[Document]]:
    collection = chroma_client.get_collection(name=existed_collection,
                                              embedding_function=LocalHuggingFaceEmbeddingFunction())

    print(f"Ответ на вопрос: '{question}'")
    result = collection.query(
        query_texts=question,
        n_results=n_results,
        # where={"metadata_field": "is_equal_to_this"},
        # where_document={"$contains": question}
    )

    print("QUERY RESPONSE (def query_collection): ")
    print()
    print(f"Documents:  {result["documents"][0]}")
    print(f"Distances:  {result["distances"][0]}")
    # print(f"Metadata:  {result["metadatas"][0][0]}")

    documents = [
        Document(page_content=doc, metadata=meta)
        for sublist in result['documents']
        for doc in sublist
        for sub_metadata in result["metadatas"]
        for meta in sub_metadata
    ]
    # Но можно возвращать List[str], str или Document при необходимости.
    # TODO: check return format!
    return documents


# :: Chroma Vector Store ::
def vs_query(existed_collection: str, question: str, search_type: Literal["simil", "simil_score", "vector", "mmr"],
             k: int = 3, filters: dict = None):
    """
    Выполняет поиск по векторной базе данных с использованием различных типов поиска.

    :param existed_collection: Название существующей коллекции в Chroma DB, в которой выполняется поиск.
    :param question: Вопрос или запрос, по которому выполняется поиск.
    :param search_type: Тип поиска, который будет использован:
        - "simil": Поиск по векторному сходству без оценки схожести.
        - "simil_score": Поиск по векторному сходству с возвращением оценки схожести.
        - "vector": Поиск по вектору запроса.
        - "mmr": Поиск с использованием Maximal Marginal Relevance (MMR).
    :param k: Количество возвращаемых результатов (по умолчанию = 3).
    :param filters: Словарь фильтров для ограничения поиска по метаданным.
                    Если не указан, используется фильтр {"source": "pdf/side_effects_guidelines.pdf"}.
    :return: Возвращает None, но выводит результаты поиска в виде содержимого страниц и метаданных.
    :raises Exception: Если возникает ошибка во время выполнения поиска.
    """
    print(f"Ответ на вопрос: {question}")

    documents: List[Document] = []

    try:
        embedding_function = HuggingFaceEmbeddings(model_name=model_only)

        vector_store_from_client = Chroma(
            client=chroma_client,
            collection_name=existed_collection,
            embedding_function=embedding_function,
        )

        # Default filter
        if filters is None:
            filters = {"source": "pdf/side_effects_guidelines.pdf"}

        def print_results(all_results):
            for c_res in all_results:
                if isinstance(c_res, tuple):
                    c_res, score = c_res
                    print(f"* [SIM={score:.3f}] {c_res.page_content} [{c_res.metadata}]")
                else:
                    print(f"* {c_res.page_content} [{c_res.metadata}]")

        if search_type == "simil":
            documents = vector_store_from_client.similarity_search(
                question,
                k=k,
                filter=filters,
            )
            print("search_type = similarity")
            print_results(documents)
            # for res in results:
            #     print(f"* {res.page_content} [{res.metadata}]")

        elif search_type == "simil_score":
            results = vector_store_from_client.similarity_search_with_score(
                question,
                k=k,
                filter=filters,
            )
            print("search_type = similarity with score")
            print_results(results)
            documents = [
                Document(page_content=doc.page_content, metadata=score)
                for doc, score in results
            ]
            # for res, score in results:
            #     print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

        elif search_type == "vector":
            documents = vector_store_from_client.similarity_search_by_vector(
                embedding=embedding_function.embed_query(question), k=1
            )
            print("search_type = search by vector")
            print_results(documents)
            # for doc in results:
            #     print(f"* {doc.page_content} [{doc.metadata}]")

        elif search_type == "mmr":
            retriever = vector_store_from_client.as_retriever(
                search_type="mmr", search_kwargs={"k": k, "fetch_k": 5}
            )
            print("search_type = mmr")
            documents = retriever.invoke(question, filter=filters)
            print_results(documents)
            # print(res[0].page_content)
            # print(res[0].metadata)
    except Exception as e:
        print(f"An error occurred while searching vector store (using def vs_query()): {e}")

    return documents


# Тестирование
if __name__ == '__main__':
    print(':: TESTING ::')
    print("Preparing an environment for working with collections...")
    ch_s = ChromaService(c.chroma_host, c.chroma_port)
    ch_s.info_chroma()
    ch_s.preconditioning(collection_name)
    print("Create collection...")
    create_collection(collection_name)
    print("Add web/pdf/txt data to collection...")
    add_data(exist_collection_name=collection_name, upload_type="TXT", add_path=file_path)
    print("Common data info:")
    handle_collection(collection_name)
    print(f"Query to collection {collection_name}:")
    vs_query(collection_name, "апатия", search_type="vector", k=2)
    print(" ==== ==== ")
    print(" ==== ==== ")
    query_collection(collection_name, "апатия", n_results=2)
