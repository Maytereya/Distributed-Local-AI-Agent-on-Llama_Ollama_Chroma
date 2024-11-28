#Async Retriever for Chroma DB v 3.0

# Model loading for embeddings
# from InstructorEmbedding import INSTRUCTOR
# i_model = INSTRUCTOR('hkunlp/instructor-large')

# model_only = "cointegrated/LaBSE-en-ru"
# model_only = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' # эффективность под вопросом
# model_only = 'sentence-transformers/LaBSE'
# model_only = "sentence-transformers/distiluse-base-multilingual-cased-v1"


# ==== Medical models =====
# model_only = "dmis-lab/biobert-v1.1"
# model_only = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

# ==== Russian models =====
# model_only = "ai-forever/sbert_large_nlu_ru"

import asyncio
from typing import Literal, Optional, Union
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
import config as c  # # Stores IP, LLM names, and other important configurations
import formulate

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)

# Initialize Chroma client
chroma_client = chromadb.HttpClient(host=c.chroma_host, port=c.chroma_port)


def choose_model(model: Literal["distiluse", "sbert", "instructor", "default"] = "default",
                 return_type: Literal["model", "name"] = "model") -> Union[SentenceTransformer, str]:
    """
    Select and return either the sentence transformer model or the model name string.

    :param model: The name of the model to select. Options: "distiluse", "sbert", "instructor", "default".
    :param return_type: Determines whether to return the model object ("model") or just the model name ("name").
    :return: The selected model as a SentenceTransformer object or a string with the model name.
    """

    model_mapping = {
        "distiluse": "sentence-transformers/distiluse-base-multilingual-cased-v1",
        "sbert": "ai-forever/sbert_large_nlu_ru",
        "instructor": "hkunlp/instructor-xl",
        "default": "cointegrated/LaBSE-en-ru"
    }

    selected_model_name = model_mapping.get(model, model_mapping["default"])

    if return_type == "name":
        # Return only the model name as a string
        return selected_model_name

    else:
        # Return the model object
        return SentenceTransformer(selected_model_name)


class ChromaService:
    """
    A service class for managing Chroma DB operations.
    """
    def __init__(self, host: str, port: int):
        self.chroma_client = chromadb.HttpClient(host=host, port=port)

    def info_chroma(self):
        """
        Print current Chroma version, collections count, and heartbeat - health satus of database.
        """
        print("Chroma current version: " + str(self.chroma_client.get_version()))
        print("Collections count: " + str(self.chroma_client.count_collections()))
        print("Chroma heartbeat: " + str(round(self.chroma_client.heartbeat() / 3_600_000_000_000, 2)), " hours")

    def reset_chroma(self):
        self.chroma_client.reset()
        self.chroma_client.clear_system_cache()

    def display_collections(self):
        """
        Display all collections stored in Chroma DB.
        :return: List of collections on screen.
        """
        list_col = self.chroma_client.list_collections()
        for col in list_col:
            # Преобразуем объект в строку и находим значение name
            name_part = str(col).split(", name=")[1].rstrip(")")
            print(name_part)
            # print(col)
            print("=============")

    def preconditioning(self, target_name: str):
        """
        Prepare the conditions for creating and using collections by removing an existing collection if found.
        :param target_name: The target collection name to check and reset if necessary.
        """
        # Имя коллекции = target_name
        list_col = self.chroma_client.list_collections()
        found = False
        for col in list_col:
            # Convert the object to string and extract the name value
            name_part = str(col).split(", name=")[1].rstrip(")")

            if name_part == target_name:
                found = True
                break

        if found:
            print(f"Collection with name '{target_name}' exists, we'll delete it")
            self.chroma_client.delete_collection(target_name)
        else:
            print(f"Collection with name '{target_name}' does not exist, we'll create it on the next step.")


class HuggingFaceEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A custom embedding function for Chroma server database.

    This class allows embedding documents using a pre-selected HuggingFace model.
    Note:
    - Default embedding model is "cointegrated/LaBSE-en-ru".
    - To switch to a different model (e.g., "sbert"), specify it using the `set_model` method.
    """

    def set_model(self, model: Literal["distiluse", "sbert", "instructor", "default"] = "default"):
        """
        Set the model to be used for embedding.

        :param model: The name of the model to use for embeddings. Options: "distiluse", "sbert", "instructor", or "default".
        """
        self._model = choose_model(model)

    def __call__(self, input: Documents) -> Embeddings:
        """
        Embed the input documents using the pre-selected model.

        :param input: A list of LangChain Document objects to embed.
        :return: A list of embeddings in Python list format.
        """
        if not hasattr(self, '_model'):
            # Use the default model if none has been set
            self._model = choose_model("default")

        # Convert numpy array to Python list
        return self._model.encode(input, show_progress_bar=True, ).tolist()


def web_txt_splitter(add_urls) -> List[Document]:
    """
    Split web documents into smaller chunks for processing.

    :param add_urls: A list of URLs to fetch and split into chunks.
    :return: A list of Document objects containing the split text.
    """

    doc_splits: List[Document] = []
    if add_urls:  # Ensure the URL list is not empty
        docs = []
        for url in add_urls:
            loaded_docs = WebBaseLoader(url).load()
            if loaded_docs:
                docs.append(loaded_docs)

        # Flatten the list of loaded documents
        docs_list = [item for sublist in docs for item in sublist]

        if docs_list:  # Ensure the document list is not empty
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


def txt_loader(path: str = "Upload/") -> List[Document]:
    """
    Load and split text files from a directory into chunks.

    :param path: The directory path containing text files to load.
    :return: A list of Document objects with the split text.
    """

    split_docs: List[Document] = []
    text_loader_kwargs = {"autodetect_encoding": True}

    loader = DirectoryLoader(path=path, glob="**/*.txt", loader_cls=TextLoader,
                             show_progress=True,
                             loader_kwargs=text_loader_kwargs)
    docs = loader.load()

    if docs:  # Ensure the list is not empty
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
    Load and split a PDF document by pages, retaining page number and path metadata.

    :param path: The file path of the PDF document to load.
    :return: A list of Document objects containing the split pages.
    """

    docs = []
    loader = PyPDFLoader(path)
    docs_lazy = loader.lazy_load()

    # Step counter
    step = 0

    for doc in docs_lazy:
        step += 1  # Increment step for each page processed
        print(f"\rStep {step}: Processing page...", end='', flush=True)
        docs.append(doc)

    print("\nAll pages have been processed.")

    return docs


def handle_collection(existed_collection):
    collection = chroma_client.get_collection(name=existed_collection,
                                              embedding_function=HuggingFaceEmbeddingFunction())
    # peek = collection.peek()  # returns a list of the first 10 items in the collection
    count = collection.count()  # returns the number of items in the collection
    print(f'the number of items in the collection: {count}')
    # print(f'list of the first 10 items in the collection: {peek}')


def create_collection(exist_collection_name,
                      model: Literal["distiluse", "sbert", "default"] = "default") -> Collection | None:
    """
    Создание коллекции с заданным именем
    Optional metadata argument which can be used to customize the distance method of the embedding
    space by setting the value of hnsw:space

    Valid options for hnsw:space are "l2", "ip", or "cosine".
    The default is "l2" which is the squared L2 norm.
    """
    embedding_function = HuggingFaceEmbeddingFunction()
    embedding_function.set_model(model)  # И так установлено по умолчанию.

    try:
        chroma_collection = chroma_client.create_collection(name=exist_collection_name,
                                                            embedding_function=embedding_function,
                                                            metadata={"hnsw:space": "cosine"})
        if chroma_collection:
            print(f"Creating collection: {exist_collection_name}")
            return chroma_collection
        else:
            print(f"Failed to create collection: {exist_collection_name}")
            return None
    except Exception as e:
        print(f"An error occurred while creating the collection / Ошибка при создании коллекции: {e}")
        return None


def add_data(exist_collection_name, upload_type: Literal["URL", "PDF", "TXT"], add_urls: Optional[list] = None,
             add_path: Optional[str] = None, model: Literal["distiluse", "sbert", "instructor", "default"] = "default"):
    """

    Добавляет данные в существующую коллекцию Chroma DB в зависимости от типа загрузки (URL или PDF).

    :param model: Выбор между "LaBSE-en-ru" (по умолчанию), "sbert", "instructor", "distiluse".
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

        embedding_function = HuggingFaceEmbeddingFunction()
        embedding_function.set_model(model)

        # Получаем коллекцию
        collection = chroma_client.get_collection(name=exist_collection_name,
                                                  embedding_function=embedding_function)
        if not collection:
            print(f"Collection {exist_collection_name} not found.")
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


# :: Chroma DB ::
def query_collection(existed_collection, question: str, contains: str = " ", n_results: int = 2,
                     model: Literal["distiluse", "sbert", "instructor", "default"] = "default") -> Optional[
    List[Document]]:
    embedding_function = HuggingFaceEmbeddingFunction()
    embedding_function.set_model(model)

    collection = chroma_client.get_collection(name=existed_collection,
                                              embedding_function=embedding_function)

    # print(f"Ответ на вопрос: '{question}'")
    result = collection.query(
        query_texts=question,
        n_results=n_results,
        # where={"metadata_field": "is_equal_to_this"},
        where_document={"$contains": contains}
    )

    # print("QUERY RESPONSE (def query_collection): ")
    # print()
    # print(f"Documents:  {result["documents"][0]}")
    # print(f"Distances:  {result["distances"][0]}")
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
             model: Literal["distiluse", "sbert", "instructor", "default"] = "default", k: int = 3,
             filters: dict = None, fetch_k: int = 25, lambda_mult: float = 0.85):
    """
    Выполняет поиск по векторной базе данных с использованием различных типов поиска и выбранной модели эмбеддингов.

    :param lambda_mult: Retrieve more documents with higher diversity (0.25). Lower diversity: 0.85. Useful if your dataset has many similar documents
    :param fetch_k: Fetch more documents for the MMR algorithm to consider
    :param existed_collection: Название существующей коллекции в Chroma DB, в которой выполняется поиск.
    :param question: Вопрос или запрос, по которому выполняется поиск.
    :param search_type: Тип поиска, который будет использован:
        - "simil": Поиск по векторному сходству без оценки схожести.
        - "simil_score": Поиск по векторному сходству с возвращением оценки схожести.
        - "vector": Поиск по вектору запроса.
        - "mmr": Поиск с использованием Maximal Marginal Relevance (MMR).
    :param model: Название модели эмбеддингов, которая будет использована ("distiluse", "sbert", или "default").
    :param k: Количество возвращаемых результатов (по умолчанию = 3).
    :param filters: Словарь фильтров для ограничения поиска по метаданным. Если не указан, используется фильтр
                    {"source": "pdf/side_effects_guidelines.pdf"}.
    :return: Список документов, полученных в результате поиска.
    :raises Exception: Если возникает ошибка во время выполнения поиска.
    """
    # print(f"Ответ на вопрос: {question}")

    documents: List[Document] = []

    try:
        # Получаем название модели через функцию choose_model
        model_name = choose_model(model, return_type="name")
        embedding_function = HuggingFaceEmbeddings(model_name=model_name)

        vector_store_from_client = Chroma(
            client=chroma_client,
            collection_name=existed_collection,
            embedding_function=embedding_function,
        )

        # Фильтр по умолчанию
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
            # print_results(documents)

        elif search_type == "simil_score":
            results = vector_store_from_client.similarity_search_with_score(
                question,
                k=k,
                filter=filters,
            )
            print("search_type = similarity with score")
            # print_results(results)
            documents = [
                Document(page_content=doc.page_content, metadata=score)
                for doc, score in results
            ]

        elif search_type == "vector":
            documents = vector_store_from_client.similarity_search_by_vector(
                embedding=embedding_function.embed_query(question), k=k
            )
            print("search_type = search by vector")
            # print_results(documents)

        elif search_type == "mmr":
            retriever = vector_store_from_client.as_retriever(
                search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult, "filters": filters}
            )
            print("search_type = mmr")
            documents = retriever.invoke(question, )
            # print_results(documents)

    except Exception as e:
        print(f"An error occurred while searching vector store (using def vs_query()): {e}")

    return documents


# Тестирование
if __name__ == '__main__':
    print(':: TESTING ::')

    # PDF document to load pass
    # file_path = "pdf/taking_guidelines.pdf"
    # txt document directory pass
    file_path = "Upload/"

    urls_rus = [
        "https://neiro-psy.ru/blog/monopobiya-kak-nazyvaetsya-strah-ostavatsya-odnomu-i-kak-s-nim-spravitsya",
        "https://neiro-psy.ru/blog/bipolyarnoe-rasstrojstvo-i-depressiya-ponimanie-razlichij",
        "https://neiro-psy.ru/blog/razdvoenie-lichnosti-kak-raspoznat-simptomy-i-obratitsya-za-pomoshchyu",
    ]

    collection_name: str = "25_10_2024_LaBSE-en-ru_pdf"
    question = "причины апатии, коррекция"

    print("Preparing an environment for working with collections...")
    ch_s = ChromaService(c.chroma_host, c.chroma_port)
    ch_s.info_chroma()
    ch_s.display_collections()
    #
    # ch_s.preconditioning(collection_name)
    # print("Create collection...")
    # create_collection(collection_name)
    # print("Add web/pdf/txt data to collection...")
    # add_data(exist_collection_name=collection_name, upload_type="PDF",
    #          add_path="Upload/side_effects_guideline_for_RAG_paged.pdf", )
    # print("Common data info:")
    # handle_collection(collection_name)
    print(f"Query to collection {collection_name}:")
    print("Вопрос:", question)


    # documents = vs_query(collection_name, question=question, search_type="mmr", k=5, )
    #
    # for doc in documents:
    #     print("##############")
    #     print(doc.page_content)
    #     print(doc.metadata)
    #     print("##############")
    # print(" ==== ==== ")
    # print(" ==== ==== ")
    async def main(question_in):
        return await formulate.extract_keyword(question_in)


    a = asyncio.run(main(question))

    documents = query_collection(collection_name, question, contains=a, n_results=2,
                                 model="default")
    for doc in documents:
        print("###############")
        print(doc.page_content)
        print(doc.metadata)
        print("###############")
