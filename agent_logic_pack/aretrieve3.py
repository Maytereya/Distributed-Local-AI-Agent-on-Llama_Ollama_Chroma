# Async Retriever for Chroma DB v 3.0
import asyncio
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
from agent_logic_pack import formulate, embedding_filtration, path_handling
import config as c

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)

# Initialize Chroma client
chroma_client = chromadb.HttpClient(host=c.chroma_host, port=c.chroma_port)


def choose_model(model: Literal["distiluse", "sbert", "instructor", "default"] = "default",
                 return_type: Literal["model", "name"] = "model") -> SentenceTransformer | str:
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
            # Преобразуем объект в строку и находим значение name (не работает)
            # name_part = str(col).split(", name=")[1].rstrip(")")
            # print(name_part)
            print(col)
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
            print(f"Collection name '{target_name}' does not exist, we'll create it on the next step.")


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
        print("No web documents passed.")

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
        print("Text document splitting done.")
    else:
        print("No text documents found in the specified path.")

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


def handle_collection(existed_collection: str):
    """
        Retrieve and display details of an existing Chroma DB collection.

        :param existed_collection: The name of the existing collection to retrieve.
    """
    collection = chroma_client.get_collection(name=existed_collection,
                                              embedding_function=HuggingFaceEmbeddingFunction())
    print("Common collection info:")
    # peek = collection.peek()  # returns a list of the first 10 items in the collection
    # count =   # Get the number of items in the collection
    # print(f"list of the first 10 items in the collection: {collection.peek()}")
    print(f'the number of items in the collection: {collection.count()}')


def create_collection(
        new_collection_name: str,
        model: Literal["distiluse", "sbert", "default"] = "default"
) -> Optional[Collection]:
    """
    Create a new collection in Chroma DB with the specified name.

    :param new_collection_name: The name of the new collection to create.
    :param model: The embedding model to use for the collection. Default: "default" (LaBSE-en-ru).
    :return: The created Collection object if successful, otherwise None.
    """
    embedding_function = HuggingFaceEmbeddingFunction()
    embedding_function.set_model(model)  # Set the embedding model.

    try:
        chroma_collection = chroma_client.create_collection(
            name=new_collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"})  # Use cosine similarity as the default metric.

        if chroma_collection:
            print(f"Creating collection: {new_collection_name}")
            return chroma_collection
        else:
            print(f"Failed to create collection: {new_collection_name}")
            return None
    except Exception as e:
        print(f"An error occurred while creating the collection: {e}")
        return None


def add_data(
        exist_collection_name: str,
        upload_type: Literal["URL", "PDF", "TXT"],
        add_urls: List[str] | None = None,
        add_path: str | None = None,
        model: Literal["distiluse", "sbert", "instructor", "default"] = "default"
):
    """
    Add data to an existing Chroma DB collection based on the upload type.

    :param exist_collection_name: The name of the existing collection to which data will be added.
    :param upload_type: The type of data to upload. Options:
        - "URL": Load text data from a list of URLs.
        - "PDF": Load data from a PDF file at the specified path.
        - "TXT": Load data from text files in the specified directory.
    :param add_urls: A list of URLs for loading documents (used if upload_type = "URL").
    :param add_path: The file path or directory path for loading data (used for "PDF" or "TXT").
    :param model: The embedding model to use. Options: "default" (LaBSE-en-ru), "sbert", "instructor", "distiluse".
    """

    print(f"Adding data to collection: {exist_collection_name}")
    docs = []
    try:
        # Fetch documents
        if upload_type == "URL" and add_urls is not None:
            print("Loading documents from URLs...")
            docs = web_txt_splitter(add_urls)
            if not docs:
                print("No documents found at the specified URLs.")
                return

        elif upload_type == "PDF" and add_path is not None:
            print("Loading PDF document...")
            docs = pdf_loader(add_path)
            if not docs:
                print("No content found in the PDF file.")
                return

        elif upload_type == "TXT" and add_path is not None:
            print("Loading TXT documents...")
            docs = txt_loader(add_path)
            if not docs:
                print("No text files found in the specified directory.")
                return

        else:
            print("No valid data provided for upload.")

        # Set embedding function
        embedding_function = HuggingFaceEmbeddingFunction()
        embedding_function.set_model(model)

        # Fetch the collection
        collection = chroma_client.get_collection(name=exist_collection_name,
                                                  embedding_function=embedding_function)
        if not collection:
            print(f"Collection {exist_collection_name} not found.")
            return

        total_chunks = len(docs)
        print(f"Total chunks: {total_chunks}")

        for i, doc in enumerate(docs, start=1):
            print(f"\rProcessing chunk: {i}/{total_chunks}", end="", flush=True)
            print()
            print(f"IDs: {str(uuid.uuid1())}")
            print(f"Metadata: {doc.metadata}")
            print(f"Type: {doc.type}")
            print(f"Content preview: {doc.page_content[:10]}")
            try:

                # Add documents to the collection
                collection.add(
                    ids=[str(uuid.uuid1())],
                    metadatas=doc.metadata,
                    documents=doc.page_content
                )
            except Exception as e:
                print(f"Failed to process document {i}/{total_chunks}: {e}")

        print()  # Print an empty line after processing all chunks
    except Exception as e:
        print(f"An error occurred while adding data: {e}")


# :: Chroma DB ::
def query_collection(
        existed_collection: str,
        question: str,
        contains: str = " ",
        n_results: int = 2,
        model: Literal["distiluse", "sbert", "instructor", "default"] = "default"
) -> Optional[List[Document]]:
    """
    Query a Chroma DB collection and return matching documents.

    :param existed_collection: The name of the collection to query.
    :param question: The query text to search for.
    :param contains: A filter for documents containing this text.
    :param n_results: The number of results to return.
    :param model: The embedding model to use for the query.
    :return: A list of matching Document objects.
    """
    embedding_function = HuggingFaceEmbeddingFunction()
    embedding_function.set_model(model)

    collection = chroma_client.get_collection(name=existed_collection,
                                              embedding_function=embedding_function)

    result = collection.query(
        query_texts=question,
        n_results=n_results,
        # where={"metadata_field": "is_equal_to_this"},
        where_document={"$contains": contains}  # Filter documents containing the specified text.
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
def vs_query(
        existed_collection: str,
        question: str,
        search_type: Literal["simil", "simil_score", "vector", "mmr"],
        model: Literal["distiluse", "sbert", "instructor", "default"] = "default",
        k: int = 3,
        filters: dict = None,
        fetch_k: int = 25,
        lambda_mult: float = 0.85
) -> List[Document]:
    """
    Perform a vector-based search in Chroma DB using various search types.

    :param existed_collection: The name of the collection to search in.
    :param question: The query text to search for.
    :param search_type: The type of search to perform:
        - "simil": Similarity search.
        - "simil_score": Similarity search with scores.
        - "vector": Search by embedding vector.
        - "mmr": Maximal Marginal Relevance (MMR) search.
    :param model: The embedding model to use for the query.
    :param k: The number of results to return.
    :param filters: Filters to apply during the search. Default: {"source": "pdf/side_effects_guidelines.pdf"}.
    :param fetch_k: The number of documents to fetch for MMR.
    :param lambda_mult: Adjusts diversity in MMR results (lower = less diverse).
    :return: A list of Document objects containing the results.
    """

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

        # Default filter if none provided
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
            print("Search type: similarity search")
            # print_results(documents)

        elif search_type == "simil_score":
            results = vector_store_from_client.similarity_search_with_score(
                question,
                k=k,
                filter=filters,
            )
            print("Search type: similarity search with scores")
            # print_results(results)
            documents = [
                Document(page_content=doc.page_content, metadata=score)
                for doc, score in results
            ]

        elif search_type == "vector":
            documents = vector_store_from_client.similarity_search_by_vector(
                embedding=embedding_function.embed_query(question), k=k
            )
            print("Search type: search by vector")
            # print_results(documents)

        elif search_type == "mmr":
            retriever = vector_store_from_client.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult, "filters": filters}
            )
            print("Search type: Maximal Marginal Relevance (MMR)")
            documents = retriever.invoke(question, )
            # print_results(documents)

    except Exception as e:
        print(f"An error occurred during vector store search using def vs_query(): {e}")

    return documents


# Main retrieve function, include all
async def main_retrieve_async(search_type: Literal["vectorstore", "db"] = "vectorstore",
                              return_type: Literal["list", "str"] = "list",
                              k: int = 5,
                              n_results: int = 2,
                              threshold: float = 0.005,
                              collection: str = "25_01_2025_LaBSE-en-ru_pdf",
                              question: str = "причины апатии, коррекция",
                              ) -> List[Document] | str:
    """

    :param return_type: Определяет, в каком виде выдается итог поиска: [Document] или str, по умолчанию [Document].
    :param threshold: Порог срабатывания отсекателя похожих документов. Чем больше значение, тем больше документов попадет в выдачу.
    :param n_results: Определяет количество документов в выдаче при поиске по ChromaDB.
    :param k: Определяет количество документов в выдаче при поиске по Chroma Vectorstore.
    :param search_type: Определяет принцип поиска: Vectorstore с фильтрацией по ключевому слову или ChromaDB c ключевым словом.
    :param collection: Имя коллекции, которая будет создаваться.
    :param question: Вопрос заданный пользователем.
    :return: Список документов для ответа Агента.
    """
    print("Preparing an environment for working with collections...")
    chr_service = ChromaService(c.chroma_host, c.chroma_port)
    chr_service.info_chroma()
    print(f"Query to collection: {collection}")
    print("Question:", question)

    async def keyword_from_question(question_in):
        return await formulate.extract_keyword(question_in)

    keyword = await keyword_from_question(question)

    if search_type == "vectorstore":
        documents = vs_query(collection, question=question, search_type="mmr", k=k, )
        # Need two-step filtration
        print("vectorstore search type enabled")
        filtrated_docs = embedding_filtration.filtrate(keyword, documents, threshold=threshold)
    else:
        # Already filtrated in chroma_db algorythm
        print("chroma_db search type enabled")
        filtrated_docs = query_collection(collection, question, contains=keyword, n_results=n_results,
                                          model="default")

    print(f'{filtrated_docs=}')

    if len(filtrated_docs) == 0:
        filtrated_docs = [
            Document(
                metadata={"source": "neiry-ai.ru"},
                page_content="Совпадений не найдено, cформулируйте запрос иначе",
            )]

    if return_type == "str":
        result_str: str = '\n'.join(
            '\n'.join(f_doc.page_content) if isinstance(f_doc.page_content, list)
            else f_doc.page_content
            for f_doc in filtrated_docs
        )
        return result_str

    return filtrated_docs


# ToDo: доделать!
def main_upload(filename: str = "side_effects_guideline_for_RAG_paged.pdf",
                collection: str = "25_01_2025_LaBSE-en-ru_pdf",
                doc_type: Literal["URL", "PDF", "TXT"] = "PDF",
                ) -> None:
    """
    Create collection and Upload the document in it.

    :param filename:
    :param collection: Name of the collection, that will be created in Chroma database.

    :param doc_type: Choose one of supported types of the document to upload: "URL", "PDF", "TXT".
    :return: None
    """
    chroma_service = ChromaService(c.chroma_host, c.chroma_port)
    chroma_service.info_chroma()
    # ToDo: Функция preconditioning требует доработки
    chroma_service.preconditioning(collection)

    path = path_handling.create_path(filename)
    # Create collection:
    create_collection(collection)
    # Add web/pdf/txt data to collection...
    add_data(exist_collection_name=collection, upload_type=doc_type,
             add_path=path, )

    return None


# Test section
#
if __name__ == '__main__':
    print(':: TESTING ::')
    result = asyncio.run(main_retrieve_async(return_type="str"))
    print("==============")
    print(result)
    print("==============")

# PDF document to load pass
# file_path = "pdf/taking_guidelines.pdf"
# txt document directory pass
# file_path = "../Upload/"

# urls_rus = [
#     "https://neiro-psy.ru/blog/monopobiya-kak-nazyvaetsya-strah-ostavatsya-odnomu-i-kak-s-nim-spravitsya",
#     "https://neiro-psy.ru/blog/bipolyarnoe-rasstrojstvo-i-depressiya-ponimanie-razlichij",
#     "https://neiro-psy.ru/blog/razdvoenie-lichnosti-kak-raspoznat-simptomy-i-obratitsya-za-pomoshchyu",
# ]

# chr_service.display_collections()
#
# chr_service.preconditioning(collection_name)

# Create collection:
# create_collection(collection_name)
# Add web/pdf/txt data to collection...
# add_data(exist_collection_name=collection_name, upload_type="PDF",
#          add_path="/Users/rakhmanov/PycharmProjects/LocalRAGagent0.1/Upload/side_effects_guideline_for_RAG_paged.pdf", )

# for doc in documents:
#     print("##############")
#     print(doc.page_content)
#     print(doc.metadata)
#     print("##############")
# print(" ==== ==== ")

# for doc in documents:
#     print("###############")
#     print(doc.page_content)
#     print(doc.metadata)
#     print("###############")
