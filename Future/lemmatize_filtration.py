import pymorphy2

from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Инициализация функции эмбеддинга
embedding_function = HuggingFaceEmbeddings(model_name="your_model_name")

# Инициализация Chroma Vector Store
vector_store = Chroma(
    collection_name="your_collection_name",
    embedding_function=embedding_function,
    persist_directory="path_to_persist_directory"
)

# Определение запроса и ключевого слова
query = "Ваш поисковый запрос"
keyword = "ключевое слово"

# Инициализация ретривера с использованием MMR-поиска
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.5}
)

# Выполнение поиска
results = retriever.get_relevant_documents(query)

# Фильтрация результатов по наличию ключевого слова
filtered_results = [doc for doc in results if keyword.lower() in doc.page_content.lower()]

# Вывод отфильтрованных результатов
for doc in filtered_results:
    print(f"Содержимое: {doc.page_content}")
    print(f"Метаданные: {doc.metadata}")
    print("-" * 50)

# Инициализация морфологического анализатора
morph = pymorphy2.MorphAnalyzer()


# Функция для приведения текста к начальным формам слов
def lemmatize(text):
    words = text.split()
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmas)


# Пример текста и ключевого слова
text = "В тексте встречаются различные формы слова."
keyword = "форма"

# Лемматизация текста и ключевого слова
lemmatized_text = lemmatize(text)
lemmatized_keyword = lemmatize(keyword)

# Проверка наличия ключевого слова в тексте
if lemmatized_keyword in lemmatized_text:
    print("Ключевое слово найдено в тексте.")
else:
    print("Ключевое слово не найдено в тексте.")

# Фильтрация результатов с учетом лемматизации
filtered_results = [
    doc for doc in results
    if lemmatize(keyword) in lemmatize(doc.page_content)
]
