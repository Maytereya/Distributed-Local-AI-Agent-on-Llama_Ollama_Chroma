# Local RAG Agent on LangGraph
__Llama 3.1, Ollama 0.3.4, Chroma 0.5.4, Tavily AI__

### Code Description

The code provides a framework for an agent that uses a state graph to handle user queries, perform actions such as document retrieval, answer generation, and web search.

### Core Logic

**AgentState**: Defines the data structure for storing the agent’s current state. This is a TypedDict with fields for messages, the question, generation, web search state, and documents.

**Agent**: The `__init__` constructor initializes the system, tools, and state graph.

- The `retrieve` method fetches documents from the indexed storage based on the query.

- The `generate` method produces an answer using the retrieved documents.

- The `grade_documents` method assesses the relevance of documents to the given query.

- The `web_search` method performs an Internet search and appends the results to the documents.

- The `route_question` method determines whether the query should be directed to a web search or the vector store.

- The `decide_to_generate` method decides whether to proceed with generating a response or to perform a web search.

- The `grade_generation_v_documents_and_question` method checks whether the generated answer is correct and matches the query.


## The same on russian below 

### Описание кода

Код представляет собой структуру для агента, который использует граф состояний для обработки запросов пользователя, выполняет действия, такие как поиск документов, генерация ответов и веб-поиск. 

### Основная логика

AgentState: определяет структуру данных для хранения текущего состояния агента. Это TypedDict с полями для сообщений, вопроса, генерации, состояния веб-поиска и документов.

Agent: Конструктор __init__ инициализирует систему, инструменты и граф состояния.

Метод retrieve извлекает документы из индексированного хранилища на основе вопроса.

Метод generate генерирует ответ, используя извлеченные документы.

Метод grade_documents оценивает релевантность документов к заданному вопросу.

Метод web_search выполняет поиск в Интернете и добавляет результаты к документам.

Метод route_question определяет, следует ли направить запрос на веб-поиск или векторное хранилище.

Метод decide_to_generate решает, следует ли продолжать генерацию ответа или выполнить веб-поиск.

Метод grade_generation_v_documents_and_question проверяет, корректен ли ответ и соответствует ли он вопросу.
