# Distributed Local Retrieval-Augmented Generation (RAG) Agent Using LangGraph. 
## Adapted for the Russian Language
__Llama 3.1, vikhr nemo 12b, Command-R, Ollama 0.4.6, Chroma 0.5.4, Tavily AI__

# Agent Operation Algorithm

## Request Processing Workflow

### Speech Recognition
- **Real-time recording:** Operator-patient dialogues are recorded on the fly.
- **Audio-to-text conversion:** Speech is converted to text using [VOSK](https://alphacephei.com/vosk/).

### Keyword Extraction
- **Text analysis:** The input text is analyzed with an LLM (Large Language Model).
- **Keyword extraction:** Key terms are identified for information retrieval.

### Information Retrieval
- **Database query:** The vector database ([ChromaDB](https://www.trychroma.com/)) is queried for relevant information.
- **Text filtering:** Retrieved text is filtered using the `cointegrated/rubert-tiny2` model.

### Response Formation
- **Text generation:** Responses are created using an LLM.
- **Hallucination check:**
  - Ensures the response aligns with the database query results.
  - Verifies correspondence with the extracted keywords.

### Prompt Display
- **Operator assistance:** The generated text is presented to the operator in the chatbot interface.

---

## Task Distribution Between Servers

### Server 1 (2 x NVIDIA RTX 4090)
- Manages the vector database.
- Performs text embedding.
- Handles speech recognition.
- Controls agent logic and integrates with clinical systems.

### Server 2 (4 x AMD Radeon RX 7900 XTX)
- Processes requests using the LLM.
- Generates textual responses.

---

## Tools and Technologies
- **Speech Recognition:** [VOSK](https://alphacephei.com/vosk/)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/)
- **Text Filtering Model:** `cointegrated/rubert-tiny2`
- **Language Models:** Large Language Models (LLMs)

---

## Functional Capabilities of the AI Agent

### AI Agent Features

The AI agent supports a flexible approach to request processing using complex routing logic and retrieval of relevant information. Its key functionalities include:

### 1. Request Recognition and Routing
- Identifies the data source for request processing:
  - **Vector Storage:** Uses [ChromaDB](https://www.trychroma.com/) optimized for result diversity (MMR).
  - **Chat with Memory:** Supports conversational mode with prior interaction history.
  - **Web Search:** Acts as a fallback if no relevant data is found locally.
  - **Session Termination:** Provides an option to end the agent's operation.

### 2. Request Processing
- **Focus and Enhancement of Queries:**
  - Extracts keywords to optimize the search process.
  - Refines queries for precise information retrieval.
- **Multi-Step Data Retrieval:** Ensures high accuracy by:
  - Utilizing various search strategies in the vector database:
    - High response diversity (MMR, `lambda_mult=0.25`).
    - Moderate diversity (`lambda_mult=0.85`).
    - Mathematical computation of diversity based on the probability of finding relevant content in the collection.
    - Specific embedding models for fallback searches.
  - Selecting appropriate data collections for specific tasks.

### 3. Filtering and Relevance Verification
- Filters documents based on similarity to user query keywords (cosine distance evaluation).
- Evaluates document relevance:
  - Matches document content to the request using scoring models.
  - Automatically switches to alternate sources if no relevant data is found.

### 4. Response Generation
- Utilizes LLMs like `Llama 3.1 70b fp16`, `rscr/vikhr_nemo_12b`, or `Command-R` for task-specific generation:
  - Generates text based on:
    - Local vector database (RAG).
    - Web search results if local data is insufficient.
  - Verifies responses to eliminate hallucinations and ensure alignment with user queries.

### 5. Conversational Memory
- Maintains dialogue context for more accurate responses.
- Stores interaction history for seamless user experience.

### 6. Error Handling and Session Logic
- Proceeds to the next processing step if relevant data is unavailable, up to session termination.
- Supports multi-step routing through a state graph.

### 7. Asynchronous Processing
- Employs asynchronous methods for parallel task execution, such as:
  - Data retrieval.
  - Response generation.
  - Document relevance assessment.

### 8. Integration with ChromaDB
- Works with ChromaDB through built-in retrievers:
  - Implements search strategies like MMR and cosine similarity for diverse results.
  - Leverages fallback collections built using various embedding models (e.g., LaBSE, Distiluse).

### 9. Response Quality Verification
- Uses evaluation models to:
  - Analyze document relevance.
  - Ensure the adequacy of generated responses.

### 10. Interface and Session Management
- User interaction through a chat interface.
- Session termination command support.
- Automatic handling of multiple requests within a session.

---

### User Interface Features

An asynchronous web server based on [Quart](https://pgjones.gitlab.io/quart/) will provide the interface for user interaction and agent integration into workflows. The interface supports session management for call center operators and file upload for data placement in vector collections based on user access rights.

### Key Interface Features

#### 1. Web Chat Interface
- Users can send queries through a web page (CRM integration or standalone Windows app as per client agreement).
- AI agent responses are displayed in real time.
- Dialogue history is preserved for context-aware responses.

#### 2. Asynchronous Query Handling
- Queries are processed asynchronously for optimal performance.

#### 3. File Upload and Processing
- Supports uploading TXT, PDF, and web links (additional formats per client agreement).
- Uploaded files are stored and processed for addition to ChromaDB.

#### 4. Data Collection Management
- Allows prioritization of specific collections for user requests.

### Detailed Interface Description

#### 1. Main Page
- **Route `/`:** Displays the chat interface using the `index.html` template.

#### 2. Text Query Handling
- **Route `/get`:**
  - Accepts text queries via AJAX requests.
  - Saves queries and responses in session history.
  - Asynchronously calls the `get_agent_response` function for generating responses.

#### 3. File Upload
- **Route `/upload`:**
  - Accepts files via POST requests.
  - Supports `.pdf` and `.txt` formats.
  - Processes files asynchronously and adds their content to the vector database.

#### 4. Document Processing
- **Function `process_file`:**
  - Processes uploaded files.
  - Adds content to ChromaDB for future retrieval.

#### 5. Session Support
- Utilizes `session` for storing dialogue history.
- Ensures context-aware responses in ongoing interactions.

---

### Data Retrieval and Vector Database Integration

The AI agent uses ChromaDB and embedding models to perform high-precision search, data addition, and document processing. Its retrieval logic is built on an adaptive approach to search and collection management.

### Key Retrieval Features

#### 1. Embedding Models
- Supported models for text embeddings include:
  - `cointegrated/LaBSE-en-ru`
  - `sentence-transformers/distiluse-base-multilingual-cased-v1`
  - `ai-forever/sbert_large_nlu_ru`
  - `hkunlp/instructor-xl` (instruction-based embedding training)

#### 2. ChromaDB Collection Management
- Creation, deletion, and listing of collections.
- Environment preparation for collection updates.

#### 3. Data Upload and Processing
- Supported data types:
  - **PDF:** Split into pages and indexed.
  - **TXT:** Split into fragments for optimal indexing.
  - **URL:** Text extracted, processed, and added to the database.

#### 4. Search and Filtering
- Multiple search types:
  - **Simil:** Vector similarity search.
  - **Simil_score:** Similarity with scoring.
  - **Vector:** Direct vector-based search.
  - **MMR:** Maximal Marginal Relevance for diverse results.
- Metadata-based filtering.

#### 5. Flexible Search Parameter Management
- Adjustable parameters:
  - Number of returned documents (`k`).
  - Number of fetched documents (`fetch_k`).
  - Diversity coefficient (`lambda_mult`).

---

### Speech Recognition

The module records speech from a call center operator’s headset and processes it through an ASR server ([VOSK](https://alphacephei.com/vosk/)) via WebSocket. This enables the agent to handle voice requests and convert them into structured text for further processing.

### Key Features

#### 1. Goal
- Provide a voice interface for user interaction.
- Ensure accurate speech-to-text conversion.

#### 2. Workflow
- Records audio signals in real time.
- Sends audio blocks to the ASR server via WebSocket.
- Processes recognized text for use in queries.

#### 3. Technical Details
- Uses `sounddevice` for audio recording.
- Asynchronous processing with `websockets`.
- Customizable settings via command-line arguments.

---

For detailed technical documentation and examples, refer to the [Documentation](./docs/README.md).



## Code Description

The code provides a framework for an agent that uses a state graph to handle user queries, perform actions such as document retrieval, answer generation, and web search.

### Core LangGraph Logic

**AgentState**: Defines the data structure for storing the agent’s current state. This is a TypedDict with fields for messages, the question, generation, web search state, and documents.

**Agent**: The `__init__` constructor initializes the system, tools, and state graph.

- The `retrieve` method fetches documents from the indexed storage based on the query.

- The `generate` method produces an answer using the retrieved documents.

- The `grade_documents` method assesses the relevance of documents to the given query.

- The `web_search` method performs an Internet search and appends the results to the documents.

- The `route_question` method determines whether the query should be directed to a web search or the vector store.

- The `decide_to_generate` method decides whether to proceed with generating a response or to perform a web search.

- The `grade_generation_v_documents_and_question` method checks whether the generated answer is correct and matches the query.

## Contact
For any questions or contributions, feel free to open an issue or submit a pull request.
