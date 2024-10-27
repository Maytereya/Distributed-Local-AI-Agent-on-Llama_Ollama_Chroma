from langchain_core.documents import Document

import config as c  # Here are all ip, llm names and other important things
import time
from ollama import AsyncClient, Options
import json_converter as j

ollama_aclient = AsyncClient(host=c.ollama_url)

# Выбор llm
llm = c.ll_model_big


# Post-processing
def format_docs(docs):
    """Convert Document to string
        This option need for some functions inside async_graph_operator.py: generate_final
    """
    return "\n\n".join(doc.page_content for doc in docs)


async def chat(question: str, history: list = None):
    """
    Just chat with the short-term history
    """

    if history is None:
        history = []

    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
              'You are an assistant for question-answering tasks. Answer the question in plain text format. '
              'If you do not know the answer, simply state that you do not know. '
              'Use a maximum of five sentences and keep your answer concise. '
              'If relevant, reference the history of previous conversations in your answer. '
              'If needed, reason with the user about information found in the conversation history to provide clarity or address follow-up questions. '
              '<|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Question: {question}. \n\n'
              f'History of previous conversations: {history} \n\n'
              'Answer: '
              '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=c.ll_model,
        prompt=prompt,
        # format="json",
        # options=opt,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    print(f"Eval_duration of answer generation: {aresult['eval_duration'] / 1_000_000_000}")
    #

    return aresult['response']


async def generate_answer(question: str, documents: list[Document], history: list = None) -> list[Document]:
    """
    Generate the final answer of the agent in a question-answering cycle.
    """
    if history is None:
        history = []

    print("===================")
    print(history)
    print("===================")

    prompt = (f'<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
              'You are an assistant tasked with answering user questions based on the provided context. '
              'Use the following retrieved information to generate a concise, plain-text response. '
              'If you do not know the answer, simply state that you do not know. '
              'Limit your response to a maximum of six sentences. '
              'If previous conversation history exists, reference it in your answer. If there is just user question, you must ignore it. '
              '<|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Question: {question}. \n\n'
              f'Context: {documents}. \n\n'
              f'History of previous conversations (if available): {history} \n\n'
              'Answer: '
              '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=c.ll_model,
        prompt=prompt,
        # format="json",
        # options=opt,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    print(f"Eval_duration of answer generation: {aresult['eval_duration'] / 1_000_000_000}")
    #
    print("aresult['response'] type is: ", type(aresult))
    print("aresult['response']: ", aresult['response'])
    return aresult['response']
