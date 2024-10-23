import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

import config as c  # Here are all ip, llm names and other important things
import time
from ollama import AsyncClient, Options
from langchain_ollama import \
    ChatOllama  # Пробуем альтернативный способ указать температуру и использовать JSONOutputParser
import json_converter as j

ollama_aclient = AsyncClient(host=c.ollama_url)

# Пробуем еще раз опции добавить, авось не понизит скорость.
options = Options(temperature=0)

# Выбор llm
llm = c.ll_model_big


async def grade(question: str, document: str):
    """
    Grade the relevance of the retrieved document.
    """
    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
              'You are tasked with assessing the relevance of a retrieved document to a user’s question. '
              'If the document contains concepts or keywords closely related to the user’s question, consider it relevant. '
              'The assessment does not need to be overly strict—the aim is to filter out irrelevant documents, not to demand exact matches. '
              'Return a binary "yes" or "no" to indicate whether the document is relevant. '
              'Provide your answer as a JSON object with a single key "score" and no additional text. '
              'Example: {"score": "yes"} or {"score": "no"}.'
              '<|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Here is the retrieved document: \n\n{document} \n\n'
              f'Here is the user question: {question} \n\n'
              '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=llm,
        prompt=prompt,
        format="json",
        options=options,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")
    #
    json_result = j.str_to_json(aresult['response'])
    print("Module Check. Grade retrieved response: " + str(json_result))

    return json_result


async def hallucinations_checker(documents, generation):
    """Hallucination Grader"""

    # prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
    #           'You are a grader assessing whether an answer is grounded in / supported by a set of facts, which are in Russian. '
    #           'You must provide your answer as a JSON object with a single key "score". Do not provide any explanations, translations, reformulations, or additional information. '
    #           'If the answer is supported by the Russian text, return {"score": "yes"}. If it is not, return {"score": "no"}. '
    #           'Do not include any other content besides the JSON object. '
    #           '<|eot_id|><|start_header_id|>user<|end_header_id|> '
    #           f'Here is the user question: \n\n{question} \n\n'
    #           f'Here are the provided facts (use only Russian text for evaluation): \n\n{documents} \n\n'
    #           f'Here is the generated answer: \n\n{generation} \n\n'
    #           '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether '
              'an answer is grounded in / supported by a set of facts. Give a binary "yes" or "no" score to indicate '
              'whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a '
              'single key "score" and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>'
              'If the answer is supported by the set of facts, return {"score": "yes"}. If it is not, '
              'return {"score": "no"}.'
              'Here are the facts:'
              '\n ------- \n'
              f'{documents} '
              '\n ------- \n'
              f'Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=llm,
        prompt=prompt,
        format="json",
        options=options,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    # print('Предыдущий результат: 227.32 секунд (LTE, MSK)')
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")

    json_result = j.str_to_json(aresult['response'])
    print("Module Check. Hallucinations checker response: " + aresult['response'])
    print("Module Check. Hallucinations checker answer length JSON: " + str(len(json.dumps(json_result))))
    print("Module Check. Hallucinations checker answer length str: " + str(len(aresult['response'])))
    if len(json.dumps(json_result)) != 16:
        json_result = j.str_to_json('{"score": "no"}')

    return json_result
    # return aresult['response']


async def hallucinations_checker_v2(documents, generation):
    """Hallucination Grader"""
    loc_llm = ChatOllama(model=llm, format="json", temperature=0, )

    # Prompt
    prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | loc_llm | JsonOutputParser()
    hallucination_grader.invoke({"documents": documents, "generation": generation})


async def answer_grader(question: str, generation):
    """Answer Grader"""

    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
              'You are a grader assessing whether an answer is useful to resolve a question. Give a binary score "yes" or "no" to indicate whether the answer is useful to resolve a question. '
              'Provide the binary score as a JSON with a single key "score" and no preamble or explanation. '
              'Example: {"score": "yes"} or {"score": "no"}. '
              '<|eot_id|><|start_header_id|>user<|end_header_id|> '
              f'Here is the generated answer: \n\n{generation} \n\n'
              f'Here is the question: \n\n{question} \n\n'
              '<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=llm,
        prompt=prompt,
        format="json",
        options=options,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")

    json_result = j.str_to_json(aresult['response'])
    print("Module Check. Answer grader response: " + str(json_result))

    return json_result
    # return aresult['response']
