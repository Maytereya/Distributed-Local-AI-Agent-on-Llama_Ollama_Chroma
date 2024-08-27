# Router

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import config as c  # Here are all ip, llm names and other important things


async def route(question: str):
    """"Routing Questions"""

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        user question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, 
        prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
        or 'vectorstore' based on the question. Return the JSON with a single key 'datasource' and 
        no preamble or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    question_router = prompt | c.llm | JsonOutputParser()
    return await question_router.ainvoke({"question": question})

# question = "llm agent memory"
# # docs = index.retriever.get_relevant_documents(question)
# docs = index.retriever.invoke(question)
# doc_txt = docs[1].page_content
# print(question_router.invoke({"question": question}))
