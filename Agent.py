import os
import Tools
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import chain, RunnablePassthrough
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    """You are a knowledgeable assistant tasked with answering questions. Use only the provided context to generate the most accurate and relevant answer. If the answer cannot be answered by the provided context, say that you do not know.
        
        Context: {context}

        Question: {question}
    """
    )

vectorstore = Chroma.from_documents(
    Tools.text_splitter("samplearticle.txt"),
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model="gpt-3.5-turbo")

@chain
def construct_prompt(passthrough_object):
    context = passthrough_object.get("context")
    question = passthrough_object.get("question")
    prompt = prompt_template.format(context=context, question=question)
    return prompt

chain = {"context": retriever, "question": RunnablePassthrough()} | construct_prompt | llm | StrOutputParser()

def run():
    while (True):
        query = input("Enter question here: ")
        if query == 'exit':
            break
        result = chain.invoke(query)
        print(result)

run()