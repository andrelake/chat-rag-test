# -*- encoding: utf-8 -*-
import os
import numpy as np
import math
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Redis
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, MapReduceDocumentsChain, StuffDocumentsChain, \
    ReduceDocumentsChain
from chromadb.config import Settings

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt


from variables import OPENAI_KEY

import os


os.environ['OPENAI_API_KEY'] = OPENAI_KEY


def getnearpos(vector, value):
    idx = (np.abs(vector-value)).argmin()
    return idx


@retry(stop=stop_after_attempt(3))
def start(document_id, session_id):
    try:
        llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo-16k")
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

        client_settings = Settings(
            chroma_api_impl="rest",
            chroma_server_host="localhost",
            chroma_server_http_port="8000"
        )

        rds = Chroma(collection_name="transactions",
                     embedding_function=embeddings,
                     client_settings=client_settings)

        retriever = rds.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        template = """You are a data analyst and should keep your responses based on the context below and chat_history below to answer the question.
        If there is no context-related data for the question, simply say 'I don't have that answer,' do not attempt to invent an answer or
        look outside the context. Use a maximum of three sentences and keep the response as concise as possible. Answer only on the subject below:

        context: {context}

        chat_history: {chat_history}

        Question: {question}
        Helpful Answer in portuguese:"""

        prompt = PromptTemplate(
            template=template, input_variables=["context", "chat_history", "question"]
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            chain_type="stuff",
            verbose=True,
            combine_docs_chain_kwargs={'prompt': prompt}
        )

        question = "Faça um relatório sobre pontos inportantes das transações"

        result = qa_chain.run(question)
        print(result)

        return None
    except Exception as error:
        print(error)
        print(
            type(error).__name__,  # TypeError
            __file__,  # /tmp/example.py
            error.__traceback__.tb_lineno  # 2
        )


def is_previous_or_next(cluster_index_list, index):
    for i in cluster_index_list:
        previous = i - 1
        next_pos = i + 1
        if next_pos == index or previous == index:
            return True

    return False


if __name__ == '__main__':
    start('document-01', 'SESSION-ID_USER')


