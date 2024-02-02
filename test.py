import os
import time

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Annoy
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "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"

loader = DirectoryLoader(".", glob="*.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=85, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = Annoy.from_documents(texts, embeddings)

template = """You are a bot that answers questions about transactions, using only the context provided.
If you don't know the answer, simply state that you don't know.

{context}

Question: {question}"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

llm = ChatOpenAI(
    temperature=0.0,
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    output_key='answer',
    memory_key='chat_history',
    return_messages=True
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4, "include_metadata": True}
)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    get_chat_history=lambda h: h,
    verbose=False)

# qa_with_source = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     memory_key="chat_history",
#     chain_type_kwargs={"prompt": PROMPT},
#     return_source_documents=True,
# )


def timer(callback, seconds):
    time.sleep(seconds)
    callback()


def first_prompt():
    response = chain({"question": "Sum the amount of transactions"})
    print(f"{response['chat_history']}\n")

def second_prompt():
    response = chain({"question": "Explain me how do you made this sum step by step"})
    print(f"{response['chat_history']}\n")


def third_prompt():
    response = chain({"question": "Get the most recent transaction"})
    print(f"{response['chat_history']}\n")


def forth_prompt():
    response = chain({"question": "Remember my latest 2 questions"})
    print(f"{response['chat_history']}\n")


timer(first_prompt, 3)
timer(second_prompt, 5)
timer(third_prompt, 5)
# client = chromadb.PersistentClient(path="/db")
#
# if(client.get_collection("transactions_docs")== null) :
# collection = client.create_collection("transactions_docs")
# with open("texto.txt", "r") as f:
#     lines = f.readlines()
#
# for line in lines:
#     transaction_id = str(uuid.uuid4())
#     collection.add(documents=[line], ids=[transaction_id])
#
# f.close()

# loader = DirectoryLoader(".", glob="*.txt")
# collection_resp = client.get_collection("transactions_docs")
# get_results = collection_resp.get()
# print(json.dumps(get_results, indent=4))
# documents = json.dumps(get_results, indent=4)
# print(documents)
# index = VectorstoreIndexCreator().from_orm(documents=[documents])
# #
# print(index.query(prompt, llm=ChatOpenAI()))
