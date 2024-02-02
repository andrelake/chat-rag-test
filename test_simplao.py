import os

from langchain_community.document_loaders import DirectoryLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = ""

prompt = "Give me the sum of the amounts"
loader = DirectoryLoader(".", glob="*.txt")
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(prompt, llm=ChatOpenAI()))
