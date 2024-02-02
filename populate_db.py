import chromadb
from chromadb.config import Settings
import uuid

# cria client
client = chromadb.Client(settings=Settings(
    chroma_server_host="localhost",
    chroma_server_http_port="8000"
))
#
# # cria collection
collection = client.create_collection('transactions')

# # persiste o conteúdo do txt
with open("texto.txt", "r") as f:
    lines = f.readlines()

for line in lines:
    transaction_id = str(uuid.uuid4())
    collection.add(documents=[line], ids=[transaction_id])

# printa os registros
records = collection.get()
# for record in records:
print(records.get('documents'))
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
