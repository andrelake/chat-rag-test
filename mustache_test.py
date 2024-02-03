import openai

with open('texto.txt', 'r') as arquivo:
    conteudo = arquivo.read()

def enviar_mensagem(mensagem, lista_mensagens=[]):

    lista_mensagens.append(
        {"role": "user", "content": mensagem}
    )

    resposta = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = lista_mensagens
    )

    return resposta["choices"][0]["message"]

lista_mensagens = [
    {"role": "system", "content": "Você é um assistente que responde em portugues, o usuário enviará um texto que representa algumas transacoes financeiras e fará algumas peguntas sobre elas."},
    {"role": "user", "content": conteudo}
]

while True:
    texto = input("Você: ")

    if texto == "sair":
        break
    else:
        resposta = enviar_mensagem(texto, lista_mensagens)
        lista_mensagens.append(resposta)
        print("Chatbot: ", resposta["content"])