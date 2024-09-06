from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

import dotenv


dotenv.load_dotenv()

chat = ChatOpenAI(model="gpt-3.5-turbo-0125")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

chain = prompt | chat

demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="input",  # specifies which part of the input should be tracked and stored in the chat history. In this example, we want to track the string passed in as input.
    history_messages_key="chat_history",  # specifies what the previous messages should be injected into the prompt as. Our prompt has a MessagesPlaceholder named chat_history, so we specify this property to match.
)


chain_with_message_history.invoke(
    {"input": "Translate this sentence from English to French: I love programming."},
    {"configurable": {"session_id": "unused"}},
)

def run_chat():
    print("Hello, I am your friendly chatbot. Let's chat!")
    print("Type 'STOP' to end the conversation, 'HISTORY' to view chat history, or 'CLEAR' to clear the chat history.")
    while True:
        user_input = input('User: ')
        if user_input.strip().upper() == 'STOP':
            print('ChatGPT: Goodbye! It was a pleasure chatting with you.')
            break

        user_inp = {'input': user_input}
        response = chain_with_message_history.invoke(
            user_inp,
            {"configurable": {"session_id": "unused"}},
        )
        print('ChatGPT:', response.content)

if __name__ == '__main__':
    run_chat()
