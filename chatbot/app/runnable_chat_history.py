from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
            "You are a helpful assistant. Answer all questions to the best of your ability. The provided chat history includes facts about the user you are speaking with.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

chain = prompt | chat

demo_ephemeral_chat_history = ChatMessageHistory()


def summarize_messages(chain_input):
    stored_messages = demo_ephemeral_chat_history.messages
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"),
            (
                "user",
                "Distill the above chat messages into a single summary message. Include as many specific details as you can.",
            ),
        ]
    )
    summarization_chain = summarization_prompt | chat
    summary_message = summarization_chain.invoke({"chat_history": stored_messages})
    demo_ephemeral_chat_history.clear()
    demo_ephemeral_chat_history.add_message(summary_message)

    return True

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: demo_ephemeral_chat_history,
    input_messages_key="input",  # specifies which part of the input should be tracked and stored in the chat history. In this example, we want to track the string passed in as input.
    history_messages_key="chat_history",  # specifies what the previous messages should be injected into the prompt as. Our prompt has a MessagesPlaceholder named chat_history, so we specify this property to match.
)

chain_with_summarization = (
    RunnablePassthrough.assign(messages_summarized=summarize_messages)| chain_with_message_history
)
# chain_with_message_history.invoke(
#     {"input": "Translate this sentence from English to French: I love programming."},
#     {"configurable": {"session_id": "unused"}},
# )

def run_chat():
    print("Hello, I am your friendly chatbot. Let's chat!")
    print("Type 'STOP' to end the conversation, 'HISTORY' to view chat history, or 'CLEAR' to clear the chat history.")
    while True:
        user_input = input('User: ')
        if user_input.strip().upper() == 'STOP':
            print('ChatGPT: Goodbye! It was a pleasure chatting with you.')
            break

        user_inp = {'input': user_input}
        response = chain_with_summarization.invoke(
            user_inp,
            {"configurable": {"session_id": "unused"}},
        )
        print('ChatGPT:', response.content)


if __name__ == '__main__':
    run_chat()
