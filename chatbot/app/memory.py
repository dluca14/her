import dotenv
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.memory import ConversationSummaryMemory


dotenv.load_dotenv()

REDIS_URL = 'redis://localhost:6379/0'

llm = ChatOpenAI(model="gpt-3.5-turbo")

def get_session_history_summarized(session_id) -> RedisChatMessageHistory:
    session_history = RedisChatMessageHistory(session_id, url=REDIS_URL)
    stored_messages = session_history.messages
    if len(stored_messages) != 0:
        summarization_prompt = ChatPromptTemplate.from_messages(
            [
                ("placeholder", "{chat_history}"),
                (
                    "user",
                    "Distill the above chat messages into a single summary message."
                    "Do it in such a way that the summary includes names or places mentioned in the chat in a clear format."
                    "Include as many specific details as you can.",
                ),
            ]
        )
        summarization_chain = summarization_prompt | llm
        summary_message = summarization_chain.invoke({"chat_history": stored_messages})
        session_history.clear()
        session_history.add_message(summary_message)

    return session_history


def get_session_history_summarized_langchain(session_id) -> RedisChatMessageHistory:
    """ TODO: Implement this function to use langchain memory """
    session_history = RedisChatMessageHistory(session_id, url=REDIS_URL)
    stored_messages = session_history.messages

    summary_message = ConversationSummaryBufferMemory(llm=llm, max_token_limit=40).predict(
        input=stored_messages
    )

    session_history.clear()
    session_history.add_message(summary_message.chat_memory.messages)
    return session_history