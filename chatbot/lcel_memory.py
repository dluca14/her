import dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
# ----------------------------------------------------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini")
# ----------------------------------------------------------------------------------------------------------------------
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an assistant who speaks in {language}. Respond in 20 words or fewer",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
# ----------------------------------------------------------------------------------------------------------------------
runnable = prompt | llm
# ----------------------------------------------------------------------------------------------------------------------
runnable_with_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    output_messages_key="output",
    history_messages_key="history",
)
# ----------------------------------------------------------------------------------------------------------------------
runnable_with_history.invoke(
    {"language": "romana", "input": "hi im bob!"},
    config={"configurable": {"session_id": "2"}},
)
response = runnable_with_history.invoke(
    {"language": "romana", "input": "whats my name?"},
    config={"configurable": {"session_id": "2"}},
)
print(response)