import bs4
import dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder


dotenv.load_dotenv()
# ----------------------------------------------------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini")
# ----------------------------------------------------------------------------------------------------------------------
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
# ----------------------------------------------------------------------------------------------------------------------
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_system_prompt = (
    "Given a question retrieve relevant documents in order to answer it. "
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        # MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# -------------------------create_history_aware_retriever -> just returns relevant docs --------------------------------
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
# response = history_aware_retriever.invoke(
#     {"input": "What is Task Decomposition?"},
#     config={"configurable": {"session_id": "abc123"}},
# )
# print(response)
# ----------------------------------------------------------------------------------------------------------------------
'''
This chain prepends a rephrasing of the input query to our retriever, so that the retrieval incorporates the context of
the conversation.
'''
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
'''
Now we can build our full QA chain.

As in the RAG tutorial, we will use create_stuff_documents_chain to generate a question_answer_chain, with input keys
context, chat_history, and input-- it accepts the retrieved context alongside the conversation history and query to
generate an answer.

We build our final rag_chain with create_retrieval_chain. This chain applies the history_aware_retriever and
question_answer_chain in sequence, retaining intermediate outputs such as the retrieved context for convenience.
It has input keys input and chat_history, and includes input, chat_history, context, and answer in its output.
'''
# ----------------------------------------------------------------------------------------------------------------------
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# -------------------------------------------create_retrieval_chain-----------------------------------------------------
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
# ----------------------------------------------------------------------------------------------------------------------
'''
----Adding chat history----
To manage the chat history, we will need:

An object for storing the chat history;
An object that wraps our chain and manages updates to the chat history.
For these we will use BaseChatMessageHistory and RunnableWithMessageHistory. The latter is a wrapper for an LCEL chain 
and a BaseChatMessageHistory that handles injecting chat history into inputs and updating it after each invocation.

For a detailed walkthrough of how to use these classes together to create a stateful conversational chain, head to the 
How to add message history (memory) LCEL how-to guide.
https://python.langchain.com/v0.2/docs/how_to/message_history/

Below, we implement a simple example of the second option, in which chat histories are stored in a simple dict. 
LangChain manages memory integrations with Redis and other technologies to provide for more robust persistence.

Instances of RunnableWithMessageHistory manage the chat history for you. They accept a config with a key 
("session_id" by default) that specifies what conversation history to fetch and prepend to the input, and append the 
output to the same conversation history. Below is an example:
'''
# ----------------------------------------------------------------------------------------------------------------------
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
# ----------------------------------------------------------------------------------------------------------------------
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
# ------------------------------------------run step by step------------------------------------------------------------
# context = history_aware_retriever.invoke(
#     {"input": "What is my name?"},
#     config={"configurable": {"session_id": "abc123"}},
# )
# print(context)
# ''' return relevant docs '''
# response = question_answer_chain.invoke(
#     {"input": "What is my name?", "chat_history": ["Hi, my name is David", "Nice to meet you"], "context": context}
# )
# print(response)
# ''' return answer '''
# ----------------------------------------------run entire chain--------------------------------------------------------
response = conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)
print(f'-' * 50)
print(f'QUESTION: {response["input"]}')
print(f'-' * 50)
print(f'CONTEXT: {response["context"]}')
print(f'-' * 50)
print(f'ANSWER: {response["answer"]}')
print(f'-' * 50)