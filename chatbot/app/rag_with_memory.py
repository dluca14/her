from itertools import chain

import bs4
import dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from memory import get_session_history_summarized


dotenv.load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")


### Construct retriever ###
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

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_system_prompt = """You're an assistant who's good at conversations. """

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context and the chat history to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    runnable=rag_chain,
    get_session_history=get_session_history_summarized,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def run_chat():
    print("Hello, I am your friendly chatbot. Let's chat!")
    print("Type 'STOP' to end the conversation, 'HISTORY' to view chat history, or 'CLEAR' to clear the chat history.")
    while True:
        user_input = input('User: ')
        if user_input.strip().upper() == 'HISTORY':
            print(get_session_history_summarized('session1').messages)
            continue
        elif user_input.strip().upper() == 'STOP':
            print('ChatGPT: Goodbye! It was a pleasure chatting with you.')
            break

        user_inp = {'input': user_input}
        response = conversational_rag_chain.invoke(
            input=user_inp,
            config={"configurable": {"user_id": "user1", "session_id": "session1"}},
        )
        print('ChatGPT:', response['answer'])
        # print('ChatGPT:', response['context'])


if __name__ == '__main__':
    ''' Somehow the chatbot doesn't seam to make use of the chat history. 
    Check how it's passed to the chain upwards.'''
    # run_chat()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're an assistant who's good at conversations.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm
    runnable = RunnableWithMessageHistory(
        chain,
        get_session_history_summarized,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    # response = runnable.invoke(
    #     input={"input": "My name is David and I live in Timisoara"},
    #     config={"configurable": {"user_id": "user1", "session_id": "session2"}},
    # )
    # print(response['answer'])

    response = conversational_rag_chain.invoke(
        input={"input": "What is my name?"},
        config={"configurable": {"user_id": "user1", "session_id": "session2"}},
    )
    print(response['answer'])




    # response = conversational_rag_chain.invoke(
    #     input = {"input": "What is the agent's goal in reinforcement learning?"},
    #     config = {"configurable": {"user_id": "user1", "session_id": "session1"}},
    # )
    # print(f'QUESTION: {response["input"]}')
    # print(f'CONTEXT: {response['context']}')
    # print(f'ANSWER: {response['answer']}')
