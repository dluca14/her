import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from gradio.themes.builder_app import history
from langchain.chains.llm import LLMChain
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence, RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# bring in streamlit for UI dev
import streamlit as st

# secret
root_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(root_path)
load_dotenv(f'{root_path}/chatbot/.env', verbose=True, override=True)

from chatbot.app.bot import get_llm


''' run app with: >>> streamlit run test_streamlit.py'''

# -------------------------------------------LLM------------------------------------------------------

# setup credentials dict
creds = {
    'apikey': 'xxx',
    'url': 'https://us-south.ml.cloud.ibm.com'
}
# create llm using langchain
# bring in watsonx interface
# from watsonxlangchain import LangChainInterface
# llm = LangChainInterface(
#     credetials=creds,
#     model='meta-llama/llama-t5-base',
#     params={
#         'decoding_method': 'sample',
#         'max_new_tokens': 200,
#         'temperature': 0.5,
#     },
#     project_id='id'
# )
llm = get_llm()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at {ability}"),
    # MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | llm


# ----------------------------------------RAG---------------------------------------------------------

# this function a PDF of you choosing
@st.cache_resource
def load_pdf():
    # update pdf name here
    pdf_name = 'genai.pdf'
    loaders = [PyPDFLoader(pdf_name)]
    # create index - aka vector db - chromadb
    index = VectorstoreIndexCreator(
        embeddings=HuggingFaceEmbeddings(model_name='all_MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)
    # return the vector db
    return index
# loader on up
# index = load_pdf()

# create QA chain
# chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type='stuff',
#     retriever=index.vectorstore.as_retriever(),
#     input_key='question'
# )

# ---------------------------------------Streamlit----------------------------------------------------------

# set up the app title
st.title("History Chatbot")

# set up a session state to store the conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# display all historical messages
for message in st.session_state.messages:
    # display the message
    st.chat_message(message['role']).markdown(message['content'])

# build a prompt input template to display the prompts
prompt = st.chat_input("Enter a prompt here...")

# if the user hits enter then
if prompt:
    # display the user prompt
    st.chat_message("human").markdown(prompt)
    # store the user prompt in the session state
    st.session_state.messages.append({'role': 'human', 'content': prompt})

    # print('='*40, st.session_state.messages)
    messages = []
    for i in st.session_state.messages:
        if i['role'] == 'human':
            messages.append((i['role'], i['content']))
        elif i['role'] == 'ai':
            messages.append((i['role'], i['content']))

    chat_template = ChatPromptTemplate.from_messages(messages)
    # send the prompt to the llm together with the chat history.
    chain = chat_template | llm
    response = chain.invoke(
        {"user_input": prompt}
    ).content
    # send the prompt to the PDF QA chain
    # response = chain.run(prompt)

    # display the llm response
    st.chat_message("ai").markdown(response)
    # store the llm response in the session state
    st.session_state.messages.append({'role': 'ai', 'content': response})
