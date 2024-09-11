import bs4
import dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

'''Let’s put it all together into a chain that takes a question, retrieves relevant documents, constructs a prompt, 
passes that to a model, and parses the output.'''

# ----------------------------------------------------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini")
# ----------------------------------------------------------------------------------------------------------------------
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
# ----------------------------------------------------------------------------------------------------------------------
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
# ----------------------------------------------------------------------------------------------------------------------
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
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
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
# ----------------------------------------------------------------------------------------------------------------------
'''
LangChain includes convenience functions that implement the above LCEL. We compose two functions:
 - create_stuff_documents_chain specifies how retrieved context is fed into a prompt and LLM. In this case, we will "stuff" 
the contents into the prompt -- i.e., we will include all retrieved context without any summarization or other processing. 
It largely implements our above rag_chain, with input keys context and input-- it generates an answer using retrieved 
context and query.
 - create_retrieval_chain adds the retrieval step and propagates the retrieved context through the chain, 
providing it alongside the final answer. It has input key input, and includes input, context, and answer in its output.
'''
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# ----------------------------------------------------------------------------------------------------------------------
response = rag_chain.invoke({"input": "What is Task Decomposition?"})
print(f'-' * 50)
print(f'QUESTION: {response["input"]}')
print(f'-' * 50)
print(f'CONTEXT: {response['context']}')
print(f'-' * 50)
print(f'ANSWER: {response["answer"]}')
print(f'-' * 50)
'''
Returning sources
Often in Q&A applications it's important to show users the sources that were used to generate the answer. LangChain's 
built-in create_retrieval_chain will propagate retrieved source documents through to the output in the "context" key
for document in response["context"]:
    print(document)
    print()
'''