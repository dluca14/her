import bs4
import dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
prompt = ChatPromptTemplate.from_template('''
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
''')
# ----------------------------------------------------------------------------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
'''
We’ll use the LCEL Runnable protocol to define the chain, allowing us to

pipe together components and functions in a transparent way
automatically trace our chain in LangSmith
get streaming, async, and batched calling out of the box.
'''
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# ----------------------------------------------------------------------------------------------------------------------

for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)
    
'''
Let's dissect the LCEL to understand what's going on.

First: each of these components (retriever, prompt, llm, etc.) are instances of Runnable. This means that they implement the same methods-- such as sync and async .invoke, .stream, or .batch-- which makes them easier to connect together. They can be connected into a RunnableSequence-- another Runnable-- via the | operator.

LangChain will automatically cast certain objects to runnables when met with the | operator. Here, format_docs is cast to a RunnableLambda, and the dict with "context" and "question" is cast to a RunnableParallel. The details are less important than the bigger point, which is that each object is a Runnable.

Let's trace how the input question flows through the above runnables.

As we've seen above, the input to prompt is expected to be a dict with keys "context" and "question". So the first element of this chain builds runnables that will calculate both of these from the input question:

retriever | format_docs passes the question through the retriever, generating Document objects, and then to format_docs to generate strings;
RunnablePassthrough() passes through the input question unchanged.
That is, if you constructed:
------------------------------------
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
)
print(chain.invoke("What is Task Decomposition?"))
------------------------------------
Then chain.invoke(question) would build a formatted prompt, ready for inference. (Note: when developing with LCEL, it can be practical to test with sub-chains like this.)

The last steps of the chain are llm, which runs the inference, and StrOutputParser(), which just plucks the string content out of the LLM's output message.
'''



