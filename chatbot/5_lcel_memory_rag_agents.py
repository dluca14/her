import bs4
import dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


dotenv.load_dotenv()
# ----------------------------------------------------------------------------------------------------------------------
memory = MemorySaver()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# ----------------------------------------------------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
# ----------------------------------------------------------------------------------------------------------------------
'''
Agents leverage the reasoning capabilities of LLMs to make decisions during execution. Using agents allow you to offload 
some discretion over the retrieval process. Although their behavior is less predictable than chains, they offer some 
advantages in this context:

Agents generate the input to the retriever directly, without necessarily needing us to explicitly build in 
contextualization, as we did above;
Agents can execute multiple retrieval steps in service of a query, or refrain from executing a retrieval step altogether 
(e.g., in response to a generic greeting from a user).
'''
### Build retriever tool ###
tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)
tools = [tool]
agent_executor = create_react_agent(llm, tools, checkpointer=memory)
# ----------------------------------------------------------------------------------------------------------------------
'''
Let's observe its behavior. Note that if we input a query that does not require a retrieval step, the agent does not 
execute one:
'''
config = {"configurable": {"thread_id": "abc123"}}
agent_executor.invoke({"messages": [HumanMessage(content="Hi! I'm Bob")]}, config=config)
response = agent_executor.invoke({"messages": [HumanMessage(content="What is my name?")]}, config=config)
for message in response['messages']:
    print(message)

''' compared to this one'''
print(f'-' * 50)
query = "What is Task Decomposition?"
response = agent_executor.invoke({"messages": [HumanMessage(content=query)]}, config=config)
for message in response['messages']:
    print(message)
# print(response["messages"][-1].content)
# for s in agent_executor.stream(
#     {"messages": [HumanMessage(content=query)]}, config=config,
# ):
#     print(s)
#     print("----")
