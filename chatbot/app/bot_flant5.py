from langchain import HuggingFaceHub, LLMChain
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI


def get_llm():
    llm = ChatOpenAI(temperature=0,)
    return llm


def run_chain():
    template = """Question: {question}

    Answer: """
    prompt = PromptTemplate(
            template=template,
        input_variables=['question']
    )

    # user question
    question = "Which NFL team won the Super Bowl in the 2010 season?"

    # initialize Hub LLM
    hub_llm = HuggingFaceHub(
            repo_id='google/flan-t5-xl',
        model_kwargs={'temperature':1e-10}
    )

    # create prompt template > LLM chain
    llm_chain = LLMChain(
        prompt=prompt,
        llm=hub_llm
    )

    # ask the user question about NFL 2010
    print(llm_chain.run(question))