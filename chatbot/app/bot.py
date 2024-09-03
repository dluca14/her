from langchain import HuggingFaceHub, LLMChain
from langchain import PromptTemplate

import sys
from pathlib import Path

# Add the parent directory of 'configs' to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs import initialize


initialize(env_file_path='../.env')


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