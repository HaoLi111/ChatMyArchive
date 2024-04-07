# %%
from utils import *

# %%
from langchain.chains import RetrievalQA



def search_func_vdb(query, docs, vdb, llm):
    return vdb.similarity_search(query)


# %%
# from local_light_llm import llm


# from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
# agent_executor = initialize_agent(tools, llm, 
#         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
#         handle_parsing_errors=True,
#         # memory = memory, 
#         )
from functools import partial

from utils import search_metadata_util2
def make_neighbor_search_tool_func(vdb,all_pdf_docs):
    search_tool_func = partial(
        search_metadata_util2,
        docs=all_pdf_docs,
        search_func=vdb.similarity_search,
        # search_args={'k': 6},
        n_args= {'n':64, 'overlap_length':32, 'key':'in_file'},
    )
    return search_tool_func

def make_search_agent_executor(vdb, tools,llm):
    return initialize_agent(tools, llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
        handle_parsing_errors=True,
        # memory = memory, 
        )

# # Create a partial function with the search_metadata_util and the desired arguments
# search_tool_func = partial(
#     search_metadata_util2,
#     docs=all_pdf_docs,
#     search_func=vdb.similarity_search,
#     # search_args={'k': 6},
#     n_args= {'n':64, 'overlap_length':32, 'key':'in_file'},
# )
# Create a Langchain Tool with the search function and a description
# search_tool = Tool(
#     name='case search',
#     func=search_tool_func,
#     description="""
#     Search from source documents.
#     """
# )# %%

# from langchain_openai import OpenAI

# prompt = hub.pull("hwchase17/react")

# # Construct the ReAct agent
# agent = create_react_agent(llm, tools, prompt)
# agent_executor = AgentExecutor.from_agent_and_tools(agent = agent, tools=tools, handle_parsing_errors=True)

# # %%


# tools = [search_tool]


# %%


# %%




# %%
# result = agent_executor.invoke({"input": question[0]})

# %%

# qa_chain.invoke(question)

# # %%
# qa_chain.invoke(questions[2])

# # %%
# llm.invoke("how to find the final decision from a number of legal files for a case")

# # %%
# llm.invoke("How to search among law documents")

# # %%
# result["intermediate_steps"]


# # %%
# result['final_answer']

# # %%
# result

# # %%
# question

# # %%


# # %%
# list_files(directory_path)

# %%
# len(all_pdf_docs)

# %%
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def create_question_answering_chain(llm):


    # Define the prompt template
    template = """
    Context: {context}

    Question: {question}

    Answer: Let's break this down:
    """

    # Create a prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # Create an LLMChain using the prompt and OpenAI LLM
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain

def answer_question(chain, context, question):
    # Run the LLMChain to generate the answer
    result = chain.run(context=context, question=question)

    # Check if a final answer is found in the result
    if "Final Answer:" in result:
        return result
    else:
        return result + "\n\nFinal Answer: answer not found"

def extract_final_answer(result):
    if "Final Answer:" in result and ('not found' not in result):
        # Extract the final answer using string splitting
        final_answer = result.split("Final Answer:")[1].strip()
        return final_answer
    else:
        return 'Not_solved'



# Create the question-answering chain
# chain = create_question_answering_chain(llm)

# # Answer the question
# result = answer_question(chain, context, question)
# print(result)

import pandas as pd



# chain = create_question_answering_chain(llm)

        
def collect_all_QAs(docs, query):
    doc_content = []
    doc_ids = []
    answers = []
    final_answers = []

    for i, doc in enumerate(docs):
        print("working on ",i)
        doc_content.append(doc.page_content)
        doc_ids.append(i)

        # chain = create_question_answering_chain(llm)
        context = doc.page_content
        # Answer the question
        result = answer_question(chain, context, query)
        answers.append(result)

        # Extract the final answer
        final_answer = extract_final_answer(result)
        final_answers.append(final_answer)

    # Create a DataFrame with the collected vectors
    df = pd.DataFrame({
        'doc_id': doc_ids,
        'doc_content': doc_content,
        'answer': answers,
        'final_answer': final_answers
    })

    return df
    


# dfr=collect_all_QAs(all_pdf_docs[:3],questions[0])

# # %%
# dfr.doc_content[0]

# %%



