
from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents.load_tools import get_all_tool_names
from langchain import ConversationChain
import os
import openai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap
from langchain.memory import ChatMessageHistory, ConversationSummaryBufferMemory, ConversationBufferMemory


# openai.api_key = "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"

# one time api key this is the only location it can be found
class LangModel:
    
	def __init__(self):
		os.environ["OPENAI_API_KEY"] = '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'
		load_dotenv(find_dotenv())
		self.llm     = OpenAI(model_name="text-davinci-003")
		self.history = ConversationBufferMemory()
        
	def get_model(self, message):
		conversation = ConversationChain(llm=self.llm, verbose=True, memory = self.history)
		output       = conversation.predict(input=message)
		print('output: ', output)
		# self.collect_memory(output, message)
		return output

		

                
# LangModel().get_model('do you miss 1979')
# os.environ['']
# # --------------------------------------------------------------
# # Prompt Templates: Manage prompts for LLMs
# # --------------------------------------------------------------

# prompt = PromptTemplate(
#     input_variables=["product"],
#     template="What is a good name for a company that makes {product}?",
# )

# prompt.format(product="Smart Apps using Large Language Models (LLMs)")

# # --------------------------------------------------------------
# # Chains: Combine LLMs and prompts in multi-step workflows
# # --------------------------------------------------------------

# llm = OpenAI()
# prompt = PromptTemplate(
#     input_variables=["product"],
#     template="What is a good name for a company that makes {product}?",
# )

# chain = LLMChain(llm=llm, prompt=prompt)
# print(chain.run("AI Chatbots for Dental Offices"))


# # --------------------------------------------------------------
# # Agents: Dynamically Call Chains Based on User Input
# # --------------------------------------------------------------


# llm = OpenAI()

# get_all_tool_names()
# tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
# agent = initialize_agent(
#     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# )

# # Now let's test it out!
# result = agent.run(
#     "In what year was python released and who is the original creator? Multiply the year by 3"
# )
# print(result)


# # --------------------------------------------------------------
# # Memory: Add State to Chains and Agents
# # --------------------------------------------------------------

# llm = OpenAI()
# conversation = ConversationChain(llm=llm, verbose=True)

# output = conversation.predict(input="Hi there!")
# print(output)

# output = conversation.predict(
#     input="I'm doing well! Just having a conversation with an AI."
# )
# print(output)
