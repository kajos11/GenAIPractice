from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI

import os

#program to fetch info from chat gpt via api

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

our_query = "How many awards did sachin tendulkar win?"
print(llm.invoke(our_query))