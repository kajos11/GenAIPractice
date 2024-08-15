#Streamlit is a framework for building interactive web application

import streamlit as st
import os 

#FAISS: library by facebook AI research for efficient similarity seatch and clustering of large-scale datasets
#Provides optimized indexing structures and algorithms for tasks like nearest neighbors search and recommendation systems
from langchain_community.vectorstores import FAISS


#to load .env file
from dotenv import load_dotenv

#New import from langchain, which replaces the above
from langchain_openai import OpenAIEmbeddings

load_dotenv()

 
st.set_page_config(page_title="", page_icon=":robot:")

st.header("Ask anything will fetch similar items/things")

embeddings = OpenAIEmbeddings()

from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='myData.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Words']
})

csv_data = loader.load()

print(f"csv_data {csv_data}")

#create vector representation and store using embedding
db = FAISS.from_documents(csv_data, embeddings)

def get_text():
    input_text = st.text_input("You: ", key= input)
    return input_text

user_inp = get_text()
submit = st.button("Get similar things")

if submit:
    #If the button is clicked, the below snippet will fetch us the similar text
    docs = db.similarity_search(user_inp)
    st.subheader("Top Matches:")
    #st.text(docs[0])
    st.text(docs[0].page_content)



