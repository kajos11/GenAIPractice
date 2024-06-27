from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI

import os

#program to fetch info from chat gpt via api

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
our_query = "How many awards did FooFoo win?"
print("without pdf chaining, answer:")
print(llm.invoke(our_query))
data = PdfReader("chainExample.pdf")
combined_text = ''
for i,page in enumerate(data.pages):
    text = page.extract_text()
    if text:
        combined_text += text
#Step: break pdf data into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap = 20,
    length_function = len,
)
finalData = text_splitter.split_text(combined_text)
len(finalData)
#Step: generate text embeddings and storing them in vector store
embeddings = OpenAIEmbeddings()
documentSearch = FAISS.from_texts(finalData, embeddings)
#Step: fetching answers to user query
chain = load_qa_chain(OpenAI(), chain_type="stuff")
docs = documentSearch.similarity_search(our_query)
response = chain.invoke({"input_documents":docs, "question":our_query}, return_only_outputs=True)
print("=========================================================================================================")
print("with pdf chaining, answer:")
print(response["output_text"])
