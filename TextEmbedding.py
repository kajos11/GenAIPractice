import os

#reduce the dimentionality and predict similar words, help find semantics
#ML models are not capable of char, numeric is used to feed

from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = ""


embeddings = OpenAIEmbeddings()


import pandas as pd

dfs = pd.read_excel("TestEmbeddings.xlsx")
#print(dfs)

dfs["embedding"]=dfs["Words"].apply(lambda x: embeddings.embed_query(x))
dfs.to_csv("word_embeddings.csv")
#print(dfs)

new_dfs = pd.read_csv("word_embeddings.csv")
#print(new_dfs)

#get embeddings for our text
our_text = "Mango"

text_embedding = embeddings.embed_query(our_text)

#print(f"our embedding is: {text_embedding}")

# now get the simalarity between two words by using COSINE SIMILARITY
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

dfs["similarity_score"] = dfs['embedding'].apply(lambda x: cosine_similarity(x, text_embedding))

print (dfs)




