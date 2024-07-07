import os


from langchain_openai import ChatOpenAI

from langchain.schema import HumanMessage, SystemMessage, AIMessage

chat = ChatOpenAI(temperature=.7, model='gpt-3.5-turbo')

print(chat(
    [
        SystemMessage(content="You are a sarcastic AI assitant"),
        HumanMessage(content="Please answer in 30 words: How to ride a helicopter?")
    ]
).content) 


