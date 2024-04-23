from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
from src.helper import *
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain_community.document_loaders import WebBaseLoader

site_url = [
    "https://prospexs.ai/"
     ]

#Load data

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(site_url)
documents = loader.load()
#print(documents)


#Split Text into Chunks
text_splitter=CharacterTextSplitter(separator='\n',
                                    chunk_size=500,
                                    chunk_overlap=50)
text_chunks=text_splitter.split_documents(documents)
print("text_chunks")
print(text_chunks)

#Load the Embedding Model
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                 model_kwargs={'device':'cpu'})




#Convert the Text Chunks into Embeddings and Create a FAISS Vector Store
vector_store=FAISS.from_documents(text_chunks, embeddings)



# Initialize the LLM
llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':128,
                          'temperature':0.01})
                          

'''
llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
            huggingfacehub_api_token="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"
        )
print('LLM model Loaded')
'''


print("llm")
print(llm)

qa_prompt=PromptTemplate(template=template, input_variables=['context', 'question'])


# Initialize the RetrievalQA module

chain = RetrievalQA.from_chain_type(llm=llm,
                                   chain_type='stuff',
                                   retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                   return_source_documents=False,
                                   chain_type_kwargs={'prompt': qa_prompt},
                                    )

#qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())


# Define the Streamlit app
def main():
    st.title("Prospexs Custom Website Streamlit App")
    
    # Input variables
    question = st.text_input("Question:", "Enter your question here...")
    
    # Button to generate answer
    if st.button("Get Answer"):
        if question.strip():
            print(f"Question: {question}")
            #answer = print(chain.run(question))
            #print(f"Answer: {answer}")
            answer = chain({'query':question})
            print("Response : ", answer["result"])
            st.write("Answer:", answer['result'])
        else:
            st.warning("Please provide  and question.")





if __name__ == "__main__":
    main()
