import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
# from dotevn import load_dotenv
# load_dotenv()

# load the groq api key
# groq_api_key = os.getenv('GROQ_API_KEY')
groq_api_key = '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'
# 
st.title('ChatGroq with DeepSeek R1')

llm = ChatGroq(groq_api_key=groq_api_key,
               model_name='deepseek-r1-distill-llama-70b')

prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context.
Please provide the most accurate respone based on the question
<context>
{context}
<context>
Question:{input}
"""
)

def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('../data/') # data ingestion
        st.session_state.docs = st.session_state.loader.load() # document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #chunk creation
        st.session_state.chunks = st.session_state.text_splitter.split_documents(st.session_state.docs) # splitting
        st.session_state.vectors = Chroma.from_documents(st.session_state.chunks[:20], st.session_state.embeddings) # vector embeddings    


prompt1 = st.text_input('Enter Your Question from Documetns')

if st.button('Read Documents'):
    vector_embedding()
    st.write('Vector Store DB is Ready')

if prompt1:
    start = time.process_time()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input':prompt1})
    timetaken = time.process_time()-start
    print('Respones time :', timetaken)
    st.write(response['answer'])

    # with streamlit expander
    with st.expander('Document Similarity Search'):
        st.write('Response Time : ', timetaken)
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('-----------X-----------')