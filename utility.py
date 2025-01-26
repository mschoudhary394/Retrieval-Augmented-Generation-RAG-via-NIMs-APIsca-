from langchain.chains import ConversationalRetrievalChain, LLMChain

from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT

from langchain.chains.question_answering import load_qa_chain

from langchain.memory import ConversationBufferMemory

#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Union
from typing import List, Union
       
'''
def html_document_loader(url: Union[str, bytes]) -> str:
    """
    Loads the HTML content of a document from a given URL and return it's content.

    Args:
        url: The URL of the document.

    Returns:
        The content of the document.

    Raises:
        Exception: If there is an error while making the HTTP request.

    """
    try:
        
        response = requests.get(url)
        html_content = response.text
    except Exception as e:
        print(f"Failed to load {url} due to exception {e}")
        return ""

    try:
        # Create a Beautiful Soup object to parse html
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style tags
        for script in soup(["script", "style"]):
            script.extract()

        # Get the plain text from the HTML document
        text = soup.get_text()

        # Remove excess whitespace and newlines
        text = re.sub("\s+", " ", text).strip()

        return text
    except Exception as e:
        print(f"Exception {e} while loading document")
        return ""


def index_docs(embeddings_model,url: Union[str, bytes], splitter, documents: List[str], dest_embed_dir: str) -> None:
    """
    Split the documents into chunks and create embeddings for them.
    
    Args:
        embeddings_model: Model used for creating embeddings.
        url: Source url for the documents.
        splitter: Splitter used to split the documents.
        documents: List of documents whose embeddings need to be created.
        dest_embed_dir: Destination directory for embeddings.
    """
    texts = []
    metadatas = []

    for document in documents:
        chunk_texts = splitter.split_text(document.page_content)
        texts.extend(chunk_texts)
        metadatas.extend([document.metadata] * len(chunk_texts))
    
    if os.path.exists(dest_embed_dir):
        docsearch = FAISS.load_local(
            folder_path=dest_embed_dir, 
            embeddings=embeddings_model, 
            allow_dangerous_deserialization=True
        )
        docsearch.add_texts(texts, metadatas=metadatas)
    else:
        docsearch = FAISS.from_texts(texts, embedding=embeddings_model, metadatas=metadatas)

    docsearch.save_local(folder_path=dest_embed_dir)


def create_embeddings(embedding_path: str = "./embed"):

    embedding_path = "./embed"
    print(f"Storing embeddings to {embedding_path}")
    urls = [
        "https://docs.nvidia.com/cuda/",
        "https://github.com/NVIDIA/cuda-samples", "https://github.com/openhackathonsorg/nways_accelerated_programming/blob/main/_basic/cuda/jupyter_notebook/nways_cuda.ipynb"
        "https://github.com/openhackathons-org/nways_multi_gpu/tree/main",
        "https://reference.wolfram.com/language/CUDALink/tutorial/Programming.html",
        "https://docs.python.org/3/reference/datamodel.html"
       ]
    documents = []
    for url in urls:
        document = html_document_loader(url)
        documents.append(document)

    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
    )
    embeddings_model = NVIDIAEmbeddings(model="NV-Embed-QA")
   # print("Total documents:",len(documents))
    texts = text_splitter.create_documents(documents)
   # print("Total texts:",len(texts))
    index_docs(embeddings_model,url, text_splitter, texts, embedding_path,)
   # print("Generated embedding successfully")
'''

import random
import socket
def find_available_port(start=9000, end=9999):
    while True:
        # Randomly select a port between start and end range
        port = random.randint(start, end)
        
        # Try to create a socket and bind to the port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("localhost", port))
                # If binding is successful, the port is free
                return port
            except OSError:
                # If binding fails, the port is in use, continue to the next iteration
                continue

def index_docs(embeddings_model, url: Union[str, bytes], splitter, documents: List[str], dest_embed_dir: str) -> None:
    """
    Split the documents into chunks and create embeddings for them.
    
    Args:
        embeddings_model: Model used for creating embeddings.
        url: Source url for the documents.
        splitter: Splitter used to split the documents.
        documents: List of documents whose embeddings need to be created.
        dest_embed_dir: Destination directory for embeddings.
    """
    texts = []
    metadatas = []

    for document in documents:
        chunk_texts = splitter.split_text(document.page_content)
        texts.extend(chunk_texts)
        metadatas.extend([document.metadata] * len(chunk_texts))

    if os.path.exists(dest_embed_dir):
        docsearch = FAISS.load_local(
            folder_path=dest_embed_dir, 
            embeddings=embeddings_model, 
            allow_dangerous_deserialization=True
        )
        docsearch.add_texts(texts, metadatas=metadatas)
    else:
        docsearch = FAISS.from_texts(texts, embedding=embeddings_model, metadatas=metadatas)

    docsearch.save_local(folder_path=dest_embed_dir)

def html_document_loader(url: Union[str, bytes]) -> str:
    """
    Loads the HTML content of a document from a given URL and return it's content.

    Args:
        url: The URL of the document.

    Returns:
        The content of the document.

    Raises:
        Exception: If there is an error while making the HTTP request.

    """
    try:
        response = requests.get(url)
        html_content = response.text
    except Exception as e:
        print(f"Failed to load {url} due to exception {e}")
        return ""

    try:
        response = requests.get(url)
        html_content = response.text
    except Exception as e:
        print(f"Failed to load {url} due to exception {e}")
        return ""

    try:
        # Create a Beautiful Soup object to parse html
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style tags
        for script in soup(["script", "style"]):
            script.extract()

        # Get the plain text from the HTML document
        text = soup.get_text()

        # Remove excess whitespace and newlines
        text = re.sub("\s+", " ", text).strip()

        return text
    except Exception as e:
        print(f"Exception {e} while loading document")
        return ""


def create_embeddings(embeddings_model,embedding_path: str = "./embed"):

    embedding_path = "./embed"
    #print(f"Storing embeddings to {embedding_path}")

    documents = []
    urls = [
        "https://docs.nvidia.com/cuda/","https://docs.nvidia.com/cuda/cuda-c-programming-guid","https://github.com/NVIDIA/cuda-samples","https://github.com/openhackathons-org/nways_accelerated_programming/blob/main/_basic/cuda/jupyter_notebook/nways_cuda.ipynb","https://github.com/openhackathons-org/nways_multi_gpu/tree/main","https://reference.wolfram.com/language/CUDALink/tutorial/Programming.html","https://docs.python.org/3/reference/datamodel.html"
       ]
    for url in urls:
        document = html_document_loader(url)
        documents.append(document)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
    )
    #print("Total documents:",len(documents))
    texts = text_splitter.create_documents(documents)
    #print("Total texts:",len(texts))
    index_docs(embeddings_model,url, text_splitter, texts, embedding_path,)
    #print("Generated embedding successfully")

def get_response(query):
    os.environ["NVIDIA_API_KEY"] = "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"
    embeddings_model = NVIDIAEmbeddings(model="NV-Embed-QA")
    create_embeddings(embeddings_model=embeddings_model)
    #print(os.environ.get('CONTAINER_PORT'))
    #load Embed documents
    embedding_path = "./embed/"
    docsearch = FAISS.load_local(folder_path=embedding_path, embeddings=embeddings_model, allow_dangerous_deserialization=True)
    # Find and print an available port
    PORT1 = str(find_available_port())
    #create_embeddings("./embed")
    #os.environ.get('CONTAINER_PORT')
    llm = ChatNVIDIA(base_url="http://0.0.0.0:{}/v1".format(9528),
                 model="codellama/codellama-34b-instruct", temperature=0.2, max_tokens=1200, top_p=1.0)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer", max_token_limit=1200)
    
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    #PORT2 = str(find_available_port())
    #chat = ChatNVIDIA(base_url="http://0.0.0.0:{}/v1".format(PORT2),
                     # model="mistralai/mistral-7b-instruct-v0.3", temperature=0.2, max_tokens=1200, top_p=1.0)
    
    #doc_chain = load_qa_chain(chat , chain_type="stuff", prompt=QA_PROMPT)
    doc_chain = load_qa_chain(llm , chain_type="stuff", prompt=QA_PROMPT)

    qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=docsearch.as_retriever(),
    chain_type="stuff",
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={'prompt': QA_PROMPT},
    )
    
    #result = qa({"question": query})
    result = qa.invoke({"question": query})
    answer = result.get('answer', result.get('output', ''))
    context = "\n".join([doc.page_content for doc in result['source_documents']])
    
    return answer, context

