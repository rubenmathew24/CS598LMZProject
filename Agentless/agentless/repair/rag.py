import os
import logging
import sys
from typing import List, Optional

from time import sleep

import pathlib

# Configure basic logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document,
)

from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.readers.web import SimpleWebPageReader
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore


EMBEDDING_MODEL_NAME = "models/gemini-embedding-exp-03-07"
LLM_MODEL_NAME = "models/gemini-2.5-pro-preview-05-06"


rag_storage_path = pathlib.Path(__file__).parent.parent.parent.joinpath("rag_storage").resolve()
#print(rag_storage_path)

def setup_gemini_models(
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    llm_model_name: str = LLM_MODEL_NAME,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
):
    """
    Configures and sets the global Gemini LLM and embedding models for LlamaIndex.
    Also configures the text splitter.
    """
    logging.info(f"Setting up Gemini models: Embeddings ('{embedding_model_name}'), LLM ('{llm_model_name}')")
    try:
        Settings.llm = Gemini(model_name=llm_model_name)
        Settings.embed_model = GeminiEmbedding(model_name=embedding_model_name)
        Settings.text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # Settings.chunk_size = chunk_size # Alternative way to set global chunk size
        # Settings.chunk_overlap = chunk_overlap
        logging.info("Gemini models and text splitter configured in LlamaIndex Settings.")
    except Exception as e:
        logging.error(f"Error initializing Gemini models: {e}")
        logging.error("Ensure your GOOGLE_API_KEY is valid and has the Generative Language API enabled.")
        raise


def load_rag_index(persist_dir: str = rag_storage_path) -> VectorStoreIndex:
    """
    Loads a previously created LlamaIndex RAG index from disk.
    
    Args:
        persist_dir (str): Directory where the index persistence files are stored.
        
    Returns:
        VectorStoreIndex: The loaded LlamaIndex VectorStoreIndex object.
        
    Raises:
        FileNotFoundError: If the persistence directory doesn't exist.
        Exception: If there's an error loading the index.
    """
    logging.info(f"Loading existing index from {persist_dir}...")
    
    if not os.path.exists(persist_dir):
        logging.error(f"Persistence directory {persist_dir} does not exist")
        raise FileNotFoundError(f"No index found at {persist_dir}")

    setup_gemini_models()
    
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    
    return index


def create_rag_index(
    urls: List[str],
    persist_dir: str = rag_storage_path,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
) -> VectorStoreIndex:
    """
    Creates a new LlamaIndex RAG index from a list of web page URLs.

    Args:
        urls (List[str]): A list of URLs to scrape documentation from.
        persist_dir (str): Directory to save the index persistence files.
        chunk_size (int): The target size for text chunks (nodes).
        chunk_overlap (int): The overlap between consecutive text chunks.

    Returns:
        VectorStoreIndex: The created LlamaIndex VectorStoreIndex object.

    Raises:
        ValueError: If no documents are loaded or GOOGLE_API_KEY is not set.
    """
    # Ensure models are set up with appropriate chunking configuration
    setup_gemini_models(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    logging.info(f"Creating new RAG index. Will persist to: '{persist_dir}'")

    logging.info(f"Loading documents from URLs: {urls}")
    reader = SimpleWebPageReader(html_to_text=True)
    documents = []
    for url in urls:
        logging.info(f"Attempting to load: {url}")
        try:
            page_docs = reader.load_data([url])
            for doc in page_docs:
                doc.metadata = doc.metadata or {}
                doc.metadata['source_url'] = url
            documents.extend(page_docs)
            logging.info(f"Successfully loaded {len(page_docs)} document part(s) from: {url}")
        except Exception as e:
            logging.error(f"Failed to load URL {url}: {e}")

    if not documents:
        logging.error("No documents were successfully loaded. Cannot build index.")
        raise ValueError("Failed to load any documents from the provided URLs.")

    logging.info(f"Loaded a total of {len(documents)} document sections.")
    logging.info(f"Text splitter configured with chunk_size={Settings.text_splitter.chunk_size}, chunk_overlap={Settings.text_splitter.chunk_overlap}")

    logging.info("Creating new index with Gemini embeddings from all documents...")


    index = VectorStoreIndex.from_documents(
        documents,          # Pass the entire list of documents
        show_progress=True
        # You can optionally create and pass a storage_context here if needed
        # for specific configurations, but from_documents handles defaults well.
        # storage_context=StorageContext.from_defaults() # Usually not needed unless customizing stores
    )

    logging.info("Index creation complete.")

    logging.info(f"Persisting index to {persist_dir}...")
    os.makedirs(persist_dir, exist_ok=True)
    # Persist the storage context associated with the single created index
    index.storage_context.persist(persist_dir=persist_dir)
    logging.info("Index persisted successfully.")

    return index



def get_context_for_query(
    query_text: str,
    index: VectorStoreIndex,
    similarity_top_k: int = 5,
    context_separator: str = "\n\n---\n\n"
) -> str:
    """
    Retrieves relevant context from the index for a given query and returns it as a string.

    Args:
        query_text (str): The query to find context for.
        index (VectorStoreIndex): The LlamaIndex VectorStoreIndex to query.
        similarity_top_k (int): The number of top similar document chunks to retrieve.
                                Defaults to 5.
        context_separator (str): The string to use for separating retrieved context chunks.
                                 Defaults to a newline, horizontal rule, and newline.

    Returns:
        str: A single string containing the concatenated content of the
             retrieved document chunks, separated by `context_separator`.
             Returns an empty string if no context is found.
    """
    logging.info(f"Retrieving context for query: '{query_text}' (top_k={similarity_top_k})")
    # Ensure models are set up (especially embed_model for the retriever)
    # This might be redundant if create_or_load_rag_index was called, but safe.
    setup_gemini_models()

    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    retrieved_nodes: List[NodeWithScore] = retriever.retrieve(query_text)

    if not retrieved_nodes:
        logging.info("No relevant context found for the query.")
        return ""

    context_chunks = []
    for node_with_score in retrieved_nodes:
        context_chunks.append(node_with_score.node.get_content())
        # Optional: log score and metadata
        # logging.debug(f"Retrieved node score: {node_with_score.score}, metadata: {node_with_score.node.metadata}")


    full_context = context_separator.join(context_chunks)
    logging.info(f"Retrieved {len(retrieved_nodes)} context chunks.")
    return full_context


# --- Example Usage ---
if __name__ == "__main__":
    target_urls = [
        "https://docs.djangoproject.com/en/5.0/intro/overview/",
        "https://docs.djangoproject.com/en/5.0/topics/db/models/",
        "https://docs.djangoproject.com/en/5.0/topics/http/views/",
        "https://docs.djangoproject.com/en/5.0/topics/templates/"
    ]

    try:
        # Create or Load the Index
        # Make sure models are set up with desired chunking before index creation
        # This call to setup_gemini_models is slightly redundant if create_or_load_rag_index
        # also calls it, but it emphasizes that settings should be applied *before* indexing.
        # The create_or_load_rag_index now calls setup_gemini_models internally.
        vector_index = create_rag_index(
            urls=target_urls,
        )


    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except ImportError as ie:
        print(f"Import Error: {ie}. Make sure you have installed all required libraries.")
        print("pip install llama-index llama-index-embeddings-gemini llama-index-llms-gemini llama-index-readers-web beautifulsoup4")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")