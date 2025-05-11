import os
import logging
import sys
from typing import List, Optional

# Configure basic logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader, # Although we load from web, sometimes needed indirectly or for comparison
    StorageContext,
    load_index_from_storage,
    Settings,
    Document # Import Document type
)
from llama_index.readers.web import SimpleWebPageReader # Basic reader for specific URLs
# For more advanced crawling (whole site), consider readers from llama-hub:
# from llama_index.readers.web import TrafilaturaWebReader # Good at main content extraction
# from llama_index.readers.web import BeautifulSoupWebReader # More control over parsing
# Or community readers like ScrapeWebsiteReader (requires installing llama-index-readers-web or similar)

from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.node_parser import SentenceSplitter # Example of customizing parsing

# --- Configuration ---
# Load API key securely from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Optional: Specify Gemini model names
# Check Google's documentation for the latest available model names
# Common choices: models/embedding-001, models/text-embedding-004 (newer)
EMBEDDING_MODEL_NAME = "models/embedding-001"
# Common choices: gemini-pro, gemini-1.5-pro-latest, etc.
LLM_MODEL_NAME = "models/gemini-pro"


def create_or_load_rag_index(
    urls: List[str],
    persist_dir: str = "./rag_storage",
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    llm_model_name: str = LLM_MODEL_NAME,
    chunk_size: int = 1024, # Optional: Adjust chunk size for splitting documents
    chunk_overlap: int = 100, # Optional: Adjust chunk overlap
    force_reload: bool = False # Set to True to always re-scrape and re-index
) -> VectorStoreIndex:
    """
    Creates or loads a LlamaIndex RAG index from a list of web page URLs.

    Uses Gemini for embeddings and LLM (for potential query engine use),
    and persists the index to disk.

    Args:
        urls (List[str]): A list of URLs to scrape documentation from.
                          Note: SimpleWebPageReader fetches only these specific URLs.
                          For crawling an entire site, consider a different reader.
        persist_dir (str): Directory to save/load the index persistence files.
                           Defaults to "./rag_storage".
        embedding_model_name (str): The Gemini model name for embeddings.
        llm_model_name (str): The Gemini model name for the LLM.
        chunk_size (int): The target size for text chunks (nodes).
        chunk_overlap (int): The overlap between consecutive text chunks.
        force_reload (bool): If True, ignores existing persisted index and rebuilds.
                             Defaults to False.

    Returns:
        VectorStoreIndex: The created or loaded LlamaIndex VectorStoreIndex object.

    Raises:
        ValueError: If the GOOGLE_API_KEY environment variable is not set.
        ImportError: If necessary libraries are not installed.
    """
    logging.info(f"Initializing RAG index creation/loading process.")
    logging.info(f"Persistence directory: {persist_dir}")
    logging.info(f"Force reload: {force_reload}")

    # --- Global Settings (LLM and Embeddings) ---
    # It's good practice to set these globally via Settings
    try:
        Settings.llm = Gemini(model_name=llm_model_name, api_key=GOOGLE_API_KEY)
        Settings.embed_model = GeminiEmbedding(model_name=embedding_model_name, api_key=GOOGLE_API_KEY)
        # Optional: Customize the text splitter
        Settings.text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # Optional: Set chunk_size directly in Settings (affects default splitting)
        # Settings.chunk_size = chunk_size
        # Settings.chunk_overlap = chunk_overlap

    except Exception as e:
        logging.error(f"Error initializing Gemini models: {e}")
        logging.error("Ensure your GOOGLE_API_KEY is valid and has the Generative Language API enabled.")
        raise

    # --- Check for Existing Index ---
    if not force_reload and os.path.exists(persist_dir):
        try:
            logging.info(f"Loading existing index from {persist_dir}...")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            logging.info("Successfully loaded index from storage.")
            return index
        except Exception as e:
            logging.warning(f"Failed to load index from {persist_dir}: {e}. Rebuilding...")
            # Clean up potentially corrupted directory if loading failed before rebuilding
            import shutil
            shutil.rmtree(persist_dir, ignore_errors=True)


    # --- Load Data from Web ---
    logging.info(f"Loading documents from URLs: {urls}")
    # SimpleWebPageReader loads content from the specified URLs.
    # It does NOT crawl links within those pages by default.
    # For full site scraping, you'd need a crawler like ScrapeWebsiteReader (from llama-hub)
    # or implement custom logic with BeautifulSoupWebReader.
    reader = SimpleWebPageReader(html_to_text=True) # html_to_text helps clean up HTML tags
    documents = []
    try:
        # Load documents individually to better handle potential errors per URL
        for url in urls:
             logging.info(f"Attempting to load: {url}")
             try:
                 # SimpleWebPageReader returns a list, potentially with multiple documents per URL
                 # depending on its internal logic, but often just one Document object per URL.
                 page_docs = reader.load_data([url])
                 # Add metadata (like the source URL) to each document chunk
                 for doc in page_docs:
                     doc.metadata = doc.metadata or {} # Ensure metadata dict exists
                     doc.metadata['source_url'] = url
                 documents.extend(page_docs)
                 logging.info(f"Successfully loaded {len(page_docs)} document part(s) from: {url}")
             except Exception as e:
                 logging.error(f"Failed to load URL {url}: {e}")
                 # Decide if you want to continue with other URLs or raise an error
                 # continue
    except Exception as e:
        logging.error(f"An critical error occurred during document loading: {e}")
        raise

    if not documents:
        logging.error("No documents were successfully loaded. Cannot build index.")
        # Depending on requirements, you might return None or raise an error
        raise ValueError("Failed to load any documents from the provided URLs.")

    logging.info(f"Loaded a total of {len(documents)} document sections.")
    # LlamaIndex will automatically chunk these documents based on Settings.text_splitter

    # --- Create Index and Persist ---
    logging.info("Creating new index with Gemini embeddings...")
    # Define the storage context *before* building the index to enable persistence
    storage_context = StorageContext.from_defaults()

    # Build the index. Documents are automatically parsed and embedded.
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        # embed_model=Settings.embed_model can be explicitly passed but Settings is preferred
        show_progress=True # Shows progress bars for embedding
    )

    logging.info("Index creation complete.")

    # Persist the index to disk
    logging.info(f"Persisting index to {persist_dir}...")
    os.makedirs(persist_dir, exist_ok=True) # Ensure directory exists
    index.storage_context.persist(persist_dir=persist_dir)
    logging.info("Index persisted successfully.")

    return index

# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration for Example ---
    # Use a few pages from Django docs as an example
    # For the *entire* documentation, you'd need a crawler or a list of *all* relevant URLs.
    target_urls = [
        "https://docs.djangoproject.com/en/5.0/intro/overview/",
        "https://docs.djangoproject.com/en/5.0/topics/db/models/",
        "https://docs.djangoproject.com/en/5.0/topics/http/views/",
        "https://docs.djangoproject.com/en/5.0/topics/templates/"
    ]
    persistence_location = "./django_docs_index_gemini"

    # --- Create or Load the Index ---
    try:
        vector_index = create_or_load_rag_index(
            urls=target_urls,
            persist_dir=persistence_location,
            # force_reload=True # Uncomment to force rebuild the index
        )

        # --- Create Query Engine ---
        # The query engine uses the index and the configured LLM (from Settings)
        query_engine = vector_index.as_query_engine(
            # Optional: Configure retriever parameters
            # similarity_top_k=3, # Retrieve top 3 most similar nodes
            # Optional: Configure response synthesis mode
            # response_mode="compact", # Different ways to synthesize the response
        )
        logging.info("Query engine created. Ready to answer questions.")

        # --- Ask Questions ---
        print("\n--- Asking Questions ---")

        question1 = "What is a Django Model?"
        print(f"\nQ: {question1}")
        response1 = query_engine.query(question1)
        print(f"A: {response1}")
        # You can also inspect the source nodes used for the answer
        # print("Sources:")
        # for node in response1.source_nodes:
        #     print(f"- Score: {node.score:.4f}, URL: {node.metadata.get('source_url', 'N/A')}")
        #     # print(f"  Text: {node.get_content()[:150]}...") # Print snippet of source text

        question2 = "How do you define a view in Django?"
        print(f"\nQ: {question2}")
        response2 = query_engine.query(question2)
        print(f"A: {response2}")

        question3 = "Tell me about Django templates."
        print(f"\nQ: {question3}")
        response3 = query_engine.query(question3)
        print(f"A: {response3}")

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except ImportError as ie:
        print(f"Import Error: {ie}. Make sure you have installed all required libraries.")
        print("pip install llama-index llama-index-embeddings-gemini llama-index-llms-gemini llama-index-readers-web beautifulsoup4")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")