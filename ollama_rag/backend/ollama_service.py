"""The script to initialize the Qdrant db backend with aleph alpha."""

import os

from dotenv import load_dotenv
from loguru import logger
import weaviate
from omegaconf import DictConfig
from llama_index.llms import Ollama
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.embeddings import LangchainEmbedding
from ultra_simple_config import load_config
from langchain.embeddings import OllamaEmbeddings
from llama_index.text_splitter import SentenceSplitter

# from ollama_rag.utils.tokenizing import count_tokens, get_tokenizer
# from ollama_rag.utils.utility import generate_prompt
# from ollama_rag.utils.vdb import get_db_connection
load_dotenv()

auth_config = weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))


class OllamaService:
    @load_config(location="config/main.yml")
    def __init__(self, cfg: DictConfig):
        """Initialize the Ollama Service."""
        self.cfg = cfg
        self.collection_name = cfg.weaviate.collection_name
        self.vector_db = weaviate.Client(
            url=cfg.weaviate.url, auth_client_secret=auth_config
        )
        self.llm = Ollama(model=cfg.ollama.qa_model, temperature=cfg.ollama.temperature)

        # pull the ollama models when the service is initialized
        # ollama.pull(cfg.ollama.embedding_model)
        # ollama.pull(cfg.ollama.qa_model)

    def start(self):
        """Start the Ollama Service."""
        logger.info("Starting the Ollama Service.")

        embeddings = LangchainEmbedding(
            OllamaEmbeddings(model=self.cfg.ollama.embedding_model)
        )
        text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        documents = SimpleDirectoryReader("data").load_data()

        service_context = ServiceContext.from_defaults(
            chunk_size=self.cfg.weaviate.chunk_size,
            llm=self.llm,
            embed_model=embeddings,
            text_splitter=text_splitter,
        )

        vector_store = WeaviateVectorStore(
            weaviate_client=self.vector_db, index_name=self.collection_name
        )

        index = VectorStoreIndex.from_documents(
            documents=documents,
            vector_store=vector_store,
            service_context=service_context,
        )

        query_engine = index.as_query_engine(
            streaming=False,
        )

        response = query_engine.query("Who invented the go programming language?")

        print(response)


if __name__ == "__main__":
    service = OllamaService()

    service.start()

    # index = VectorStoreIndex.from_documents(documents)
    # query_engine = index.as_query_engine()
    # response = query_engine.query("Who invented the go programming language?")
