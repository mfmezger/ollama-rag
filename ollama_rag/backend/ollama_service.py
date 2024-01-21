"""The script to initialize the Qdrant db backend with aleph alpha."""

import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import nltk
import numpy as np
from aleph_alpha_client import Client, CompletionRequest, ExplanationRequest, Prompt
from dotenv import load_dotenv
from langchain.docstore.document import Document as LangchainDocument
from langchain.document_loaders import DirectoryLoader, PyPDFium2Loader
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import NLTKTextSplitter
from langchain.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
from omegaconf import DictConfig

from aleph_alpha_rag.utils.configuration import load_config
from aleph_alpha_rag.utils.tokenizing import count_tokens, get_tokenizer
from aleph_alpha_rag.utils.utility import generate_prompt
from aleph_alpha_rag.utils.vdb import get_db_connection

nltk.download("punkt")  # This needs to be installed for the tokenizer to work.
load_dotenv()

aleph_alpha_token = os.getenv("HF_TOKEN")
tokenizer = None


class OllamaService:
    @load_config(location="config/main.yml")
    def __init__(self, cfg: DictConfig, collection_name: str, hf_token: str ):
        """Initialize the Ollama Service."""
        self.cfg = cfg
        self.collection_name = collection_name
        self.hf_token = hf_token
        self.vector_db: Qdrant = get_db_connection(collection_name=self.collection_name)

        if not self.aleph_alpha_token:
            raise ValueError("Token cannot be None or empty.")




    def send_completion_request(self, text: str) -> str:
        """Sends a completion request to the Luminous API.

        Args:
            text (str): The prompt to be sent to the API.
            token (str): The token for the Luminous API.

        Returns:
            str: The response from the API.

        Raises:
            ValueError: If the text or token is None or empty, or if the response or completion is empty.
        """
        if not text:
            raise ValueError("Text cannot be None or empty.")
        if not token:
            raise ValueError("Token cannot be None or empty.")

        client = Client(token=self.aleph_alpha_token)

        request = CompletionRequest(
            prompt=Prompt.from_text(text),
            maximum_tokens=self.aleph_alpha_completion.max_tokens,
            stop_sequences=[self.aleph_alpha_completion.stop_sequences],
            repetition_penalties_include_completion=self.aleph_alpha_completion.repetition_penalties_include_completion,
        )
        response = client.complete(request, model=self.aleph_alpha_completion.model)

        # ensure that the response is not empty
        if not response.completions:
            raise ValueError("Response is empty.")

        # ensure that the completion is not empty
        if not response.completions[0].completion:
            raise ValueError("Completion is empty.")

        return str(response.completions[0].completion)


    def embedd_documents(self, dir: str, aleph_alpha_token: str, ) -> None:
        """Embeds the documents in the given directory in the Aleph Alpha database.

        This method uses the Directory Loader for PDFs and the PyPDFium2Loader to load the documents.
        The documents are then added to the Qdrant DB which embeds them without deleting the old collection.

        Args:
            dir (str): The directory containing the PDFs to embed.
            aleph_alpha_token (str): The Aleph Alpha API token.

        Returns:
            None
        """
        vector_db: Qdrant = get_db_connection(collection_name=self.collection_name, aleph_alpha_token=self.aleph_alpha_token)

        loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFium2Loader)
        get_tokenizer()

        splitter = NLTKTextSplitter(length_function=count_tokens, chunk_size=300, chunk_overlap=50)
        docs = loader.load_and_split(splitter)

        logger.info(f"Loaded {len(docs)} documents.")
        text_list = [doc.page_content for doc in docs]
        metadata_list = [doc.metadata for doc in docs]
        vector_db.add_texts(texts=text_list, metadatas=metadata_list)

        logger.info("SUCCESS: Texts embedded.")


    def embedd_text_files(
        self,
        folder: str,
        seperator: str,
    ) -> None:
        """Embeds text files in the Aleph Alpha database.

        Args:
            folder (str): The folder containing the text files to embed.
            aleph_alpha_token (str): The Aleph Alpha API token.
            seperator (str): The seperator to use when splitting the text into chunks.

        Returns:
            None
        """
        vector_db: Qdrant = get_db_connection(collection_name=self.collection_name, aleph_alpha_token=self.aleph_alpha_token)

        # iterate over the files in the folder
        for file in os.listdir(folder):
            # check if the file is a .txt or .md file
            if not file.endswith((".txt", ".md")):
                continue

            # read the text from the file
            text = pathlib.Path(os.path.join(folder, file)).read_text()
            text_list: List = text.split(seperator)

            # check if first and last element are empty
            if not text_list[0]:
                text_list.pop(0)
            if not text_list[-1]:
                text_list.pop(-1)

            # ensure that the text is not empty
            if not text_list:
                raise ValueError("Text is empty.")

            logger.info(f"Loaded {len(text_list)} documents.")
            # get the name of the file
            metadata = os.path.splitext(file)[0]
            # add _ and an incrementing number to the metadata
            metadata_list: List = [{"source": f"{metadata}_{str(i)}", "page": 0} for i in range(len(text_list))]
            vector_db.add_texts(texts=text_list, metadatas=metadata_list)

        logger.info("SUCCESS: Text embedded.")


    def search_documents_aleph_alpha(
        self,
        query: str,
        amount: int = 1,
        threshold: float = 0.0,

    ) -> List[Tuple[LangchainDocument, float]]:
        """Searches the Aleph Alpha service for similar documents.

        Args:
            aleph_alpha_token (str): Aleph Alpha API Token.
            query (str): The query that should be searched for.
            amount (int, optional): The number of documents to return. Defaults to 1.

        Returns
            List[Tuple[Document, float]]: A list of tuples containing the documents and their similarity scores.
        """
        if not query:
            raise ValueError("Query cannot be None or empty.")
        if amount < 1:
            raise ValueError("Amount must be greater than 0.")
        # TODO: FILTER
        try:
            vector_db: Qdrant = get_db_connection(collection_name=self.collection_name, aleph_alpha_token=self.aleph_alpha_token)
            docs = vector_db.similarity_search_with_score(query=query, k=amount, score_threshold=threshold)
            logger.info("SUCCESS: Documents found.")
            return docs
        except Exception as e:
            logger.error(f"ERROR: Failed to search documents: {e}")
            raise Exception(f"Failed to search documents: {e}") from e


    def qa_aleph_alpha(
        self,
        documents: list[tuple[LangchainDocument, float]],
        query: str,
        summarization: bool = False,
    ) -> Tuple[str, str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]:
        """QA takes a list of documents and returns a list of answers.

        Args:
            aleph_alpha_token (str): The Aleph Alpha API token.
            documents (List[Tuple[Document, float]]): A list of tuples containing the document and its relevance score.
            query (str): The query to ask.
            summarization (bool, optional): Whether to use summarization. Defaults to False.

        Returns:
            Tuple[str, str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]: A tuple containing the answer, the prompt, and the metadata for the documents.
        """
        # if the list of documents contains only one document extract the text directly
        if len(documents) == 1:
            text = documents[0][0].page_content
            meta_data = documents[0][0].metadata

        else:
            # extract the text from the documents
            texts = [doc[0].page_content for doc in documents]
            if summarization:
                text = "".join(self.summarize_text_aleph_alpha(t) for t in texts)
            else:
                # combine the texts to one text
                text = " ".join(texts)
            meta_data = [doc[0].metadata for doc in documents]

        # load the prompt
        prompt = generate_prompt("aleph_alpha_qa.j2", text=text, query=query)

        try:
            # call the luminous api
            answer = self.send_completion_request(prompt)

        except ValueError as e:
            # if the code is PROMPT_TOO_LONG, split it into chunks
            if e.args[0] == "PROMPT_TOO_LONG":
                logger.info("Prompt too long. Summarizing.")

                # summarize the text
                short_text = self.summarize_text_aleph_alpha(text)

                # generate the prompt
                prompt = generate_prompt("aleph_alpha_qa.j2", text=short_text, query=query)

                # call the luminous api
                answer = self.send_completion_request(prompt)

        # extract the answer
        return answer, prompt, meta_data


  


if __name__ == "__main__":
    if not token:
        raise ValueError("Token cannot be None or empty.")

    aa_service = AlephAlphaService(collection_name="aleph_alpha", aleph_alpha_token=token)

    aa_service.embedd_documents_aleph_alpha("data")

    aa_service.qa_chain("What is Attention", "aleph_alpha")

# open the text file and read the text
# DOCS = search_documents_aleph_alpha(aleph_alpha_token=token, query="What are Attentions?", amount=1)
# logger.info(DOCS)
# explanation, score, text, answer, meta_data = explain_qa(aleph_alpha_token=token, document=DOCS, query="What are Attentions?")
# logger.info(f"Answer: {answer}")
# explanations = explain_completion(prompt, answer, token)

# print(explanation)
