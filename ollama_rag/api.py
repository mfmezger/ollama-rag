"""FastAPI Backend for the Aleph Alpha RAG."""
import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.openapi.utils import get_openapi
from langchain.docstore.document import Document as LangchainDocument
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient, models
from qdrant_client.http.models.models import UpdateResult
from starlette.responses import JSONResponse

from ollama_rag.backend.aleph_alpha_service import (
    custom_completion_prompt_aleph_alpha,
    embedd_documents,
    embedd_text_files,
    explain_qa,
    qa_aleph_alpha,
    search_documents_aleph_alpha,
)
from ollama_rag.data_model.request_data_model import (
    CustomPromptCompletion,
    EmbeddTextFilesRequest,
    ExplainQARequest,
    QARequest,
    SearchRequest,
)
from ollama_rag.data_model.response_data_model import (
    EmbeddingResponse,
    ExplainQAResponse,
    QAResponse,
    SearchResponse,
)
from ollama_rag.utils.configuration import load_config
from ollama_rag.utils.utility import (
    combine_text_from_list,
    create_tmp_folder,
    get_token,
    load_vec_db_conn,
)

# add file logger for loguru
logger.add("logs/file_{time}.log", backtrace=False, diagnose=False)
logger.info("Startup.")


def my_schema() -> dict:
    """Used to generate the OpenAPI schema.

    Returns:
        FastAPI: FastAPI App
    """
    openapi_schema = get_openapi(
        title="Conversational AI API",
        version="1.0",
        description="Retrieval Augmented Generation using Ollama.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# initialize the Fast API Application.
app = FastAPI(debug=True)
app.openapi = my_schema  # type: ignore

load_dotenv()

# load the token from the environment variables, is None if not set.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ALEPH_ALPHA_API_KEY = os.environ.get("ALEPH_ALPHA_API_KEY")
logger.info("Loading REST API Finished.")


@app.get("/")
def read_root() -> str:
    """Returns the welcome message.

    Returns:
        str: The welcome message.
    """
    return "Welcome to the Simple Ollama FastAPI Backend!"


def embedd_documents_wrapper(
    folder_name: str,
    token: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> None:
    """Call the right embedding function for the chosen backend.

    Args:
        folder_name (str): Name of the temporary folder.
        llm_backend (str, optional): LLM provider. Defaults to "openai".
        token (str, optional): Token for the LLM Provider of choice. Defaults to None.

    Raises:
        ValueError: If an invalid LLM Provider is set.
    """
    token = get_token(
        token=token,
        aleph_alpha_key=ALEPH_ALPHA_API_KEY,
    )

    # Embedd the documents with Aleph Alpha
    logger.debug("Embedding Documents with Aleph Alpha.")
    embedd_documents(
        dir=folder_name, aleph_alpha_token=token, collection_name=collection_name
    )


@app.post("/collection/create/{collection_name}")
@load_config(location="config/db.yml")
def create_collection(collection_name: str) -> None:
    """Create a new collection in the vector database.

    Args:
        llm_provider (LLMProvider): Name of the LLM Provider
        collection_name (str): Name of the Collection
    """
    qdrant_client, _ = initialize_qdrant_client_config()

    try:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=cfg.aleph_alpha_embeddings.size, distance=models.Distance.COSINE
            ),
        )
    except Exception:
        logger.info(
            f"FAILURE: Collection {collection_name} already exists or could not created."
        )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


@app.post("/embeddings/documents")
async def post_embedd_documents(
    files: List[UploadFile] = File(...),
    token: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> EmbeddingResponse:
    """Uploads multiple documents to the backend.

    Args:
        files (List[UploadFile], optional): Uploaded files. Defaults to File(...).

    Returns:
        JSONResponse: The response as JSON.
    """
    logger.info("Embedding Multiple Documents")
    token = get_token(token=token, aleph_alpha_key=ALEPH_ALPHA_API_KEY)
    tmp_dir = create_tmp_folder()

    file_names = []

    for file in files:
        file_name = file.filename
        file_names.append(file_name)

        # Save the file to the temporary folder
        if tmp_dir is None or not os.path.exists(tmp_dir):
            raise ValueError("Please provide a temporary folder to save the files.")

        if file_name is None:
            raise ValueError("Please provide a file to save.")

        with open(os.path.join(tmp_dir, file_name), "wb") as f:
            f.write(await file.read())

    embedd_documents_wrapper(
        folder_name=tmp_dir,
        token=token,
        collection_name=collection_name,
    )

    return EmbeddingResponse(status="success", files=file_names)


@app.post("/embeddings/texts/files")
async def post_embedd_text_files(request: EmbeddTextFilesRequest) -> EmbeddingResponse:
    """Embeds text files in the database.

    Args:
        request (EmbeddTextFilesRequest): The request parameters.

    Raises:
        ValueError: If a file does not have a valid name, if no temporary folder is provided, or if no token or LLM provider is specified.

    Returns:
        JSONResponse: A response indicating that the files were received and saved, along with the names of the files they were saved to.
    """
    logger.info("Embedding Text Files")
    tmp_dir = create_tmp_folder()

    file_names = []

    for file in request.files:
        file_name = file.filename
        file_names.append(file_name)

        if file_name is None:
            raise ValueError("File does not have a valid name.")

        # Save the files to the temporary folder
        if tmp_dir is None or not os.path.exists(tmp_dir):
            raise ValueError("Please provide a temporary folder to save the files.")

        with open(os.path.join(tmp_dir, file_name), "wb") as f:
            f.write(await file.read())

    token = get_token(
        token=request.token,
        aleph_alpha_key=ALEPH_ALPHA_API_KEY,
    )

    if request.llm_backend is None:
        raise ValueError("Please provide a LLM Provider of choice.")

    embedd_text_files(
        folder=tmp_dir, aleph_alpha_token=token, seperator=request.seperator
    )

    return EmbeddingResponse(status="success", files=file_names)


@app.post("/qa")
def post_question_answer(request: QARequest) -> QAResponse:
    """Answer a question based on the documents in the database.

    Args:
        request (QARequest): The request parameters.

    Raises:
        ValueError: Error if no query or token is provided.

    Returns:
        Tuple: Answer, Prompt and Meta Data
    """
    logger.info("Answering Question")
    # if the query is not provided, raise an error
    if request.query is None:
        raise ValueError("Please provide a Question.")

    token = get_token(
        token=request.token,
        llm_backend=request.llm_backend,
        aleph_alpha_key=ALEPH_ALPHA_API_KEY,
        openai_key=OPENAI_API_KEY,
    )

    # if the history flag is activated and the history is not provided, raise an error
    if request.history and request.history_list is None:
        raise ValueError("Please provide a HistoryList.")

    # summarize the history
    if request.history:
        # combine the texts
        text = combine_text_from_list(request.history_list)
        # summarize the text
        # FIXME: AA is going to remove the summarization function
        # summary = summarize_text_aleph_alpha(text=text, token=token)
        # combine the history and the query
        summary = ""
        request.query = f"{summary}\n{request.query}"

    documents = search_database(request.search)

    # call the qa function

    answer, prompt, meta_data = qa_aleph_alpha(
        query=request.query, documents=documents, aleph_alpha_token=token
    )

    return QAResponse(answer=answer, prompt=prompt, meta_data=meta_data)


@app.post("/explanation/explain-qa")
def post_explain_question_answer(request: ExplainQARequest) -> ExplainQAResponse:
    """Answer a question & explains it based on the documents in the database. This only works with Aleph Alpha.

    This uses the normal qa but combines it with the explain function.
    Args:
        query (str, optional): _description_. Defaults to None.
        aa_or_openai (str, optional): _description_. Defaults to "openai".
        token (str, optional): _description_. Defaults to None.
        amount (int, optional): _description_. Defaults to 1.

    Raises:
        ValueError: Error if no query or token is provided.

    Returns:
        Tuple: Answer, Prompt and Meta Data
    """
    logger.info("Answering Question and Explaining it.")
    # if the query is not provided, raise an error
    if request.qa.search.query is None:
        raise ValueError("Please provide a Question.")

    token = get_token(
        token=request.qa.search.token,
        aleph_alpha_key=ALEPH_ALPHA_API_KEY,
    )

    documents = search_database(request.qa.search)

    # call the qa function
    explanation, score, text, answer, meta_data = explain_qa(
        query=request.qa.search.query, document=documents, aleph_alpha_token=token
    )

    return ExplainQAResponse(
        explanation=explanation,
        score=score,
        text=text,
        answer=answer,
        meta_data=meta_data,
    )


@app.post("/semantic/search")
def post_search(request: SearchRequest) -> List[SearchResponse]:
    """Searches for a query in the vector database.

    Args:
        request (SearchRequest): The search request.

    Raises:
        ValueError: If the LLM provider is not implemented yet.

    Returns:
        List[str]: A list of matching documents.
    """
    logger.info("Searching for Documents")
    request.token = get_token(
        token=request.llm_backend.token,
        aleph_alpha_key=ALEPH_ALPHA_API_KEY,
    )

    if request.llm_backend is None:
        raise ValueError("Please provide a LLM Provider of choice.")

    DOCS = search_database(request)

    if not DOCS:
        logger.info("No Documents found.")
        return JSONResponse(content={"message": "No documents found."})

    logger.info(f"Found {len(DOCS)} documents.")

    response = []
    try:
        for d in DOCS:
            score = d[1]
            text = d[0].page_content
            page = d[0].metadata["page"]
            source = d[0].metadata["source"]
            response.append(
                SearchResponse(text=text, page=page, source=source, score=score)
            )
    except Exception:
        for d in DOCS:
            score = d[1]
            text = d[0].page_content
            source = d[0].metadata["source"]
            response.append(
                SearchResponse(text=text, page=0, source=source, score=score)
            )

    return response


def search_database(request: SearchRequest) -> List[tuple[LangchainDocument, float]]:
    """Searches the database for a query.

    Args:
        request (SearchRequest): The request parameters.

    Raises:
        ValueError: If the LLM provider is not implemented yet.

    Returns:
        JSON List of Documents consisting of the text, page, source and score.
    """
    logger.info("Searching for Documents")
    token = get_token(
        token=request.token,
        aleph_alpha_key=ALEPH_ALPHA_API_KEY,
    )

    # Embedd the documents with Aleph Alpha
    documents = search_documents_aleph_alpha(
        aleph_alpha_token=token,
        query=request.query,
        amount=request.amount,
        threshold=request.threshold,
        collection_name=request.collection_name,
    )

    logger.info(f"Found {len(documents)} documents.")
    return documents


@app.post("/process_document")
async def process_document(
    files: List[UploadFile] = File(...),
    llm_backend: str = "aa",
    token: Optional[str] = None,
    type: str = "invoice",
) -> None:
    """Process a document.

    Args:
        files (UploadFile): _description_
        llm_backend (str, optional): _description_. Defaults to "openai".
        token (Optional[str], optional): _description_. Defaults to None.
        type (str, optional): _description_. Defaults to "invoice".

    Returns:
        JSONResponse: _description_
    """
    logger.info("Processing Document")
    token = get_token(
        token=token,
        llm_backend=llm_backend,
        aleph_alpha_key=ALEPH_ALPHA_API_KEY,
        openai_key=OPENAI_API_KEY,
    )

    # Create a temporary folder to save the files
    tmp_dir = create_tmp_folder()

    file_names = []

    for file in files:
        file_name = file.filename
        file_names.append(file_name)

        # Save the file to the temporary folder
        if tmp_dir is None or not os.path.exists(tmp_dir):
            raise ValueError("Please provide a temporary folder to save the files.")

        if file_name is None:
            raise ValueError("Please provide a file to save.")

        with open(os.path.join(tmp_dir, file_name), "wb") as f:
            f.write(await file.read())

    process_documents_aleph_alpha(folder=tmp_dir, token=token, type=type)


@app.post("/llm/completion/custom")
async def custom_prompt_llm(request: CustomPromptCompletion) -> str:
    """This method sents a custom completion request to the LLM Provider.

    Args:
        request (CustomPromptCompletion): The request parameters.

    Raises:
        ValueError: If the LLM provider is not implemented yet.
    """
    logger.info("Sending Custom Completion Request")
    # sent a completion
    answer = custom_completion_prompt_aleph_alpha(
        prompt=request.prompt,
        token=request.token,
        model=request.model,
        stop_sequences=request.stop_sequences,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    return answer


@app.delete("/embeddings/delete/{collection_name}/{page}/{source}")
def delete(
    page: int,
    source: str,
    collection_name: str,
) -> UpdateResult:
    """Delete a Vector from the database based on the page and source.

    Args:
        page (int): The page of the Document
        source (str): The name of the Document
        llm_provider (LLMProvider, optional): The LLM Provider. Defaults to LLMProvider.OPENAI.

    Returns:
        UpdateResult: The result of the Deletion Operation from the Vector Database.
    """
    logger.info("Deleting Vector from Database")

    qdrant_client = load_vec_db_conn()

    result = qdrant_client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.page",
                        match=models.MatchValue(value=page),
                    ),
                    models.FieldCondition(
                        key="metadata.source", match=models.MatchValue(value=source)
                    ),
                ],
            )
        ),
    )

    logger.info("Deleted Point from Database via Metadata.")
    return result


@load_config(location="config/db.yml")
def initialize_qdrant_client_config(cfg: DictConfig):
    """Initialize the Qdrant Client.

    Args:
        cfg (DictConfig): Configuration from the file

    Returns:
        _type_: Qdrant Client and Configuration.
    """
    qdrant_client = QdrantClient(
        cfg.qdrant.url,
        port=cfg.qdrant.port,
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=cfg.qdrant.prefer_grpc,
    )
    return qdrant_client, cfg


def initialize_aleph_alpha_vector_db() -> None:
    """Initializes the Aleph Alpha vector db.

    Args:
        cfg (DictConfig): Configuration from the file
    """
    qdrant_client, cfg = initialize_qdrant_client_config()
    try:
        qdrant_client.get_collection(collection_name=cfg.qdrant.collection_name_aa)
        logger.info(
            f"SUCCESS: Collection {cfg.qdrant.collection_name_aa} already exists."
        )
    except Exception:
        generate_collection(
            qdrant_client,
            collection_name=cfg.qdrant.collection_name_aa,
            embeddings_size=cfg.aleph_alpha_embeddings.size,
        )


def generate_collection(qdrant_client, collection_name, embeddings_size):
    """Generate a collection for the Aleph Alpha Backend.

    Args:
        qdrant_client (_type_): _description_
        collection_name (_type_): _description_
        embeddings_size (_type_): _description_
    """


initialize_aleph_alpha_vector_db()

# for debugging useful.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
