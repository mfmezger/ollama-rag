# Ollama RAG

## Idea

The Idea is to build a production ready RAG system using ollama as Retrieval and Generation Backend and Securing the Application with GuardLlama.

## Components

Rest Backend: FastAPI
LLM Provider: Ollama
Safeguard Provider: GuardLlama by Meta AI
Vector Database: Weaviate
LLM Engineering Framework: LlamaIndex
Server Side Events: https://devdojo.com/bobbyiliev/how-to-use-server-sent-events-sse-with-fastapi to allow for streaming responses

## Features

Normal RAG
Safeguards against Prompt Injection
Safeguards against Harmful Content Generation

## FAQ

Why Weaviate?
- Because of the ability to also use Keyword Search.



# TODO: change back to original ollama after httpx>=0.26.0
