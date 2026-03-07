"""FastAPI entry point for SBFA.

Endpoints:
- POST /task - Submit a task for routing and execution
- GET /agents - List registered agents
- POST /rag/ingest - Ingest a document into RAG
- POST /rag/query - Search RAG for relevant context
- GET /.well-known/agent-card.json - A2A v0.3 AgentCard discovery
- GET /agents/{agent_id}/card - Individual agent card
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sbfa.a2a.registry import AgentRegistry
from sbfa.agents.claude_agent import ClaudeAgent
from sbfa.agents.gemini_agent import GeminiAgent
from sbfa.agents.openai_agent import OpenAIAgent
from sbfa.config import settings
from sbfa.db.client import db_client
from sbfa.db.schema import initialize_database
from sbfa.orchestrator.coordinator import Coordinator
from sbfa.orchestrator.router import TaskRouter
from sbfa.rag.embeddings import get_embedding_provider
from sbfa.rag.retrieval import RAGRetriever
from sbfa.rag.store import RAGStore

router = TaskRouter()
registry: AgentRegistry | None = None
rag_store: RAGStore | None = None
rag_retriever: RAGRetriever | None = None
coordinator: Coordinator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry, rag_store, rag_retriever, coordinator

    await db_client.connect()
    await initialize_database(db_client)

    registry = AgentRegistry(db_client)
    embedding_provider = get_embedding_provider()
    rag_store = RAGStore(db_client, embedding_provider)
    rag_retriever = RAGRetriever(db_client, embedding_provider)

    agents = []
    if settings.anthropic_api_key:
        agents.append(ClaudeAgent())
    if settings.google_api_key:
        agents.append(GeminiAgent())
    if settings.openai_api_key:
        agents.append(OpenAIAgent())

    for agent in agents:
        router.register_agent(agent)
        await registry.register(agent.to_agent_card())

    coordinator = Coordinator(router, rag_retriever)

    yield

    await db_client.disconnect()


app = FastAPI(title="SBFA", version="0.1.0", lifespan=lifespan)


# --- Request/Response models ---

class TaskRequest(BaseModel):
    task: str


class TaskResponse(BaseModel):
    agent: str
    output: str
    latency_ms: int
    success: bool
    error: str | None = None


class IngestRequest(BaseModel):
    title: str
    source: str
    content: str
    metadata: dict[str, Any] | None = None
    chunk_size: int = 512
    chunk_overlap: int = 64


class IngestResponse(BaseModel):
    document_id: str
    message: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResult(BaseModel):
    content: str
    score: float
    document_id: str


# --- Endpoints ---

@app.post("/task", response_model=TaskResponse)
async def submit_task(request: TaskRequest):
    if not coordinator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = await coordinator.execute(request.task)
    return TaskResponse(
        agent=result.agent_name,
        output=result.output,
        latency_ms=result.latency_ms,
        success=result.success,
        error=result.error,
    )


@app.get("/agents")
async def list_agents():
    if not registry:
        raise HTTPException(status_code=503, detail="Service not initialized")
    cards = await registry.list_all()
    return [card.to_json() for card in cards]


@app.post("/rag/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    if not rag_store:
        raise HTTPException(status_code=503, detail="Service not initialized")

    doc_id = await rag_store.ingest_document(
        title=request.title,
        source=request.source,
        content=request.content,
        metadata=request.metadata,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    return IngestResponse(document_id=doc_id, message=f"Document '{request.title}' ingested")


@app.post("/rag/query", response_model=list[QueryResult])
async def query_rag(request: QueryRequest):
    if not rag_retriever:
        raise HTTPException(status_code=503, detail="Service not initialized")

    results = await rag_retriever.search(request.query, top_k=request.top_k)
    return [
        QueryResult(
            content=r.content,
            score=r.score,
            document_id=r.document_id,
        )
        for r in results
    ]


@app.get("/.well-known/agent-card.json")
async def agent_card_discovery():
    """A2A v0.3 AgentCard discovery endpoint."""
    if not registry:
        raise HTTPException(status_code=503, detail="Service not initialized")
    cards = await registry.list_all()
    if not cards:
        raise HTTPException(status_code=404, detail="No agents registered")
    return cards[0].to_json()


@app.get("/agents/{agent_id}/card")
async def get_agent_card(agent_id: str):
    if not registry:
        raise HTTPException(status_code=503, detail="Service not initialized")
    card = await registry.get(agent_id)
    if not card:
        raise HTTPException(status_code=404, detail="Agent not found")
    return card.to_json()
