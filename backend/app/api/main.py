"""FastAPI entry point for the inventory agent."""

from fastapi import FastAPI
from pydantic import BaseModel

from app.agent.agent import InventoryAgent, InventoryResponse


class AskRequest(BaseModel):
    """User message sent to the inventory agent."""

    message: str


app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is responding."""
    return {"status": "healthy"}


@app.post("/ask", response_model=InventoryResponse)
def ask(request: AskRequest) -> InventoryResponse:
    """Pass a user message to the inventory agent."""
    agent = InventoryAgent()
    return agent.run(request.message)
