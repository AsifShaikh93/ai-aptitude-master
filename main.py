from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent import aptitude_agent
import json

app = FastAPI(title="AI Aptitude Master")

class QueryRequest(BaseModel):
    question: str

async def stream_generator(question: str):
    """
    Generator that yields tokens from the agent as they are produced.
    Using astream_events is the most reliable way to capture final output
    tokens while skipping internal agent thoughts.
    """
    async for event in aptitude_agent.astream_events(
        {"input": question}, 
        version="v2"
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content

@app.post("/solve")
async def solve(request: QueryRequest):
    return StreamingResponse(
        stream_generator(request.question),
        media_type="text/plain",
        headers={
            "X-Accel-Buffering": "no", 
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
