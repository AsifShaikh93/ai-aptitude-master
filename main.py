from fastapi import FastAPI
from pydantic import BaseModel
from agent import aptitude_agent

app = FastAPI(title="AI Aptitude Master")


class QueryRequest(BaseModel):
    question: str


@app.post("/solve")
def solve(request: QueryRequest):

    result = aptitude_agent.run(request.question)

    return {
        "question": request.question,
        "answer": result
    }