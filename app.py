import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import AzureOpenAI

load_dotenv()

app = FastAPI(title="Trip Planning Expert")

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = (BASE_DIR / "index.html").read_text(encoding="utf-8")

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a helpful trip-planning expert. Help users plan flights, hotels, itineraries, "
        "budgeting, destinations, transportation, and trip logistics. Be practical, clear, and "
        "specific. Mention that travelers should verify price, visa, weather, and local travel rules "
        "before booking."
    ),
}


def validate_azure_settings():
    missing = []
    for var_name in ("AZURE_ENDPOINT", "AZURE_DEPLOYMENT", "AZURE_API_KEY"):
        if not os.getenv(var_name):
            missing.append(var_name)

    if missing:
        raise HTTPException(
            status_code=500,
            detail="Azure configuration is incomplete. Set AZURE_ENDPOINT, AZURE_DEPLOYMENT, and AZURE_API_KEY.",
        )

    return (
        os.environ["AZURE_ENDPOINT"],
        os.environ["AZURE_DEPLOYMENT"],
        os.environ["AZURE_API_KEY"],
    )


@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTMLResponse(content=INDEX_HTML, status_code=200)


@app.post("/chat")
async def chat(payload: dict):
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="Expected a 'messages' list in the request body.")

    if not messages:
        raise HTTPException(status_code=400, detail="At least one message is required.")

    endpoint, deployment, api_key = validate_azure_settings()

    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-10-21",
        )

        completion = client.chat.completions.create(
            model=deployment,
            messages=[SYSTEM_MESSAGE, *messages],
            temperature=0.7,
            max_tokens=800,
        )

        reply = completion.choices[0].message.content
        if not reply:
            return {"reply": "I’m ready to help with your trip, but I did not receive a response from the model."}

        return {"reply": reply.strip()}
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The trip planner could not reach Azure OpenAI. Please verify the Azure configuration and try again.",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
