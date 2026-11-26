"""FastAPI web server for the Air Painting Studio."""

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ai_assistant import analyze_canvas, beautify_stroke
from air_tracker import AirHandTracker

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Air Painting Studio", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the main web application."""
    return FileResponse(STATIC_DIR / "index.html")


class BeautifyRequest(BaseModel):
    points: list[dict[str, float]]


class AnalyzeRequest(BaseModel):
    strokes: list[dict]
    target_challenge: str | None = None


@app.get("/api/health")
async def health_check() -> dict:
    """Report server readiness."""
    return {"status": "ok", "engine": "mediapipe"}


@app.post("/api/ai/beautify")
async def ai_beautify_stroke(payload: BeautifyRequest) -> dict:
    """Classify and beautify a single hand-drawn stroke."""
    return beautify_stroke(payload.points)


@app.post("/api/ai/analyze")
async def ai_analyze_canvas(payload: AnalyzeRequest) -> dict:
    """Recognize shapes and composite doodles on the canvas."""
    return analyze_canvas(payload.strokes, payload.target_challenge)


@app.websocket("/ws/track")
async def track_hand_ws(websocket: WebSocket) -> None:
    """Receive JPEG webcam frames and return hand landmark coordinates."""
    await websocket.accept()
    tracker = AirHandTracker()
    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            result = await asyncio.to_thread(tracker.process_jpeg, frame_bytes)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    finally:
        tracker.close()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0" if "PORT" in os.environ else "127.0.0.1")
    uvicorn.run("server:app", host=host, port=port, reload=False)
