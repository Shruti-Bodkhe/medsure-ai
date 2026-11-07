from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import router
from app.database import init_db
from pathlib import Path

app = FastAPI(title="MedSure AI Clinical Governance API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    init_db()


app.include_router(router)

ui_path = Path("ui")
if ui_path.exists():
    app.mount("/static", StaticFiles(directory="ui"), name="static")


@app.get("/")
def serve_frontend():
    index_file = ui_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"status": "online", "system": "MedSure AI Clinical Governance API"}