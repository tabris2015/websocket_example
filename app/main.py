from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.routes import router
from app.db import create_db_and_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create db
    create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(router, prefix="/od", tags=["object_detection"])


@app.get("/")
async def index():
    return FileResponse("index.html")

static_files_app = StaticFiles(directory="assets")

app.mount("/assets", static_files_app)
