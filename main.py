from fastapi import FastAPI
from app.services.crud import router as crud_router

app = FastAPI()
app.include_router(crud_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000,reload=True)