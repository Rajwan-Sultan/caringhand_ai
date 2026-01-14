from dotenv import load_dotenv
load_dotenv()  # Load environment variables first

from fastapi import FastAPI
from app.services.product_endpoints import router as product
from app.services.user_endpoints import router as user
from app.services.search_endpoints import router as search
from app.services.resume_endpoint import router as resume_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Caringhand AII",
    description="Upload and parse resume documents with LLM and regex fallback & it also do recommendation",
    version="1.0.0"
)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure appropriately for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# app.include_router(crud_router, prefix="/api/v1")
app.include_router(product,prefix="/api/v1")
app.include_router(user,prefix="/api/v1")
app.include_router(search,prefix="/api/v1")
app.include_router(resume_router,prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Welcome to the Recommender application!"}


# if __name__ == "__main__":
#     import uvicorn
#     # uvicorn.run("main:app", host="0.0.0.0", port=8000,reload=True)