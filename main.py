from fastapi import FastAPI
from app.services.product_endpoints import router as product
from app.services.user_endpoints import router as user
from app.services.search_endpoints import router as search

app = FastAPI()
# app.include_router(crud_router, prefix="/api/v1")
app.include_router(product,prefix="/api/v1")
app.include_router(user,prefix="/api/v1")
app.include_router(search,prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to the Recommender application!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000,reload=True)