from fastapi import FastAPI
from api.v1.api import api_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title='API Hackathon')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas as origens
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix='/api/v1')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level='info', reload=True)