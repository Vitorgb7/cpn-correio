from fastapi import APIRouter
from api.v1.endpoints import inference

api_router = APIRouter()
api_router.include_router(inference.router, prefix='/infer', tags=['infer'])
