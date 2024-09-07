import os
import tempfile
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from inference_sdk import InferenceHTTPClient

router = APIRouter()

# Inicializando o cliente da API
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("API_KEY", "DTEUmRzESZm6eaVCplPI")  # Melhor usando variáveis de ambiente
)

known_width_cm = 10
known_width_px = 100
scale = known_width_cm / known_width_px

padding_factor = 0.9  # Ajuste conforme necessário

@router.post("/")
async def infer_object(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    
    print(f"Arquivo recebido: {file.filename}")  # Para verificar o nome do arquivo recebido

    temp_dir = tempfile.gettempdir()
    temp_image_path = os.path.join(temp_dir, "temp_frame.jpg")
    
    try:
        with open(temp_image_path, "wb") as f:
            f.write(await file.read())  # Escrevendo o conteúdo do arquivo no formato original

        # Executa a função infer em um thread pool
        result = await run_in_threadpool(CLIENT.infer, temp_image_path, model_id="grid-2zfis/1")
        
        detections = []
        if result.get("predictions"):
            for prediction in result["predictions"]:
                width_px = int(prediction['width'])
                height_px = int(prediction['height'])

                width_px_adjusted = int(width_px * padding_factor)
                height_px_adjusted = int(height_px * padding_factor)

                width_cm = width_px_adjusted * scale
                height_cm = height_px_adjusted * scale

                detections.append({
                    "class": prediction['class'],
                    "width_cm": width_cm,
                    "height_cm": height_cm,
                    "x_min": int(prediction['x'] - prediction['width'] / 2),
                    "y_min": int(prediction['y'] - prediction['height'] / 2),
                    "x_max": int(prediction['x'] + prediction['width'] / 2),
                    "y_max": int(prediction['y'] + prediction['height'] / 2),
                })

            return JSONResponse(content={"detections": detections})
        else:
            return JSONResponse(content={"message": "No objects detected"}, status_code=400)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {e}")
    
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
