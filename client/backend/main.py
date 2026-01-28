import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from multi import recognize_speaker as multi_recognize_speaker
from single import predict_speaker

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_audio"

os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/recognize_speaker")
async def recognize_speaker(audio_file: UploadFile = File(...)):
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    print(f"DEBUG: Type of audio_file before accessing filename: {type(audio_file)}")
    file_extension = os.path.splitext(audio_file.filename)[1]
    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{file_extension}")

    try:
        with open(temp_path, "wb") as f:
            f.write(await audio_file.read())

        # Use the recognize_speaker function from multi.py
        result_dict = await multi_recognize_speaker(temp_path)
        return dict(result_dict)

    except Exception as e:
        print(f"Error during speaker recognition: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during speaker recognition.")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/verify")
async def verify_speaker(audio_file: UploadFile = File(...)):
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    print(f"DEBUG: Type of audio_file before accessing filename: {type(audio_file)}")
    file_extension = os.path.splitext(audio_file.filename)[1]
    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{file_extension}")

    try:
        with open(temp_path, "wb") as f:
            f.write(await audio_file.read())

        # Use the verify_speaker function from binary.py
        prediction, pred_id, avg_conf = predict_speaker(temp_path)
        print(prediction, pred_id, avg_conf)
        return {"prediction": prediction, "predicted_id": pred_id, "confidence": avg_conf}

    except Exception as e:
        print(f"Error during speaker recognition: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during speaker recognition.")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
def read_root():
    return {"status": "Voice Verification API is running."}
