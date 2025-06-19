import torch
import numpy as np
import io
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import time
from TTS.api import TTS

# API model for TTS request
class TTSRequest(BaseModel):
    text: str
    speed: Optional[float] = 1.0
    audio_format: Optional[str] = "mp3"  # wav or mp3

# Initialize FastAPI app
app = FastAPI()

# Global variable to store model
global_model = None

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    global global_model
    device = get_device()
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    
    # Initialize the German TTS model
    print("Loading German TTS model...")
    global_model = TTS("tts_models/de/thorsten/vits").to(device)
    print("German TTS model loaded")

@app.get("/")
async def root():
    return {"message": "German TTS API is running"}

@app.post("/tts")
async def generate_tts(request: TTSRequest):
    start_time = time.time()
    print(f"API start time: {time.strftime('%H:%M:%S.%f')[:-3]}")
    
    if global_model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Generate audio
        print(f"Generating audio for text: {request.text[:50]}{'...' if len(request.text) > 50 else ''}")
        
        before_inference_time = time.time()
        elapsed_since_start = (before_inference_time - start_time) * 1000
        print(f"Time before inference: {elapsed_since_start:.2f} ms since start")
        
        try:
            # Generate audio without saving to file
            audio_path = io.BytesIO()
            global_model.tts(text=request.text, file_path=audio_path, speed=request.speed)
            audio_path.seek(0)
        except Exception as inference_error:
            print(f"Inference error: {str(inference_error)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise
        
        after_inference_time = time.time()
        elapsed_since_start = (after_inference_time - start_time) * 1000
        elapsed_since_last = (after_inference_time - before_inference_time) * 1000
        print(f"Time after inference: {elapsed_since_start:.2f} ms since start, {elapsed_since_last:.2f} ms since before inference")
        
        # Create in-memory file
        audio_io = io.BytesIO()
        
        if request.audio_format.lower() == "wav":
            # For WAV, we can use the generated audio directly
            audio_io = audio_path
            media_type = "audio/wav"
            filename = "output.wav"
        elif request.audio_format.lower() == "mp3":
            # Convert to MP3
            try:
                import torchaudio
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Convert to MP3
                torchaudio.save(audio_io, waveform, sample_rate, format="mp3")
                media_type = "audio/mpeg"
                filename = "output.mp3"
            except Exception as e:
                print(f"MP3 conversion failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to convert to MP3: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported audio format: {request.audio_format}")
        
        audio_io.seek(0)
        
        first_byte_time = time.time()
        elapsed_since_start = (first_byte_time - start_time) * 1000
        elapsed_since_last = (first_byte_time - after_inference_time) * 1000
        print(f"Time to send first byte: {elapsed_since_start:.2f} ms since start, {elapsed_since_last:.2f} ms since after inference")
        
        generation_time = time.time() - start_time
        print(f"Audio generated in {generation_time:.2f} seconds")
        
        return StreamingResponse(
            audio_io, 
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except Exception as e:
        print(f"Error in generate_tts: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9003)
