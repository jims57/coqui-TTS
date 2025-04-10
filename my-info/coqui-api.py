# version: 1.10

import torch
from TTS.api import TTS
import gradio as gr
import os
import urllib.request
from fastapi import FastAPI, WebSocket, Query, Body, Depends, Header, HTTPException, Security
import uvicorn
from typing import Optional
from fastapi.responses import FileResponse, JSONResponse
import time
import io
import scipy.io.wavfile as wavfile
import numpy as np
from starlette.websockets import WebSocketDisconnect
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
import warnings
from pydantic import BaseModel
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


os.environ["TTS_HOME"] = "/app/coqui-tts"
print("TTS_HOME path:", os.environ.get("TTS_HOME"))

# More detailed CUDA diagnostics
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        print(f"Current GPU device: {torch.cuda.current_device()}")
        device = "cuda"
    except Exception as e:
        print(f"Error initializing CUDA: {e}")
        print("Falling back to CPU")
        device = "cpu"
else:
    device = "cpu"

print(f"Using device: {device}")

# Create directory if it doesn't exist
os.makedirs("reference_samples", exist_ok=True)

# Download a sample audio file if it doesn't exist
# sample_wav_path = "reference_samples/speaker2.mp3"
# sample_wav_path = "reference_samples/jack-mark-en-1.wav"
sample_wav_path = "reference_samples/andy-liu-en-1.wav"
# sample_wav_path = "reference_samples/leijun.wav"
if not os.path.exists(sample_wav_path):
    print("Downloading sample reference audio...")
    # Using VCTK dataset sample as reference
    sample_url = "https://raw.githubusercontent.com/coqui-ai/TTS/main/tests/data/ljspeech/wavs/LJ001-0001.wav"
    urllib.request.urlretrieve(sample_url, sample_wav_path)
    
# Add this to print the absolute path
print("Reference sample absolute path:", os.path.abspath(sample_wav_path))

# English fast
# def generate_audio(text="A journey of a thousand miles begins with a single step."):
def generate_audio(text="Great achievements often start with a small act of courage."):
# def generate_audio(text="我们计划下个月去旅行，想去看看那些美丽的风景。"):
    tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch").to(device)
    try:
        # Check device for the first model in the pipeline
        print(f"Model device: {next(tts.models['tts_model'].parameters()).device}")
    except Exception as e:
        print(f"Could not determine model device: {e}")
    
    tts.tts_to_file(text=text, file_path="outputs/output.wav")
    return "outputs/output.wav"

app = FastAPI()

# Global variable for TTS model
global_tts = None

def initialize_tts():
    global global_tts, global_chinese_tts
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        print("Initializing FastPitch model...")
        model_name = "tts_models/en/ljspeech/fast_pitch"
        global_tts = TTS(model_name=model_name).to(device)
        
        print("Initializing XTTS v2 model...")
        global_chinese_tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=True,
            gpu=torch.cuda.is_available()
        ).to(device)
        
        return global_tts, global_chinese_tts
    except Exception as e:
        print(f"Error initializing TTS models: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize TTS models when the FastAPI app starts"""
    global global_tts, global_chinese_tts
    if global_tts is None:
        global_tts, global_chinese_tts = initialize_tts()
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

def has_chinese(text):
    """Check if the text contains Chinese characters"""
    return any('\u4e00' <= char <= '\u9fff' for char in text)

async def generate_audio_ws(text: str) -> bytes:
    try:
        start_time = time.time()
        
        # Choose model based on text content
        if has_chinese(text):
            # Use XTTS v2 for Chinese text
            with torch.inference_mode():
                wav = global_chinese_tts.tts(
                    text=text,
                    speaker_wav=sample_wav_path,
                    language="zh"
                )
        else:
            # Use FastPitch for English text
            with torch.inference_mode():
                wav = global_tts.tts(text=text)

        # Calculate TTS generation time in milliseconds
        tts_time = (time.time() - start_time) * 1000
        print(f"TTS generation time: {tts_time:.2f} ms")
        
        # Normalize and convert to 16-bit PCM
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)
        
        # Define chunk size (in samples)
        chunk_size = 1024
        total_samples = len(wav)
        
        print(f"Total audio samples: {total_samples}")
        
        # Track first chunk timing
        first_chunk = True
        
        # Stream chunks
        for i in range(0, total_samples, chunk_size):
            chunk = wav[i:min(i + chunk_size, total_samples)]
            
            # Convert to bytes (raw PCM data)
            chunk_bytes = chunk.tobytes()
            
            if first_chunk:
                first_chunk_time = (time.time() - start_time) * 1000
                print(f"Time to first chunk: {first_chunk_time:.2f} ms")
                first_chunk = False
            
            print(f"Sending chunk size: {len(chunk_bytes)} bytes")
            
            yield chunk_bytes
            
    except Exception as e:
        print(f"Error in audio generation: {e}")
        raise

# API key configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# In a real application, you would store these in a database
# For demo purposes, we're using a dictionary
API_KEYS = {
    "sk-1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t": {"user": "user1", "rate_limit": 100},
    "sk-2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u": {"user": "user2", "rate_limit": 50},
    "sk-internal5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t": {"user": "internal-service", "rate_limit": -1}  # -1 indicates unlimited
}

# Update the ERROR_CODES dictionary to use errcode instead of code
ERROR_CODES = {
    "SUCCESS": {"errcode": 0, "message": "Success"},
    "INVALID_API_KEY": {"errcode": 1001, "message": "Invalid or missing API Key"},
    "REFERENCE_AUDIO_NOT_FOUND": {"errcode": 2001, "message": "Reference audio file not found"},
    "TTS_GENERATION_ERROR": {"errcode": 3001, "message": "Error generating audio"},
    "WEBSOCKET_ERROR": {"errcode": 4001, "message": "WebSocket connection error"},
    "RATE_LIMIT_EXCEEDED": {"errcode": 5001, "message": "Rate limit exceeded"}
}

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    if hasattr(exc, "detail") and isinstance(exc.detail, dict) and "errcode" in exc.detail and "message" in exc.detail:
        # If our custom format is already in the detail, use it directly
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail  # This will have errcode and message at the top level
        )
    else:
        # For other exceptions, use a generic error
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "errcode": 5000,
                "message": str(exc.detail)
            }
        )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "errcode": 4000,
            "message": "Validation error",
            "details": str(exc)
        }
    )

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        error = ERROR_CODES["INVALID_API_KEY"]
        raise HTTPException(
            status_code=403,
            detail={"errcode": error["errcode"], "message": error["message"]}
        )
    
    if api_key_header in API_KEYS:
        return api_key_header
    
    error = ERROR_CODES["INVALID_API_KEY"]
    raise HTTPException(
        status_code=403,
        detail={"errcode": error["errcode"], "message": error["message"]}
    )

@app.websocket("/tts-stream")
async def websocket_endpoint(websocket: WebSocket):
    print("WebSocket connection attempt")
    
    await websocket.accept()
    try:
        # Extract API key from headers
        headers = dict(websocket.headers)
        print(f"WebSocket headers received: {headers}")
        
        # Find API key in headers (case-insensitive)
        api_key = None
        for key, value in headers.items():
            if key.lower() == API_KEY_NAME.lower():
                api_key = value
                break
                
        print(f"API key extracted from headers: {api_key}")
        
        # Check if the API key is valid
        if not api_key or api_key not in API_KEYS:
            print(f"Invalid API key: {api_key}")
            error = ERROR_CODES["INVALID_API_KEY"]
            await websocket.send_json({"errcode": error["errcode"], "message": error["message"]})
            await websocket.close(1008)  # Policy violation close code
            return
            
        print(f"Valid API key from user: {API_KEYS[api_key]['user']}")
        
        while True:
            text = await websocket.receive_text()
            if text.startswith('[') and text.endswith(']'):
                text = text.strip('[]').strip('"\'')
            
            print(f"Processing text: {text}")
            
            try:
                async for audio_chunk in generate_audio_ws(text):
                    if audio_chunk:  # Only send non-empty chunks
                        await websocket.send_bytes(audio_chunk)
                # Send an empty chunk to signal completion
                await websocket.send_bytes(b'')
            except Exception as e:
                print(f"Error generating audio: {e}")
                error = ERROR_CODES["TTS_GENERATION_ERROR"]
                await websocket.send_json({"errcode": error["errcode"], "message": error["message"], "details": str(e)})
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            error = ERROR_CODES["WEBSOCKET_ERROR"]
            await websocket.send_json({"errcode": error["errcode"], "message": error["message"], "details": str(e)})
            await websocket.close()
        except:
            pass

# Add this class to define the request body structure
class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
async def generate_audio_http(
    request: TTSRequest,
    api_key: APIKey = Depends(get_api_key)
):
    try:
        # Log usage for the API key (in a real app, store this in a database)
        print(f"API request from user: {API_KEYS[api_key]['user']}")
        
        text = request.text
        if text.startswith('[') and text.endswith(']'):
            text = text.strip('[]').strip('"\'')
            
        print(f"Processing text: {text}")
        
        timestamp = int(time.time())
        output_path = f"outputs/output_{timestamp}.wav"
        
        # Choose model based on text content
        if has_chinese(text):
            language = "zh"
            print(f"Using XTTS model with language: {language}")
            
            if not os.path.exists(sample_wav_path):
                error = ERROR_CODES["REFERENCE_AUDIO_NOT_FOUND"]
                return {"errcode": error["errcode"], "message": error["message"]}
                
            global_chinese_tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=sample_wav_path,
                language=language
            )
        else:
            print("Using FastPitch model")
            global_tts.tts_to_file(text=text, file_path=output_path)
        
        # For file responses, we can't add errcode/message, so we keep this as is
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"tts_output_{timestamp}.wav"
        )
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        error = ERROR_CODES["TTS_GENERATION_ERROR"]
        return {"errcode": error["errcode"], "message": error["message"], "details": str(e)}

# Update the usage endpoint to reflect unlimited usage for special keys
@app.get("/usage")
async def get_usage(api_key: str = Depends(get_api_key)):
    # Check one more time if the API key is valid (for extra safety)
    if api_key not in API_KEYS:
        error = ERROR_CODES["INVALID_API_KEY"]
        return JSONResponse(
            status_code=403,
            content={"errcode": error["errcode"], "message": error["message"]}
        )
    
    # If we get here, the API key is valid
    success = ERROR_CODES["SUCCESS"]
    
    user_info = API_KEYS[api_key]
    rate_limit_value = user_info["rate_limit"]
    
    # Format the rate limit display
    rate_limit_display = "Unlimited" if rate_limit_value == -1 else rate_limit_value
    
    return {
        "errcode": success["errcode"],
        "message": success["message"],
        "data": {
            "user": user_info["user"],
            "rate_limit": rate_limit_display,
            "usage": {
                "requests_this_month": 42,  # Example value
                "tokens_this_month": 1250   # Example value
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9002)

import warnings
warnings.filterwarnings("ignore", message=".*attention mask.*")