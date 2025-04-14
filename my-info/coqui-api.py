# version: 1.10

import torch
from TTS.api import TTS
import gradio as gr
import os
import urllib.request
from fastapi import FastAPI, WebSocket, Query, Body, Depends, Header, HTTPException, Security
import uvicorn
from typing import Optional
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
from sse_starlette.sse import EventSourceResponse
import json
import base64
import struct
import asyncio
from clean_wav_files_in_outputs import async_clean_wav_files


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

def detect_language(text):
    """
    Detect the likely language of the text based on character ranges.
    Returns a language code or None if cannot determine.
    """
    # Check for language-specific characters
    
    # Chinese (Simplified & Traditional)
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        return "zh-cn"
    
    # Japanese specific characters (Hiragana and Katakana)
    if any('\u3040' <= char <= '\u30ff' for char in text):
        return "ja"
    
    # Korean (Hangul)
    if any('\uac00' <= char <= '\ud7a3' for char in text):
        return "ko"
    
    # Arabic
    if any('\u0600' <= char <= '\u06ff' for char in text):
        return "ar"
    
    # Russian (Cyrillic)
    if any('\u0400' <= char <= '\u04ff' for char in text):
        return "ru"
    
    # Hindi (Devanagari)
    if any('\u0900' <= char <= '\u097f' for char in text):
        return "hi"
    
    # For Latin-based languages (en, es, fr, de, it, pt, pl, nl, cs, hu)
    # we can't easily distinguish just by character range
    # Return None to indicate we couldn't detect it definitively
    return None

# Add this right after the ERROR_CODES dictionary
ERROR_CODES = {
    "SUCCESS": {"errorCode": 0, "message": "Success"},
    "INVALID_API_KEY": {"errorCode": 1001, "message": "Invalid or missing API Key"},
    "REFERENCE_AUDIO_NOT_FOUND": {"errorCode": 2001, "message": "Reference audio file not found"},
    "TTS_GENERATION_ERROR": {"errorCode": 3001, "message": "Error generating audio"},
    "WEBSOCKET_ERROR": {"errorCode": 4001, "message": "WebSocket connection error"},
    "RATE_LIMIT_EXCEEDED": {"errorCode": 5001, "message": "Rate limit exceeded"},
    "UNSUPPORTED_LANGUAGE": {"errorCode": 6001, "message": "Unsupported language"},
    "LANGUAGE_MISMATCH": {"errorCode": 6002, "message": "Text language doesn't match requested language"}
}

# Load supported languages
SUPPORTED_LANGUAGES = {}
try:
    with open("my-info/coqui-language/xtts_v2_supported_language.md", "r") as f:
        for line in f:
            if ":" in line:
                code, name = line.strip().split(":", 1)
                SUPPORTED_LANGUAGES[code.strip()] = name.strip()
except Exception as e:
    print(f"Error loading supported languages: {e}")
    # Fallback to the complete list of languages from xtts_v2_supported_language.md
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "pl": "Polish",
        "tr": "Turkish",
        "ru": "Russian",
        "nl": "Dutch",
        "cs": "Czech",
        "ar": "Arabic",
        "zh-cn": "Chinese (Simplified)",
        "hu": "Hungarian",
        "ko": "Korean",
        "ja": "Japanese",
        "hi": "Hindi"
    }

print(f"Supported languages loaded: {SUPPORTED_LANGUAGES}")

# Add a function to detect if text is likely Chinese
def likely_contains_chinese(text):
    """Check if the text contains Chinese characters"""
    return any('\u4e00' <= char <= '\u9fff' for char in text)

async def generate_audio_ws(text: str, language: str = "en") -> bytes:
    try:
        start_time = time.time()
        
        # Validate language
        language = language.lower() if language else "en"
        if language != "en" and language not in SUPPORTED_LANGUAGES:
            error = ERROR_CODES["UNSUPPORTED_LANGUAGE"]
            raise ValueError(f"{error['message']}: {language}")
        
        # Check for language mismatch
        detected_lang = detect_language(text)
        if detected_lang and language == "en" and detected_lang != "en":
            detected_lang_name = SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)
            error = ERROR_CODES["LANGUAGE_MISMATCH"]
            raise ValueError(f"{error['message']}: Text appears to be {detected_lang_name} but language is set to English")
        
        # Choose model based on language
        if language == "en":
            # Use FastPitch for English text
            with torch.inference_mode():
                wav = global_tts.tts(text=text)
        else:
            # Use XTTS v2 for other supported languages
            with torch.inference_mode():
                wav = global_chinese_tts.tts(
                    text=text,
                    speaker_wav=sample_wav_path,
                    language=language
                )

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
    "sk-3f4g5h6i7j8k9l0m1n2o3p4q5r6s7t8u9v0w1x2y": {"user": "user2", "rate_limit": 50},
    "sk-5z6y7x8w9v0u1t2s3r4q5p6o7n8m9l0k1j2i3h4g": {"user": "internal-service", "rate_limit": -1}  # -1 indicates unlimited
}

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    if hasattr(exc, "detail") and isinstance(exc.detail, dict) and "errorCode" in exc.detail and "message" in exc.detail:
        # If our custom format is already in the detail, use it directly
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    else:
        # For other exceptions, use a generic error
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "errorCode": 5000,
                "message": str(exc.detail)
            }
        )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "errorCode": 4000,
            "message": "Validation error",
            "details": str(exc)
        }
    )

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        error = ERROR_CODES["INVALID_API_KEY"]
        raise HTTPException(
            status_code=403,
            detail={"errorCode": error["errorCode"], "message": error["message"]}
        )
    
    if api_key_header in API_KEYS:
        return api_key_header
    
    error = ERROR_CODES["INVALID_API_KEY"]
    raise HTTPException(
        status_code=403,
        detail={"errorCode": error["errorCode"], "message": error["message"]}
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
            await websocket.send_json({"errorCode": error["errorCode"], "message": error["message"]})
            await websocket.close(1008)  # Policy violation close code
            return
            
        print(f"Valid API key from user: {API_KEYS[api_key]['user']}")
        
        while True:
            # Receive message as JSON
            data = await websocket.receive_text()
            try:
                # Try to parse as JSON
                message = json.loads(data)
                text = message.get("text", "")
                language = message.get("language", "en")
            except json.JSONDecodeError:
                # If not JSON, assume it's just text
                text = data
                language = "en"
            
            if text.startswith('[') and text.endswith(']'):
                text = text.strip('[]').strip('"\'')
            
            print(f"Processing text: {text} with language: {language}")
            
            try:
                # Validate language first
                language = language.lower() if language else "en"
                if language != "en" and language not in SUPPORTED_LANGUAGES:
                    error = ERROR_CODES["UNSUPPORTED_LANGUAGE"]
                    await websocket.send_json({
                        "errorCode": error["errorCode"], 
                        "message": f"{error['message']}: {language}"
                    })
                    continue
                
                # Check for language mismatch
                detected_lang = detect_language(text)
                if detected_lang and language == "en" and detected_lang != "en":
                    detected_lang_name = SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)
                    error = ERROR_CODES["LANGUAGE_MISMATCH"]
                    await websocket.send_json({
                        "errorCode": error["errorCode"], 
                        "message": f"{error['message']}: Text appears to be {detected_lang_name} but language is set to English"
                    })
                    continue
                
                async for audio_chunk in generate_audio_ws(text, language):
                    if audio_chunk:  # Only send non-empty chunks
                        await websocket.send_bytes(audio_chunk)
                # Send an empty chunk to signal completion
                await websocket.send_bytes(b'')
                
                # Schedule WAV cleanup without waiting for it to complete
                asyncio.create_task(async_clean_wav_files())
                
            except Exception as e:
                print(f"Error generating audio: {e}")
                error = ERROR_CODES["TTS_GENERATION_ERROR"]
                await websocket.send_json({"errorCode": error["errorCode"], "message": error["message"], "details": str(e)})
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            error = ERROR_CODES["WEBSOCKET_ERROR"]
            await websocket.send_json({"errorCode": error["errorCode"], "message": error["message"], "details": str(e)})
            await websocket.close()
        except:
            pass

# Update the TTSRequest model to include language parameter
class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "en"

@app.post("/tts")
async def generate_audio_http(
    request: TTSRequest,
    api_key: APIKey = Depends(get_api_key)
):
    try:
        # Log usage for the API key (in a real app, store this in a database)
        print(f"API request from user: {API_KEYS[api_key]['user']}")
        
        text = request.text
        language = request.language.lower() if request.language else "en"
        
        if text.startswith('[') and text.endswith(']'):
            text = text.strip('[]').strip('"\'')
            
        print(f"Processing text: {text} with language: {language}")
        
        # Validate language
        if language != "en" and language not in SUPPORTED_LANGUAGES:
            error = ERROR_CODES["UNSUPPORTED_LANGUAGE"]
            return {"errorCode": error["errorCode"], "message": f"{error['message']}: {language}"}
        
        # Check for language mismatch
        detected_lang = detect_language(text)
        if detected_lang and language == "en" and detected_lang != "en":
            detected_lang_name = SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)
            error = ERROR_CODES["LANGUAGE_MISMATCH"]
            return {"errorCode": error["errorCode"], "message": f"{error['message']}: Text appears to be {detected_lang_name} but language is set to English"}
        
        timestamp = int(time.time())
        output_path = f"outputs/output_{timestamp}.wav"
        
        # Choose model based on language
        if language == "en":
            print("Using FastPitch model for English")
            global_tts.tts_to_file(text=text, file_path=output_path)
        else:
            print(f"Using XTTS model with language: {language}")
            
            if not os.path.exists(sample_wav_path):
                error = ERROR_CODES["REFERENCE_AUDIO_NOT_FOUND"]
                return {"errorCode": error["errorCode"], "message": error["message"]}
                
            global_chinese_tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=sample_wav_path,
                language=language
            )
        
        # Schedule WAV cleanup without waiting for it to complete
        asyncio.create_task(async_clean_wav_files())
        
        # For file responses, we can't add errorCode/message, so we keep this as is
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"tts_output_{timestamp}.wav"
        )
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        error = ERROR_CODES["TTS_GENERATION_ERROR"]
        return {"errorCode": error["errorCode"], "message": error["message"], "details": str(e)}

# Update the usage endpoint to reflect unlimited usage for special keys
@app.get("/usage")
async def get_usage(api_key: str = Depends(get_api_key)):
    # Check one more time if the API key is valid (for extra safety)
    if api_key not in API_KEYS:
        error = ERROR_CODES["INVALID_API_KEY"]
        return JSONResponse(
            status_code=403,
            content={"errorCode": error["errorCode"], "message": error["message"]}
        )
    
    # If we get here, the API key is valid
    success = ERROR_CODES["SUCCESS"]
    
    user_info = API_KEYS[api_key]
    rate_limit_value = user_info["rate_limit"]
    
    # Format the rate limit display
    rate_limit_display = "Unlimited" if rate_limit_value == -1 else rate_limit_value
    
    return {
        "errorCode": success["errorCode"],
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

# Update the endpoint name and implementation for better performance
@app.post("/tts-http-stream")
async def generate_audio_http_stream(
    request: TTSRequest,
    api_key: APIKey = Depends(get_api_key)
):
    try:
        # Log usage for the API key
        print(f"API request from user: {API_KEYS[api_key]['user']}")
        
        text = request.text
        language = request.language.lower() if request.language else "en"
        
        if text.startswith('[') and text.endswith(']'):
            text = text.strip('[]').strip('"\'')
            
        print(f"Processing text: {text} with language: {language}")
        
        # Validate language
        if language != "en" and language not in SUPPORTED_LANGUAGES:
            error = ERROR_CODES["UNSUPPORTED_LANGUAGE"]
            return JSONResponse(
                status_code=400,
                content={"errorCode": error["errorCode"], "message": f"{error['message']}: {language}"}
            )
            
        # Check for language mismatch
        detected_lang = detect_language(text)
        if detected_lang and language == "en" and detected_lang != "en":
            detected_lang_name = SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)
            error = ERROR_CODES["LANGUAGE_MISMATCH"]
            return JSONResponse(
                status_code=400,
                content={"errorCode": error["errorCode"], "message": f"{error['message']}: Text appears to be {detected_lang_name} but language is set to English"}
            )
        
        # Create efficient binary stream generator
        async def binary_stream_generator():
            # Create WAV header first
            sample_rate = 22050
            bits_per_sample = 16
            channels = 1
            
            # Create a buffer for the WAV header
            header_buffer = io.BytesIO()
            
            # RIFF header
            header_buffer.write(b'RIFF')
            header_buffer.write(b'\x00\x00\x00\x00')  # placeholder for file size
            header_buffer.write(b'WAVE')
            
            # Format chunk
            header_buffer.write(b'fmt ')
            header_buffer.write(struct.pack('<I', 16))  # subchunk size (16 for PCM)
            header_buffer.write(struct.pack('<H', 1))   # PCM format
            header_buffer.write(struct.pack('<H', channels))  # channels
            header_buffer.write(struct.pack('<I', sample_rate))  # sample rate
            header_buffer.write(struct.pack('<I', sample_rate * channels * bits_per_sample // 8))  # byte rate
            header_buffer.write(struct.pack('<H', channels * bits_per_sample // 8))  # block align
            header_buffer.write(struct.pack('<H', bits_per_sample))  # bits per sample
            
            # Data chunk header
            header_buffer.write(b'data')
            header_buffer.write(b'\x00\x00\x00\x00')  # placeholder for data size
            
            # Send the header
            yield header_buffer.getvalue()
            
            # Buffer to collect audio data for size calculations
            data_buffer = bytearray()
            
            # Stream audio chunks
            async for audio_chunk in generate_audio_ws(text, language):
                if audio_chunk:
                    data_buffer.extend(audio_chunk)
                    yield audio_chunk
            
            # Calculate sizes and update the header
            data_size = len(data_buffer)
            file_size = data_size + 36  # 36 is the size of the header minus 8 bytes
            
            # Create updated header with correct sizes
            header_buffer = io.BytesIO()
            header_buffer.write(b'RIFF')
            header_buffer.write(struct.pack('<I', file_size))
            header_buffer.write(b'WAVE')
            header_buffer.write(b'fmt ')
            header_buffer.write(struct.pack('<I', 16))
            header_buffer.write(struct.pack('<H', 1))
            header_buffer.write(struct.pack('<H', channels))
            header_buffer.write(struct.pack('<I', sample_rate))
            header_buffer.write(struct.pack('<I', sample_rate * channels * bits_per_sample // 8))
            header_buffer.write(struct.pack('<H', channels * bits_per_sample // 8))
            header_buffer.write(struct.pack('<H', bits_per_sample))
            header_buffer.write(b'data')
            header_buffer.write(struct.pack('<I', data_size))
            
            # Note: We don't send this updated header because 
            # clients have already processed the initial header
            
            # Schedule WAV cleanup without waiting for it to complete
            asyncio.create_task(async_clean_wav_files())
                
        # Return the streaming response
        return StreamingResponse(
            binary_stream_generator(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tts_output_{int(time.time())}.wav"
            }
        )
        
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        error = ERROR_CODES["TTS_GENERATION_ERROR"]
        return JSONResponse(
            status_code=500,
            content={
                "errorCode": error["errorCode"], 
                "message": error["message"], 
                "details": str(e)
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9002)

import warnings
warnings.filterwarnings("ignore", message=".*attention mask.*")