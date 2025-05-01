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
import subprocess
import glob
import fnmatch
from fastapi.staticfiles import StaticFiles


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

# Mount the current directory to serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Add a specific endpoint to serve webclient.html directly from root path
@app.get("/webclient.html")
async def get_webclient():
    return FileResponse("webclient.html")

@app.get("/webclient_raw_stream.html")
async def get_webclient():
    return FileResponse("webclient_raw_stream.html")

# Global variable for TTS model
global_tts = None

# Add this near the top of the file with other global variables
# Default number of words per TTS segment for streaming
DEFAULT_WORDS_PER_SEGMENT = 3

# Add these global variables at the top of the file, near other global variables
global_streaming_model = None
global_streaming_config = None

# Add these global variables at the top of the file, near other global variables
global_cached_latents = {}

# Add this near the top of the file with other global variables
SPEAKER_IDS = {
    # 1: "reference_samples/andy-liu-en-1.wav",
    1: "reference_samples/ms-speaker-female-1.mp3",
    2: "reference_samples/jack-mark-en-1.wav",
    3: "reference_samples/leijun.wav",
    4: "reference_samples/speaker2.mp3",
    5: "reference_samples/andy-liu-en-1.wav",
    # Default to andy-liu-en-1.wav for any other value
}

# Default speaker reference audio to use if speakerId is invalid or not provided
DEFAULT_SPEAKER_AUDIO = "reference_samples/ms-speaker-female-1.mp3"

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
    global global_tts, global_chinese_tts, global_streaming_model, global_streaming_config
    
    if global_tts is None:
        global_tts, global_chinese_tts = initialize_tts()
        
    # Initialize the streaming XTTS v2 model
    if global_streaming_model is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            
            print("Initializing streaming XTTS v2 model...")
            global_streaming_config = XttsConfig()
            global_streaming_config.load_json("/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
            global_streaming_model = Xtts.init_from_config(global_streaming_config)
            global_streaming_model.load_checkpoint(
                global_streaming_config, 
                checkpoint_dir="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/",
                checkpoint_path="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/model.pth",
                use_deepspeed=True
            )
            global_streaming_model.cuda()
            print("Streaming XTTS v2 model loaded successfully")
        except Exception as e:
            print(f"Error loading streaming XTTS v2 model: {e}")
    
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
    "LANGUAGE_MISMATCH": {"errorCode": 6002, "message": "Text language doesn't match requested language"},
    "INVALID_AUDIO_FORMAT": {"errorCode": 6003, "message": "Invalid audio format. Supported formats: wav, opus, mp3"}
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

# Add this function before the /aqs-tts endpoint
async def save_audio_chunk_background(chunk, chunk_path, audio_format):
    try:
        print(f"Saving audio chunk to {chunk_path}")
        
        # Convert PyTorch tensor to proper format for saving
        if isinstance(chunk, torch.Tensor):
            # Move to CPU and ensure it's the right shape
            chunk_audio = chunk.squeeze().cpu().numpy()
            
            if audio_format == "raw":
                # Save raw PCM data directly
                chunk_audio = np.clip(chunk_audio, -1, 1)
                chunk_audio = (chunk_audio * 32767).astype(np.int16)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
                
                # Write raw PCM data to file
                with open(chunk_path, 'wb') as f:
                    f.write(chunk_audio.tobytes())
                print(f"Saved RAW audio chunk to {chunk_path}")
            elif audio_format == "wav":
                # Save WAV file directly using scipy.io.wavfile
                # Normalize and convert to 16-bit PCM
                chunk_audio = np.clip(chunk_audio, -1, 1)
                chunk_audio = (chunk_audio * 32767).astype(np.int16)
                wavfile.write(chunk_path, 24000, chunk_audio)
                print(f"Saved WAV audio chunk")
            elif audio_format == "opus":
                # Use the same FFmpeg approach as in /tts endpoint
                # Normalize and convert to 16-bit PCM
                chunk_audio = np.clip(chunk_audio, -1, 1)
                chunk_audio = (chunk_audio * 32767).astype(np.int16)
                
                # Save as WAV first
                temp_wav_path = chunk_path.replace(".opus", ".temp.wav")
                wavfile.write(temp_wav_path, 24000, chunk_audio)
                
                # Convert to opus using identical FFmpeg parameters as /tts
                subprocess.run([
                    "ffmpeg",
                    "-i", temp_wav_path,
                    "-c:a", "libopus",
                    "-b:a", "32k",
                    "-application", "voip",
                    "-vbr", "on",
                    chunk_path,
                    "-y"  # Overwrite if exists
                ], check=False, capture_output=True)
                
                # Remove temporary WAV file
                try:
                    os.remove(temp_wav_path)
                except:
                    pass
                    
                print(f"Saved Opus audio chunk using same method as /tts")
            elif audio_format == "mp3":
                # Use FFmpeg to encode to MP3
                # Save as WAV first
                temp_wav_path = chunk_path.replace(".mp3", ".temp.wav")
                wavfile.write(temp_wav_path, 24000, chunk_audio)
                
                # Convert to MP3 using FFmpeg
                subprocess.run([
                    "ffmpeg",
                    "-i", temp_wav_path,
                    "-c:a", "libmp3lame",
                    "-b:a", "128k",
                    chunk_path,
                    "-y"  # Overwrite if exists
                ], check=False, capture_output=True)
                
                # Remove temporary WAV file
                try:
                    os.remove(temp_wav_path)
                except:
                    pass
                
                print(f"Saved MP3 audio chunk")
        else:
            print(f"Error: chunk is not a tensor, got {type(chunk)}")
    except Exception as e:
        print(f"Error saving audio chunk: {e}")

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
async def websocket_endpoint(websocket: WebSocket, api_key: Optional[str] = Query(None)):
    print("WebSocket connection attempt")
    
    await websocket.accept()
    try:
        # Extract API key from query parameters first
        if not api_key:
            # Extract API key from headers (case-insensitive) as fallback
            headers = dict(websocket.headers)
            print(f"WebSocket headers received: {headers}")
            
            # Find API key in headers (case-insensitive)
            api_key = None
            language = "en"  # Default language
            audio_format = "opus"  # Default audio format
            words_per_segment = DEFAULT_WORDS_PER_SEGMENT  # Default words per segment
            save_log = False  # Default to not saving logs
            
            for key, value in headers.items():
                key_lower = key.lower()
                if key_lower == API_KEY_NAME.lower():
                    api_key = value
                elif key_lower == "language":
                    language = value.lower()
                elif key_lower == "audioformat":
                    audio_format = value.lower()
                elif key_lower == "wordspersegment":
                    try:
                        words_per_segment = int(value)
                        # Ensure at least 1 word per segment
                        words_per_segment = max(1, words_per_segment)
                    except ValueError:
                        # If not a valid integer, use default
                        words_per_segment = DEFAULT_WORDS_PER_SEGMENT
                elif key_lower == "savelog":
                    # Only set to True if value is exactly "true" (case-insensitive)
                    save_log = value.lower() == "true"
        else:
            # Initialize default values when api_key is provided in query parameters
            language = "en"
            audio_format = "opus"
            words_per_segment = DEFAULT_WORDS_PER_SEGMENT
            save_log = False
        
        # For demo purposes, use a default API key if none provided
        if not api_key:
            api_key = "sk-1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t"  # Use one of your valid API keys
            
        print(f"API key extracted: {api_key}")
        
        # Validate audio format
        if audio_format not in ["wav", "opus", "mp3"]:
            error = ERROR_CODES["INVALID_AUDIO_FORMAT"]
            await websocket.send_json({"errorCode": error["errorCode"], "message": error["message"]})
            await websocket.close(1008)
            return
        
        # Check if the API key is valid
        if not api_key or api_key not in API_KEYS:
            print(f"Invalid API key: {api_key}")
            error = ERROR_CODES["INVALID_API_KEY"]
            await websocket.send_json({"errorCode": error["errorCode"], "message": error["message"]})
            await websocket.close(1008)  # Policy violation close code
            return
            
        print(f"Valid API key from user: {API_KEYS[api_key]['user']}")
        
        # Validate language
        if language != "en" and language not in SUPPORTED_LANGUAGES:
            error = ERROR_CODES["UNSUPPORTED_LANGUAGE"]
            await websocket.send_json({
                "errorCode": error["errorCode"], 
                "message": f"{error['message']}: {language}"
            })
            await websocket.close(1008)
            return
            
        while True:
            # Receive message as JSON
            data = await websocket.receive_text()
            try:
                # Try to parse as JSON
                message = json.loads(data)
                text = message.get("text", "")
                
                # Check if there's a config flag in the message and skip text processing
                if message.get("config", False):
                    # This is a configuration message, not a text to process
                    print("Received configuration message, skipping TTS processing")
                    continue
                
                # Check if text is empty or only whitespace
                if not text or text.isspace():
                    print("Error: Empty text received")
                    error = ERROR_CODES["TTS_GENERATION_ERROR"]
                    await websocket.send_json({
                        "errorCode": error["errorCode"], 
                        "message": "Empty text received. Please provide text to convert to speech."
                    })
                    continue
                
                # If language is provided in message, it overrides the header
                message_language = message.get("language")
                if message_language:
                    language = message_language.lower()
                # If audioFormat is provided in message, it overrides the header
                message_audio_format = message.get("audioFormat")
                if message_audio_format:
                    audio_format = message_audio_format.lower()
                # If wordsPerSegment is provided in message, it overrides the header
                message_words_per_segment = message.get("wordsPerSegment")
                if message_words_per_segment is not None:
                    try:
                        words_per_segment = int(message_words_per_segment)
                        # Ensure at least 1 word per segment
                        words_per_segment = max(1, words_per_segment)
                    except ValueError:
                        # If not a valid integer, keep current value
                        pass
                # If saveLog is provided in message, it overrides the header
                message_save_log = message.get("saveLog")
                if message_save_log is not None:
                    # Only set to True if value is exactly true (boolean) or "true" (string)
                    if isinstance(message_save_log, bool):
                        save_log = message_save_log
                    elif isinstance(message_save_log, str):
                        save_log = message_save_log.lower() == "true"
                
                # Validate audio format again in case it was changed in the message
                if audio_format not in ["wav", "opus", "mp3"]:
                    error = ERROR_CODES["INVALID_AUDIO_FORMAT"]
                    await websocket.send_json({"errorCode": error["errorCode"], "message": error["message"]})
                    continue
            except json.JSONDecodeError:
                # If not JSON, assume it's just text
                text = data
                
                # Check if text is empty or only whitespace
                if not text or text.isspace():
                    print("Error: Empty text received")
                    error = ERROR_CODES["TTS_GENERATION_ERROR"]
                    await websocket.send_json({
                        "errorCode": error["errorCode"], 
                        "message": "Empty text received. Please provide text to convert to speech."
                    })
                    continue
            
            if text.startswith('[') and text.endswith(']'):
                text = text.strip('[]').strip('"\'')
            
            print(f"Processing text: {text} with language: {language}, format: {audio_format}, words per segment: {words_per_segment}")
            
            try:
                # Validate language again in case it was changed in the message
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
                
                # Split text into segments of words_per_segment words
                segments = []
                
                # Handle Chinese characters differently if detected
                if detected_lang == "zh-cn":
                    # For Chinese, each character can be considered a "word"
                    char_count = 0
                    current_segment = ""
                    
                    for char in text:
                        current_segment += char
                        # Check if the character is a Chinese character
                        if '\u4e00' <= char <= '\u9fff':
                            char_count += 1
                            if char_count >= words_per_segment:
                                segments.append(current_segment)
                                current_segment = ""
                                char_count = 0
                    
                    # Add any remaining text as the final segment
                    if current_segment:
                        segments.append(current_segment)
                else:
                    # For other languages, split by whitespace
                    words = text.split()
                    
                    for i in range(0, len(words), words_per_segment):
                        segment = " ".join(words[i:i + words_per_segment])
                        segments.append(segment)
                
                # If no segments were created or text is very short, use the whole text
                if not segments:
                    segments = [text]
                
                print(f"Split text into {len(segments)} segments with {words_per_segment} words per segment")
                
                # Generate timestamp for this session
                timestamp = int(time.time() * 1000)  # Millisecond timestamp
                segment_files = []  # List to track segment audio files
                
                # Process each segment sequentially
                for i, segment in enumerate(segments):
                    print(f"Processing segment {i+1}/{len(segments)}: {segment[:30]}...")
                    
                    # Generate the audio data for this segment
                    try:
                        if language == "en":
                            # Use FastPitch for English text
                            with torch.inference_mode():
                                wav_result = global_tts.tts(text=segment)
                        else:
                            # Use XTTS v2 for other supported languages
                            with torch.inference_mode():
                                wav_result = global_chinese_tts.tts(
                                    text=segment,
                                    speaker_wav=sample_wav_path,
                                    language=language
                                )
                        
                        # Ensure wav is a properly formatted numpy array
                        wav = ensure_numpy_array(wav_result)
                        
                    except Exception as tts_err:
                        print(f"Error in TTS generation for segment {i+1}: {tts_err}")
                        # Use tts_to_file as a fallback if direct TTS fails
                        fallback_timestamp = int(time.time())
                        fallback_wav_path = f"outputs/fallback_{fallback_timestamp}_{i}.wav"
                        
                        if language == "en":
                            global_tts.tts_to_file(text=segment, file_path=fallback_wav_path)
                        else:
                            global_chinese_tts.tts_to_file(
                                text=segment,
                                file_path=fallback_wav_path,
                                speaker_wav=sample_wav_path,
                                language=language
                            )
                        
                        # Load the wav file
                        sample_rate, wav = wavfile.read(fallback_wav_path)
                        # Convert to float in range [-1, 1]
                        wav = wav.astype(np.float32) / 32767.0
                    
                    # Normalize and convert to 16-bit PCM
                    wav = np.clip(wav, -1, 1)
                    wav = (wav * 32767).astype(np.int16)
                    
                    # Make sure wav is a 1D array
                    if not isinstance(wav, np.ndarray) or wav.ndim == 0:
                        print("Converting scalar to 1D array")
                        wav = np.array([wav.item() if hasattr(wav, 'item') else float(wav)], dtype=np.int16)
                    
                    # Save segment audio in background (don't wait for it)
                    segment_filename = f"{timestamp}-{i+1}.{audio_format}"
                    segment_path = f"outputs/{segment_filename}"
                    segment_files.append(segment_path)  # Track for later combining
                    
                    # Start async task to save the segment
                    asyncio.create_task(
                        save_segment_audio_background(wav, segment_path, audio_format)
                    )
                    
                    # For first segment, process and start streaming immediately
                    if i == 0:
                        # Define chunk size (in samples)
                        chunk_size = 1024
                        total_samples = len(wav)
                        
                        print(f"Total audio samples in segment {i+1}: {total_samples}")
                        
                        # If WAV format, stream raw PCM chunks
                        if audio_format == "wav":
                            print(f"Streaming first segment in WAV format")
                            # Stream chunks
                            for j in range(0, total_samples, chunk_size):
                                chunk = wav[j:min(j + chunk_size, total_samples)]
                                
                                # Convert to bytes (raw PCM data)
                                chunk_bytes = chunk.tobytes()
                                await websocket.send_bytes(chunk_bytes)
                        
                        # If Opus format, use FFmpeg to encode and send chunks
                        elif audio_format == "opus":
                            print(f"Streaming first segment in Opus format")
                            # Create a subprocess for FFmpeg
                            process = subprocess.Popen(
                                [
                                    "ffmpeg",
                                    "-f", "s16le",      # 16-bit PCM input
                                    "-ar", "22050",     # Sample rate
                                    "-ac", "1",         # Mono
                                    "-i", "pipe:0",     # Read from stdin
                                    "-c:a", "libopus",
                                    "-b:a", "32k",
                                    "-application", "voip",
                                    "-vbr", "on",
                                    "-f", "opus",
                                    "pipe:1"            # Output to stdout
                                ],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                            
                            # Send the entire PCM data to ffmpeg
                            opus_data, stderr = process.communicate(input=wav.tobytes())
                            
                            if process.returncode != 0:
                                print(f"FFmpeg error: {stderr.decode('utf-8', errors='ignore')}")
                                error = ERROR_CODES["TTS_GENERATION_ERROR"]
                                await websocket.send_json({
                                    "errorCode": error["errorCode"], 
                                    "message": f"{error['message']}: FFmpeg encoding failed"
                                })
                                continue
                            
                            # Stream the opus data in chunks
                            opus_chunk_size = 1024  # Bytes per chunk
                            for j in range(0, len(opus_data), opus_chunk_size):
                                opus_chunk = opus_data[j:min(j + opus_chunk_size, len(opus_data))]
                                await websocket.send_bytes(opus_chunk)
                        elif audio_format == "mp3":
                            # For MP3 format, collect audio data for silence-based splitting
                            mp3_buffer.append(first_audio_chunk)
                            
                            # Try to split based on silence
                            if len(mp3_buffer) > 1:  # Need at least some audio to analyze
                                # Combine chunks into one tensor for silence analysis
                                combined_audio = torch.cat([c.squeeze() for c in mp3_buffer], dim=0)
                                
                                # Convert to numpy array for processing
                                numpy_audio = combined_audio.cpu().numpy()
                                
                                # Convert to AudioSegment for silence detection (16-bit PCM)
                                pcm_audio = np.clip(numpy_audio, -1, 1)
                                pcm_audio = (pcm_audio * 32767).astype(np.int16)
                                
                                from pydub import AudioSegment
                                from pydub.silence import split_on_silence
                                
                                audio_segment = AudioSegment(
                                    data=pcm_audio.tobytes(),
                                    sample_width=2,  # 16-bit
                                    frame_rate=24000,
                                    channels=1
                                )
                                
                                # Split on silence using parameters from split_wav_into_mp3_based_on_silence.py
                                chunks = split_on_silence(
                                    audio_segment,
                                    min_silence_len=80,     # 80ms: Minimum silence to be considered a pause
                                    silence_thresh=-28,     # -28 dBFS: Threshold for silence
                                    keep_silence=60         # 60ms: Keep some silence for natural sound
                                )
                                
                                # If we found at least one valid split
                                if len(chunks) > 1:
                                    # Convert the first chunk to MP3 and send it
                                    first_chunk = chunks[0]
                                    
                                    # Export to MP3 in memory
                                    mp3_io = io.BytesIO()
                                    first_chunk.export(mp3_io, format="mp3", bitrate="128k", parameters=["-ac", "1"])
                                    mp3_io.seek(0)
                                    mp3_data = mp3_io.read()
                                    
                                    # Send the MP3 data
                                    await websocket.send_bytes(mp3_data)
                                    
                                    # Remove the processed audio from the buffer
                                    # Calculate how many samples we need to remove (first_chunk length in samples)
                                    samples_to_remove = len(first_chunk) * 24000 // 1000  # Convert ms to samples
                                    
                                    # Skip removing if we don't have enough samples
                                    if samples_to_remove > 0 and samples_to_remove < combined_audio.shape[0]:
                                        # Create a new buffer with the remaining audio
                                        remaining_samples = combined_audio[samples_to_remove:]
                                        
                                        # Clear the buffer and add remaining audio as a single chunk
                                        mp3_buffer = [remaining_samples.unsqueeze(0)]
                        else:  # opus
                            # Use FFmpeg to encode to opus
                            process = subprocess.Popen(
                                [
                                    "ffmpeg",
                                    "-f", "s16le",      # 16-bit PCM input
                                    "-ar", "24000",     # Sample rate - XTTS uses 24kHz
                                    "-ac", "1",         # Mono
                                    "-i", "pipe:0",     # Read from stdin
                                    "-c:a", "libopus",
                                    "-b:a", "32k",
                                    "-application", "voip",
                                    "-vbr", "on",
                                    "-f", "opus",
                                    "pipe:1"            # Output to stdout
                                ],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                            
                            # Convert the pytorch tensor to PCM data
                            chunk_pcm = np.clip(first_audio_chunk.squeeze().cpu().numpy(), -1, 1)
                            chunk_pcm = (chunk_pcm * 32767).astype(np.int16)
                            
                            # Send the PCM data to ffmpeg and get opus data
                            opus_data, stderr = process.communicate(input=chunk_pcm.tobytes())
                            
                            if process.returncode != 0:
                                print(f"FFmpeg error: {stderr.decode('utf-8', errors='ignore')}")
                            else:
                                # Send the opus data - first response goes out immediately
                                await websocket.send_bytes(opus_data)
                    else:
                        # For subsequent segments, stream immediately after generating
                        chunk_size = 1024
                        total_samples = len(wav)
                        
                        # If WAV format, stream raw PCM chunks
                        if audio_format == "wav":
                            # Stream chunks
                            for j in range(0, total_samples, chunk_size):
                                chunk = wav[j:min(j + chunk_size, total_samples)]
                                
                                # Convert to bytes (raw PCM data)
                                chunk_bytes = chunk.tobytes()
                                await websocket.send_bytes(chunk_bytes)
                        
                        # If Opus format, use FFmpeg to encode and send chunks
                        elif audio_format == "opus":
                            # Create a subprocess for FFmpeg
                            process = subprocess.Popen(
                                [
                                    "ffmpeg",
                                    "-f", "s16le",      # 16-bit PCM input
                                    "-ar", "22050",     # Sample rate
                                    "-ac", "1",         # Mono
                                    "-i", "pipe:0",     # Read from stdin
                                    "-c:a", "libopus",
                                    "-b:a", "32k",
                                    "-application", "voip",
                                    "-vbr", "on",
                                    "-f", "opus",
                                    "pipe:1"            # Output to stdout
                                ],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                            
                            # Send the entire PCM data to ffmpeg
                            opus_data, stderr = process.communicate(input=wav.tobytes())
                            
                            if process.returncode != 0:
                                print(f"FFmpeg error: {stderr.decode('utf-8', errors='ignore')}")
                                continue
                            
                            # Stream the opus data in chunks
                            opus_chunk_size = 1024  # Bytes per chunk
                            for j in range(0, len(opus_data), opus_chunk_size):
                                opus_chunk = opus_data[j:min(j + opus_chunk_size, len(opus_data))]
                                await websocket.send_bytes(opus_chunk)
                        elif audio_format == "mp3":
                            # For MP3 format, collect audio data for silence-based splitting
                            mp3_buffer.append(chunk)
                            
                            # Try to split based on silence
                            if len(mp3_buffer) > 1:  # Need at least some audio to analyze
                                # Combine chunks into one tensor for silence analysis
                                combined_audio = torch.cat([c.squeeze() for c in mp3_buffer], dim=0)
                                
                                # Convert to numpy array for processing
                                numpy_audio = combined_audio.cpu().numpy()
                                
                                # Convert to AudioSegment for silence detection (16-bit PCM)
                                pcm_audio = np.clip(numpy_audio, -1, 1)
                                pcm_audio = (pcm_audio * 32767).astype(np.int16)
                                
                                from pydub import AudioSegment
                                from pydub.silence import split_on_silence
                                
                                audio_segment = AudioSegment(
                                    data=pcm_audio.tobytes(),
                                    sample_width=2,  # 16-bit
                                    frame_rate=24000,
                                    channels=1
                                )
                                
                                # Split on silence using parameters from split_wav_into_mp3_based_on_silence.py
                                chunks = split_on_silence(
                                    audio_segment,
                                    min_silence_len=80,     # 80ms: Minimum silence to be considered a pause
                                    silence_thresh=-28,     # -28 dBFS: Threshold for silence
                                    keep_silence=60         # 60ms: Keep some silence for natural sound
                                )
                                
                                # If we found at least one valid split
                                if len(chunks) > 1:
                                    # Convert the first chunk to MP3 and send it
                                    first_chunk = chunks[0]
                                    
                                    # Export to MP3 in memory
                                    mp3_io = io.BytesIO()
                                    first_chunk.export(mp3_io, format="mp3", bitrate="128k", parameters=["-ac", "1"])
                                    mp3_io.seek(0)
                                    mp3_data = mp3_io.read()
                                    
                                    # Send the MP3 data
                                    await websocket.send_bytes(mp3_data)
                                    
                                    # Remove the processed audio from the buffer
                                    # Calculate how many samples we need to remove (first_chunk length in samples)
                                    samples_to_remove = len(first_chunk) * 24000 // 1000  # Convert ms to samples
                                    
                                    # Skip removing if we don't have enough samples
                                    if samples_to_remove > 0 and samples_to_remove < combined_audio.shape[0]:
                                        # Create a new buffer with the remaining audio
                                        remaining_samples = combined_audio[samples_to_remove:]
                                        
                                        # Clear the buffer and add remaining audio as a single chunk
                                        mp3_buffer = [remaining_samples.unsqueeze(0)]
                        else:  # opus
                            # Use FFmpeg to encode to opus
                            process = subprocess.Popen(
                                [
                                    "ffmpeg",
                                    "-f", "s16le",      # 16-bit PCM input
                                    "-ar", "24000",     # Sample rate - XTTS uses 24kHz
                                    "-ac", "1",         # Mono
                                    "-i", "pipe:0",     # Read from stdin
                                    "-c:a", "libopus",
                                    "-b:a", "32k",
                                    "-application", "voip",
                                    "-vbr", "on",
                                    "-f", "opus",
                                    "pipe:1"            # Output to stdout
                                ],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                            
                            # Convert the pytorch tensor to PCM data
                            chunk_pcm = np.clip(chunk.squeeze().cpu().numpy(), -1, 1)
                            chunk_pcm = (chunk_pcm * 32767).astype(np.int16)
                            
                            # Send the PCM data to ffmpeg and get opus data
                            opus_data, stderr = process.communicate(input=chunk_pcm.tobytes())
                            
                            if process.returncode != 0:
                                print(f"FFmpeg error: {stderr.decode('utf-8', errors='ignore')}")
                                continue
                            
                            # Send the opus data
                            await websocket.send_bytes(opus_data)
                
                # Full audio file path
                full_audio_path = f"outputs/{timestamp}-full.{audio_format}"
                
                # Start a task to combine all segment files in the background
                # But keep a reference to the task so we can wait for it before cleanup
                combine_task = asyncio.create_task(
                    combine_audio_segments_background(segment_files, full_audio_path, audio_format)
                )
                
                # Only save info log if saveLog is true
                if save_log:
                    # Create a text file with information about the segmented generation
                    info_path = f"outputs/info_{timestamp}.txt"
                    asyncio.create_task(
                        save_tts_info_background(text, language, words_per_segment, len(segments), info_path, segment_files, full_audio_path)
                    )
                
                # Send an empty chunk to signal completion
                await websocket.send_bytes(b'')
                
                # Send "EOT" text message to signal end of transmission
                await websocket.send_text("EOT")
                
                # Initialize combine_task to None to avoid reference errors
                combine_task = None
                
                # Only combine and save audio files if saveAudioFile is True
                if save_audio_file:
                    # Full audio file path (for raw format, only save the final mp3)
                    if audio_format == "raw":
                        full_audio_path = f"outputs/{timestamp}-full.mp3"
                        # For raw format, we don't need to combine chunk files as they weren't saved individually
                        if len(raw_buffer) > 0:
                            # Save the combined raw buffer to MP3 directly
                            # This will be done only once after all chunks are processed
                            all_audio = torch.cat([chunk.squeeze() for chunk in raw_buffer], dim=0)
                            combined_audio = all_audio.cpu().numpy()
                            
                            # Convert and save as MP3
                            try:
                                # Create MP3 file
                                combined_pcm = np.clip(combined_audio, -1, 1)
                                combined_pcm = (combined_pcm * 32767).astype(np.int16)
                                
                                # Save using FFmpeg with high quality settings
                                process = subprocess.Popen(
                                    [
                                        "ffmpeg",
                                        "-f", "s16le",
                                        "-ar", "24000",
                                        "-ac", "1",
                                        "-i", "pipe:0",
                                        "-c:a", "libmp3lame",
                                        "-b:a", "128k",
                                        "-q:a", "2",
                                        "-joint_stereo", "0",
                                        full_audio_path,
                                        "-y"
                                    ],
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE
                                )
                                _, stderr = process.communicate(input=combined_pcm.tobytes())
                                
                                if process.returncode == 0:
                                    print(f"Successfully saved combined raw audio to MP3: {full_audio_path}")
                                else:
                                    print(f"Error saving combined raw audio: {stderr.decode('utf-8', errors='ignore')}")
                            except Exception as e:
                                print(f"Exception saving combined raw audio: {e}")
                        
                        # For raw format, no need to run the combine task since we did it directly
                        combine_task = None
                    else:
                        # For other formats, proceed with normal combination
                        full_audio_path = f"outputs/{timestamp}-full.{audio_format}"
                        # Start a task to combine all chunk files in the background
                        combine_task = asyncio.create_task(
                            combine_audio_chunks_background(chunk_files, full_audio_path, audio_format)
                        )
                
                # Only save info log if saveLog is true
                if save_log:
                    # Create a text file with information about the chunked generation
                    info_path = f"outputs/info_{timestamp}.txt"
                    asyncio.create_task(
                        save_tts_info_background(text, language, None, len(chunk_files), info_path, chunk_files, full_audio_path)
                    )
                
                # Wait for the combination task to complete before scheduling cleanup
                try:
                    # Wait for combine task with a reasonable timeout
                    if combine_task is not None:  # Only wait if the task was created
                        await asyncio.wait_for(combine_task, timeout=30.0)
                        print(f"Audio combination completed for {full_audio_path}")
                    
                    # Print the "Time to first chunk" information again before cleanup
                    print(f"Time to first chunk: {first_chunk_time:.2f} ms")
                    
                    # Schedule WAV cleanup without waiting for it to complete
                    print("Starting cleanup process...")
                    asyncio.create_task(async_clean_wav_files())
                except asyncio.TimeoutError:
                    print(f"Warning: Audio combination timed out for {full_audio_path}")
                except Exception as e:
                    print(f"Error waiting for audio combination: {e}")
                else:
                    # Print the "Time to first chunk" information again before cleanup
                    print(f"Time to first chunk: {first_chunk_time:.2f} ms")
                    
                    # If we're not saving audio files, still run the cleanup to remove older files
                    print("Starting cleanup process...")
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

@app.websocket("/streaming-tts")
async def websocket_endpoint_streaming(websocket: WebSocket, api_key: Optional[str] = Query(None)):
    print("WebSocket Streaming TTS connection attempt")
    
    await websocket.accept()
    try:
        # Extract API key from query parameters first
        if not api_key:
            # Extract API key from headers (case-insensitive) as fallback
            headers = dict(websocket.headers)
            print(f"WebSocket headers received: {headers}")
            
            # Find API key in headers (case-insensitive)
            api_key = None
            
            for key, value in headers.items():
                key_lower = key.lower()
                if key_lower == API_KEY_NAME.lower():
                    api_key = value
        
        # For demo purposes, use a default API key if none provided
        if not api_key:
            api_key = "sk-1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t"  # Use one of your valid API keys
            
        print(f"API key extracted: {api_key}")
        
        # Check if the API key is valid
        if not api_key or api_key not in API_KEYS:
            print(f"Invalid API key: {api_key}")
            error = ERROR_CODES["INVALID_API_KEY"]
            await websocket.send_json({"errorCode": error["errorCode"], "message": error["message"]})
            await websocket.close(1008)  # Policy violation close code
            return
            
        print(f"Valid API key from user: {API_KEYS[api_key]['user']}")
        
        while True:
            # Receive message as text
            data = await websocket.receive_text()
            
            # Check if the received data is valid JSON
            try:
                # Try to parse as JSON
                message = json.loads(data)
                
                # Extract required parameters from the JSON
                text = message.get("text", "")
                language = message.get("language", "en").lower()
                audio_format = message.get("audioFormat", "opus").lower()
                save_log = message.get("saveLog", False)
                # Added the new saveAudioFile parameter
                save_audio_file = message.get("saveAudioFile", False)
                
                # Convert saveLog to boolean if it's a string
                if isinstance(save_log, str):
                    save_log = save_log.lower() == "true"
                
                # Convert saveAudioFile to boolean if it's a string
                if isinstance(save_audio_file, str):
                    save_audio_file = save_audio_file.lower() == "true"
                
                # Check if there's a config flag in the message and skip text processing
                if message.get("config", False):
                    # This is a configuration message, not a text to process
                    print("Received configuration message, skipping TTS processing")
                    continue
                
            except json.JSONDecodeError:
                # If not valid JSON, return error
                print("Error: Invalid JSON format received")
                error = ERROR_CODES["TTS_GENERATION_ERROR"]
                await websocket.send_json({
                    "errorCode": error["errorCode"], 
                    "message": "Invalid JSON format. Expected format: {\"language\": \"en\", \"audioFormat\": \"opus\", \"saveLog\": false, \"saveAudioFile\": false, \"text\": \"Your text here\"}"
                })
                continue
            
            # Check if text is empty or only whitespace
            if not text or text.isspace():
                print("Error: Empty text received")
                error = ERROR_CODES["TTS_GENERATION_ERROR"]
                await websocket.send_json({
                    "errorCode": error["errorCode"], 
                    "message": "Empty text received. Please provide text to convert to speech."
                })
                continue
            
            if text.startswith('[') and text.endswith(']'):
                text = text.strip('[]').strip('"\'')
            
            print(f"Processing text: {text} with language: {language}, format: {audio_format}, saveAudioFile: {save_audio_file}")
            
            # Validate audio format
            if audio_format not in ["wav", "opus", "mp3"]:
                error = ERROR_CODES["INVALID_AUDIO_FORMAT"]
                await websocket.send_json({"errorCode": error["errorCode"], "message": error["message"]})
                continue
            
            try:
                # Validate language again in case it was changed in the message
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
                
                # Check if the streaming model is available
                if global_streaming_model is None:
                    error = ERROR_CODES["TTS_GENERATION_ERROR"]
                    await websocket.send_json({
                        "errorCode": error["errorCode"],
                        "message": "Streaming TTS model not initialized"
                    })
                    continue
                
                # Use XTTS v2 model to generate audio streams
                timestamp = int(time.time() * 1000)  # Millisecond timestamp
                
                # Create a cache key based on the reference audio path
                cache_key = sample_wav_path
                
                # Check if we have cached latents for this reference audio
                start_time = time.time()
                if cache_key in global_cached_latents:
                    print("Using cached speaker latents...")
                    gpt_cond_latent, speaker_embedding = global_cached_latents[cache_key]
                    cached_time = (time.time() - start_time) * 1000
                    print(f"Retrieved cached speaker latents in {cached_time:.2f} ms")
                else:
                    # Compute speaker latents if not cached
                    print("Computing speaker latents (not cached)...")
                    with torch.inference_mode():
                        gpt_cond_latent, speaker_embedding = global_streaming_model.get_conditioning_latents(audio_path=[sample_wav_path])
                    
                    # Cache the computed latents for future use
                    global_cached_latents[cache_key] = (gpt_cond_latent, speaker_embedding)
                    compute_time = (time.time() - start_time) * 1000
                    print(f"Speaker latents computed and cached in {compute_time:.2f} ms")
                
                # Split text by punctuation for better TTS quality
                # Define punctuation for splitting
                punctuation_markers = ['.', '!', '?', ';', ',', ':', '。', '！', '、', '？', '；', '，', '：']
                
                # Function to split text by punctuation while keeping the punctuation
                def split_by_punctuation(text):
                    segments = []
                    current_segment = ""
                    
                    for char in text:
                        current_segment += char
                        if char in punctuation_markers:
                            if current_segment.strip():  # Only add non-empty segments
                                segments.append(current_segment.strip())
                            current_segment = ""
                    
                    # Add any remaining text
                    if current_segment.strip():
                        segments.append(current_segment.strip())
                    
                    # If we have no splits (no punctuation in text), use the whole text
                    if not segments:
                        segments = [text]
                    
                    # For Chinese text, ensure first segment isn't too long for fast response
                    if detected_lang == "zh-cn" and segments and len(segments[0]) > 25:
                        # Extract a shorter first segment if it's Chinese and too long
                        # This helps get the first audio chunk to the client faster
                        first_part = segments[0][:25]
                        rest_part = segments[0][25:]
                        segments[0] = first_part
                        # Only insert the rest if it's not empty
                        if rest_part.strip():
                            segments.insert(1, rest_part)
                    
                    # Combine very short segments with the next segment
                    combined_segments = []
                    current_combined = ""
                    
                    for segment in segments:
                        # If current segment is short (less than 5 chars) or current_combined is empty
                        if len(segment) < 5 or not current_combined:
                            current_combined += " " + segment if current_combined else segment
                        else:
                            combined_segments.append(current_combined)
                            current_combined = segment
                    
                    # Add the last combined segment if it exists
                    if current_combined:
                        combined_segments.append(current_combined)
                    
                    return combined_segments
                
                # Split the text
                text_segments = split_by_punctuation(text)
                print(f"Split text into {len(text_segments)} segments by punctuation")
                
                # Start streaming inference
                print("Starting streaming inference on text segments...")
                chunk_files = []  # List to track chunk audio files
                all_chunks = []  # To collect all chunks for combining at the end if saving
                
                # Import torchaudio for high-quality audio processing
                import torchaudio
                import io
                
                # Set up streaming parameters
                stream_chunk_size = 10  # Larger chunks for better continuity
                overlap_wav_len = 3072  # Overlap for smoother transitions
                
                print(f"Using stream_chunk_size={stream_chunk_size}, overlap_wav_len={overlap_wav_len}")
                
                first_chunk = True
                first_chunk_time = 0
                chunk_counter = 0
                
                # Process each text segment
                for segment_idx, segment in enumerate(text_segments):
                    print(f"Processing segment {segment_idx+1}/{len(text_segments)}: {segment[:50]}{'...' if len(segment) > 50 else ''}")
                    
                    # Process this segment through the streaming model
                    stream_chunks_iterator = global_streaming_model.inference_stream(
                        text=segment,
                        language=language,
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        stream_chunk_size=stream_chunk_size,
                        overlap_wav_len=overlap_wav_len,
                        temperature=0.1,
                        length_penalty=1.0,
                        repetition_penalty=90.0,
                        top_k=50,
                        speed=1.0,
                        enable_text_splitting=True
                    )
                    
                    # For first segment only, get and send the first chunk immediately for fast response
                    if segment_idx == 0:
                        try:
                            # Get just the first chunk
                            first_audio_chunk = next(stream_chunks_iterator)
                            chunk_counter += 1
                            
                            # Track timing info for first chunk
                            chunk_time = (time.time() - start_time) * 1000  # Time in milliseconds
                            first_chunk_time = chunk_time
                            print(f"Time to first chunk: {first_chunk_time:.2f} ms")
                            first_chunk = False
                            
                            # Only save the chunk to a file if saveAudioFile is True
                            if save_audio_file:
                                # If audioFormat is raw, use .raw extension instead of .mp3
                                if audio_format == "raw":
                                    chunk_filename = f"outputs/{timestamp}-{chunk_counter}.raw"
                                else:
                                    chunk_filename = f"outputs/{timestamp}-{chunk_counter}.{audio_format if audio_format != 'raw' else 'mp3'}"
                                
                                chunk_files.append(chunk_filename)
                                
                                # Start a task to save the chunk in the background
                                asyncio.create_task(
                                    save_audio_chunk_background(first_audio_chunk, chunk_filename, audio_format)
                                )
                            
                            # Keep track of all chunks for later combining if needed
                            if save_audio_file:
                                all_chunks.append(first_audio_chunk)
                            
                            # Stream the chunk to the client with high quality
                            if isinstance(first_audio_chunk, torch.Tensor):
                                # Prepare tensor for streaming
                                chunk_audio = first_audio_chunk.squeeze().unsqueeze(0).cpu()
                                
                                # Stream in the requested format
                                if audio_format == "wav":
                                    # Create in-memory WAV file
                                    wav_buffer = io.BytesIO()
                                    torchaudio.save(wav_buffer, chunk_audio, 24000, format="wav")
                                    wav_buffer.seek(0)
                                    wav_data = wav_buffer.read()
                                    
                                    # Send WAV data - first response goes out immediately
                                    await websocket.send_bytes(wav_data)
                                elif audio_format == "mp3":
                                    # For MP3 format, collect audio data for silence-based splitting
                                    mp3_buffer.append(first_audio_chunk)
                                    
                                    # Try to split based on silence
                                    if len(mp3_buffer) > 1:  # Need at least some audio to analyze
                                        # Combine chunks into one tensor for silence analysis
                                        combined_audio = torch.cat([c.squeeze() for c in mp3_buffer], dim=0)
                                        
                                        # Convert to numpy array for processing
                                        numpy_audio = combined_audio.cpu().numpy()
                                        
                                        # Convert to AudioSegment for silence detection (16-bit PCM)
                                        pcm_audio = np.clip(numpy_audio, -1, 1)
                                        pcm_audio = (pcm_audio * 32767).astype(np.int16)
                                        
                                        from pydub import AudioSegment
                                        from pydub.silence import split_on_silence
                                        
                                        audio_segment = AudioSegment(
                                            data=pcm_audio.tobytes(),
                                            sample_width=2,  # 16-bit
                                            frame_rate=24000,
                                            channels=1
                                        )
                                        
                                        # Split on silence using parameters from split_wav_into_mp3_based_on_silence.py
                                        chunks = split_on_silence(
                                            audio_segment,
                                            min_silence_len=80,     # 80ms: Minimum silence to be considered a pause
                                            silence_thresh=-28,     # -28 dBFS: Threshold for silence
                                            keep_silence=60         # 60ms: Keep some silence for natural sound
                                        )
                                        
                                        # If we found at least one valid split
                                        if len(chunks) > 1:
                                            # Convert the first chunk to MP3 and send it
                                            first_chunk = chunks[0]
                                            
                                            # Export to MP3 in memory
                                            mp3_io = io.BytesIO()
                                            first_chunk.export(mp3_io, format="mp3", bitrate="128k", parameters=["-ac", "1"])
                                            mp3_io.seek(0)
                                            mp3_data = mp3_io.read()
                                            
                                            # Send the MP3 data
                                            await websocket.send_bytes(mp3_data)
                                            
                                            # Remove the processed audio from the buffer
                                            # Calculate how many samples we need to remove (first_chunk length in samples)
                                            samples_to_remove = len(first_chunk) * 24000 // 1000  # Convert ms to samples
                                            
                                            # Skip removing if we don't have enough samples
                                            if samples_to_remove > 0 and samples_to_remove < combined_audio.shape[0]:
                                                # Create a new buffer with the remaining audio
                                                remaining_samples = combined_audio[samples_to_remove:]
                                                
                                                # Clear the buffer and add remaining audio as a single chunk
                                                mp3_buffer = [remaining_samples.unsqueeze(0)]
                                else:  # opus
                                    # Use FFmpeg to encode to opus
                                    process = subprocess.Popen(
                                        [
                                            "ffmpeg",
                                            "-f", "s16le",      # 16-bit PCM input
                                            "-ar", "24000",     # Sample rate - XTTS uses 24kHz
                                            "-ac", "1",         # Mono
                                            "-i", "pipe:0",     # Read from stdin
                                            "-c:a", "libopus",
                                            "-b:a", "32k",
                                            "-application", "voip",
                                            "-vbr", "on",
                                            "-f", "opus",
                                            "pipe:1"            # Output to stdout
                                        ],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE
                                    )
                                    
                                    # Convert the pytorch tensor to PCM data
                                    chunk_pcm = np.clip(first_audio_chunk.squeeze().cpu().numpy(), -1, 1)
                                    chunk_pcm = (chunk_pcm * 32767).astype(np.int16)
                                    
                                    # Send the PCM data to ffmpeg and get opus data
                                    opus_data, stderr = process.communicate(input=chunk_pcm.tobytes())
                                    
                                    if process.returncode != 0:
                                        print(f"FFmpeg error: {stderr.decode('utf-8', errors='ignore')}")
                                    else:
                                        # Send the opus data - first response goes out immediately
                                        await websocket.send_bytes(opus_data)
                            
                            # Now process remaining chunks for first segment
                            for chunk in stream_chunks_iterator:
                                chunk_counter += 1
                                
                                # Only save the chunk to a file if saveAudioFile is True
                                if save_audio_file:
                                    # If audioFormat is raw, use .raw extension instead of .mp3
                                    if audio_format == "raw":
                                        chunk_filename = f"outputs/{timestamp}-{chunk_counter}.raw"
                                    else:
                                        chunk_filename = f"outputs/{timestamp}-{chunk_counter}.{audio_format if audio_format != 'raw' else 'mp3'}"
                                    
                                    chunk_files.append(chunk_filename)
                                    
                                    # Start a task to save the chunk in the background
                                    asyncio.create_task(
                                        save_audio_chunk_background(chunk, chunk_filename, audio_format)
                                    )
                                
                                # Keep track of all chunks for later combining if needed
                                if save_audio_file:
                                    all_chunks.append(chunk)
                                
                                # Stream the chunk to the client with high quality
                                if isinstance(chunk, torch.Tensor):
                                    # Prepare tensor for streaming
                                    chunk_audio = chunk.squeeze().unsqueeze(0).cpu()
                                    
                                    # Stream in the requested format
                                    if audio_format == "wav":
                                        # Create in-memory WAV file
                                        wav_buffer = io.BytesIO()
                                        torchaudio.save(wav_buffer, chunk_audio, 24000, format="wav")
                                        wav_buffer.seek(0)
                                        wav_data = wav_buffer.read()
                                        
                                        # Send WAV data
                                        await websocket.send_bytes(wav_data)
                                    elif audio_format == "mp3":
                                        # For MP3 format, collect audio data for silence-based splitting
                                        mp3_buffer.append(chunk)
                                        
                                        # Try to split based on silence
                                        if len(mp3_buffer) > 1:  # Need at least some audio to analyze
                                            # Combine chunks into one tensor for silence analysis
                                            combined_audio = torch.cat([c.squeeze() for c in mp3_buffer], dim=0)
                                            
                                            # Convert to numpy array for processing
                                            numpy_audio = combined_audio.cpu().numpy()
                                            
                                            # Convert to AudioSegment for silence detection (16-bit PCM)
                                            pcm_audio = np.clip(numpy_audio, -1, 1)
                                            pcm_audio = (pcm_audio * 32767).astype(np.int16)
                                            
                                            from pydub import AudioSegment
                                            from pydub.silence import split_on_silence
                                            
                                            audio_segment = AudioSegment(
                                                data=pcm_audio.tobytes(),
                                                sample_width=2,  # 16-bit
                                                frame_rate=24000,
                                                channels=1
                                            )
                                            
                                            # Split on silence using parameters from split_wav_into_mp3_based_on_silence.py
                                            chunks = split_on_silence(
                                                audio_segment,
                                                min_silence_len=80,     # 80ms: Minimum silence to be considered a pause
                                                silence_thresh=-28,     # -28 dBFS: Threshold for silence
                                                keep_silence=60         # 60ms: Keep some silence for natural sound
                                            )
                                            
                                            # If we found at least one valid split
                                            if len(chunks) > 1:
                                                # Convert the first chunk to MP3 and send it
                                                first_chunk = chunks[0]
                                                
                                                # Export to MP3 in memory
                                                mp3_io = io.BytesIO()
                                                first_chunk.export(mp3_io, format="mp3", bitrate="128k", parameters=["-ac", "1"])
                                                mp3_io.seek(0)
                                                mp3_data = mp3_io.read()
                                                
                                                # Send the MP3 data
                                                await websocket.send_bytes(mp3_data)
                                                
                                                # Remove the processed audio from the buffer
                                                # Calculate how many samples we need to remove (first_chunk length in samples)
                                                samples_to_remove = len(first_chunk) * 24000 // 1000  # Convert ms to samples
                                                
                                                # Skip removing if we don't have enough samples
                                                if samples_to_remove > 0 and samples_to_remove < combined_audio.shape[0]:
                                                    # Create a new buffer with the remaining audio
                                                    remaining_samples = combined_audio[samples_to_remove:]
                                                    
                                                    # Clear the buffer and add remaining audio as a single chunk
                                                    mp3_buffer = [remaining_samples.unsqueeze(0)]
                                    else:  # opus
                                        # Use FFmpeg to encode to opus
                                        process = subprocess.Popen(
                                            [
                                                "ffmpeg",
                                                "-f", "s16le",      # 16-bit PCM input
                                                "-ar", "24000",     # Sample rate - XTTS uses 24kHz
                                                "-ac", "1",         # Mono
                                                "-i", "pipe:0",     # Read from stdin
                                                "-c:a", "libopus",
                                                "-b:a", "32k",
                                                "-application", "voip",
                                                "-vbr", "on",
                                                "-f", "opus",
                                                "pipe:1"            # Output to stdout
                                            ],
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE
                                        )
                                        
                                        # Convert the pytorch tensor to PCM data
                                        chunk_pcm = np.clip(chunk.squeeze().cpu().numpy(), -1, 1)
                                        chunk_pcm = (chunk_pcm * 32767).astype(np.int16)
                                        
                                        # Send the PCM data to ffmpeg and get opus data
                                        opus_data, stderr = process.communicate(input=chunk_pcm.tobytes())
                                        
                                        if process.returncode != 0:
                                            print(f"FFmpeg error: {stderr.decode('utf-8', errors='ignore')}")
                                            continue
                                        
                                        # Send the opus data
                                        await websocket.send_bytes(opus_data)
                        except StopIteration:
                            # Handle case where iterator has no chunks
                            print(f"No chunks generated for first segment: {segment}")
                            continue
                    else:
                        # For non-first segments, process as usual
                        for chunk in stream_chunks_iterator:
                            chunk_counter += 1
                            
                            # Only save the chunk to a file if saveAudioFile is True
                            if save_audio_file:
                                # If audioFormat is raw, use .raw extension instead of .mp3
                                if audio_format == "raw":
                                    chunk_filename = f"outputs/{timestamp}-{chunk_counter}.raw"
                                else:
                                    chunk_filename = f"outputs/{timestamp}-{chunk_counter}.{audio_format if audio_format != 'raw' else 'mp3'}"
                                
                                chunk_files.append(chunk_filename)
                                
                                # Start a task to save the chunk in the background
                                asyncio.create_task(
                                    save_audio_chunk_background(chunk, chunk_filename, audio_format)
                                )
                            
                            # Keep track of all chunks for later combining if needed
                            if save_audio_file:
                                all_chunks.append(chunk)
                            
                            # Stream the chunk to the client with high quality
                            if isinstance(chunk, torch.Tensor):
                                # Prepare tensor for streaming
                                chunk_audio = chunk.squeeze().unsqueeze(0).cpu()
                                
                                # Stream in the requested format
                                if audio_format == "wav":
                                    # Create in-memory WAV file
                                    wav_buffer = io.BytesIO()
                                    torchaudio.save(wav_buffer, chunk_audio, 24000, format="wav")
                                    wav_buffer.seek(0)
                                    wav_data = wav_buffer.read()
                                    
                                    # Send WAV data
                                    await websocket.send_bytes(wav_data)
                                elif audio_format == "mp3":
                                    # For MP3 format, collect audio data for silence-based splitting
                                    mp3_buffer.append(chunk)
                                    
                                    # Try to split based on silence
                                    if len(mp3_buffer) > 1:  # Need at least some audio to analyze
                                        # Combine chunks into one tensor for silence analysis
                                        combined_audio = torch.cat([c.squeeze() for c in mp3_buffer], dim=0)
                                        
                                        # Convert to numpy array for processing
                                        numpy_audio = combined_audio.cpu().numpy()
                                        
                                        # Convert to AudioSegment for silence detection (16-bit PCM)
                                        pcm_audio = np.clip(numpy_audio, -1, 1)
                                        pcm_audio = (pcm_audio * 32767).astype(np.int16)
                                        
                                        from pydub import AudioSegment
                                        from pydub.silence import split_on_silence
                                        
                                        audio_segment = AudioSegment(
                                            data=pcm_audio.tobytes(),
                                            sample_width=2,  # 16-bit
                                            frame_rate=24000,
                                            channels=1
                                        )
                                        
                                        # Split on silence using parameters from split_wav_into_mp3_based_on_silence.py
                                        chunks = split_on_silence(
                                            audio_segment,
                                            min_silence_len=80,     # 80ms: Minimum silence to be considered a pause
                                            silence_thresh=-28,     # -28 dBFS: Threshold for silence
                                            keep_silence=60         # 60ms: Keep some silence for natural sound
                                        )
                                        
                                        # If we found at least one valid split
                                        if len(chunks) > 1:
                                            # Convert the first chunk to MP3 and send it
                                            first_chunk = chunks[0]
                                            
                                            # Export to MP3 in memory
                                            mp3_io = io.BytesIO()
                                            first_chunk.export(mp3_io, format="mp3", bitrate="128k", parameters=["-ac", "1"])
                                            mp3_io.seek(0)
                                            mp3_data = mp3_io.read()
                                            
                                            # Send the MP3 data
                                            await websocket.send_bytes(mp3_data)
                                            
                                            # Remove the processed audio from the buffer
                                            # Calculate how many samples we need to remove (first_chunk length in samples)
                                            samples_to_remove = len(first_chunk) * 24000 // 1000  # Convert ms to samples
                                            
                                            # Skip removing if we don't have enough samples
                                            if samples_to_remove > 0 and samples_to_remove < combined_audio.shape[0]:
                                                # Create a new buffer with the remaining audio
                                                remaining_samples = combined_audio[samples_to_remove:]
                                                
                                                # Clear the buffer and add remaining audio as a single chunk
                                                mp3_buffer = [remaining_samples.unsqueeze(0)]
                                    else:  # opus
                                        # Use FFmpeg to encode to opus
                                        process = subprocess.Popen(
                                            [
                                                "ffmpeg",
                                                "-f", "s16le",      # 16-bit PCM input
                                                "-ar", "24000",     # Sample rate - XTTS uses 24kHz
                                                "-ac", "1",         # Mono
                                                "-i", "pipe:0",     # Read from stdin
                                                "-c:a", "libopus",
                                                "-b:a", "32k",
                                                "-application", "voip",
                                                "-vbr", "on",
                                                "-f", "opus",
                                                "pipe:1"            # Output to stdout
                                            ],
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE
                                        )
                                        
                                        # Convert the pytorch tensor to PCM data
                                        chunk_pcm = np.clip(chunk.squeeze().cpu().numpy(), -1, 1)
                                        chunk_pcm = (chunk_pcm * 32767).astype(np.int16)
                                        
                                        # Send the PCM data to ffmpeg and get opus data
                                        opus_data, stderr = process.communicate(input=chunk_pcm.tobytes())
                                        
                                        if process.returncode != 0:
                                            print(f"FFmpeg error: {stderr.decode('utf-8', errors='ignore')}")
                                            continue
                                        
                                        # Send the opus data
                                        await websocket.send_bytes(opus_data)
                
                # Only combine and save audio files if saveAudioFile is True
                if save_audio_file:
                    # Full audio file path (for raw format, only save the final mp3)
                    if audio_format == "raw":
                        full_audio_path = f"outputs/{timestamp}-full.mp3"
                        # For raw format, we don't need to combine chunk files as they weren't saved individually
                        if len(raw_buffer) > 0:
                            # Save the combined raw buffer to MP3 directly
                            # This will be done only once after all chunks are processed
                            all_audio = torch.cat([chunk.squeeze() for chunk in raw_buffer], dim=0)
                            combined_audio = all_audio.cpu().numpy()
                            
                            # Convert and save as MP3
                            try:
                                # Create MP3 file
                                combined_pcm = np.clip(combined_audio, -1, 1)
                                combined_pcm = (combined_pcm * 32767).astype(np.int16)
                                
                                # Save using FFmpeg with high quality settings
                                process = subprocess.Popen(
                                    [
                                        "ffmpeg",
                                        "-f", "s16le",
                                        "-ar", "24000",
                                        "-ac", "1",
                                        "-i", "pipe:0",
                                        "-c:a", "libmp3lame",
                                        "-b:a", "128k",
                                        "-q:a", "2",
                                        "-joint_stereo", "0",
                                        full_audio_path,
                                        "-y"
                                    ],
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE
                                )
                                _, stderr = process.communicate(input=combined_pcm.tobytes())
                                
                                if process.returncode == 0:
                                    print(f"Successfully saved combined raw audio to MP3: {full_audio_path}")
                                else:
                                    print(f"Error saving combined raw audio: {stderr.decode('utf-8', errors='ignore')}")
                            except Exception as e:
                                print(f"Exception saving combined raw audio: {e}")
                        
                        # For raw format, no need to run the combine task since we did it directly
                        combine_task = None
                    else:
                        # For other formats, proceed with normal combination
                        full_audio_path = f"outputs/{timestamp}-full.{audio_format}"
                        # Start a task to combine all chunk files in the background
                        combine_task = asyncio.create_task(
                            combine_audio_chunks_background(chunk_files, full_audio_path, audio_format)
                        )
                
                # Only save info log if saveLog is true
                if save_log:
                    # Create a text file with information about the chunked generation
                    info_path = f"outputs/info_{timestamp}.txt"
                    asyncio.create_task(
                        save_tts_info_background(text, language, None, len(chunk_files), info_path, chunk_files, full_audio_path)
                    )
                
                # Wait for the combination task to complete before scheduling cleanup
                try:
                    # Wait for combine task with a reasonable timeout
                    if combine_task is not None:  # Only wait if the task was created
                        await asyncio.wait_for(combine_task, timeout=30.0)
                        print(f"Audio combination completed for {full_audio_path}")
                    
                    # Print the "Time to first chunk" information again before cleanup
                    print(f"Time to first chunk: {first_chunk_time:.2f} ms")
                    
                    # Schedule WAV cleanup without waiting for it to complete
                    print("Starting cleanup process...")
                    asyncio.create_task(async_clean_wav_files())
                except asyncio.TimeoutError:
                    print(f"Warning: Audio combination timed out for {full_audio_path}")
                except Exception as e:
                    print(f"Error waiting for audio combination: {e}")
                else:
                    # Print the "Time to first chunk" information again before cleanup
                    print(f"Time to first chunk: {first_chunk_time:.2f} ms")
                    
                    # If we're not saving audio files, still run the cleanup to remove older files
                    print("Starting cleanup process...")
                    asyncio.create_task(async_clean_wav_files())
                
                # Send an empty chunk to signal completion
                await websocket.send_bytes(b'')
                
                # Send "EOT" text message to signal end of transmission
                await websocket.send_text("EOT")
                
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

@app.websocket("/aqs-tts")
async def audio_queue_service_endpoint_streaming(websocket: WebSocket, api_key: Optional[str] = Query(None)):
    print("WebSocket Streaming TTS connection attempt")
    
    await websocket.accept()
    try:
        # Extract API key from query parameters first
        if not api_key:
            # Extract API key from headers (case-insensitive) as fallback
            headers = dict(websocket.headers)
            print(f"WebSocket headers received: {headers}")
            
            # Find API key in headers (case-insensitive)
            api_key = None
            
            for key, value in headers.items():
                key_lower = key.lower()
                if key_lower == API_KEY_NAME.lower():
                    api_key = value
        
        # For demo purposes, use a default API key if none provided
        if not api_key:
            api_key = "sk-1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t"  # Use one of your valid API keys
            
        print(f"API key extracted: {api_key}")
        
        # Check if the API key is valid
        if not api_key or api_key not in API_KEYS:
            print(f"Invalid API key: {api_key}")
            error = ERROR_CODES["INVALID_API_KEY"]
            await websocket.send_json({"errorCode": error["errorCode"], "message": error["message"]})
            await websocket.close(1008)  # Policy violation close code
            return
            
        print(f"Valid API key from user: {API_KEYS[api_key]['user']}")
        
        while True:
            # Receive message as text
            data = await websocket.receive_text()
            
            # Check if the received data is valid JSON
            try:
                # Try to parse as JSON
                message = json.loads(data)
                
                # Extract required parameters from the JSON
                text = message.get("text", "")
                language = message.get("language", "en").lower()
                # Handle "zh" language code - convert it to "zh-cn"（Changing zh to zh-cn is intentional, don'te delete the code）
                if language == "zh":
                    language = "zh-cn"
                    print("Converted language code 'zh' to 'zh-cn'")
                audio_format = message.get("audioFormat", "opus").lower()
                save_log = message.get("saveLog", False)
                # Added the new saveAudioFile parameter
                save_audio_file = message.get("saveAudioFile", False)
                # Get speakerId parameter
                speaker_id = message.get("speakerId", None)
                # Add parameter for speed - default to 1.0
                speed = message.get("speed", 1.0)
                
                # Determine reference audio based on speakerId
                if speaker_id is not None:
                    # Try to get the reference audio from the dictionary
                    reference_audio = SPEAKER_IDS.get(speaker_id, DEFAULT_SPEAKER_AUDIO)
                else:
                    # Use default speaker if speakerId not provided
                    reference_audio = DEFAULT_SPEAKER_AUDIO
                
                # Convert saveLog to boolean if it's a string
                if isinstance(save_log, str):
                    save_log = save_log.lower() == "true"
                
                # Convert saveAudioFile to boolean if it's a string
                if isinstance(save_audio_file, str):
                    save_audio_file = save_audio_file.lower() == "true"
                
                # Convert speed to float if it's a string
                if isinstance(speed, str):
                    try:
                        speed = round(speed, 1)
                    except ValueError:
                        # If conversion fails, use default
                        speed = 1.0

                # Adjust speed for speakerId = 1 and language = en
                if speaker_id == 1 and language == "en":
                    original_speed = speed
                    speed = max(0.1, speed - 0.2)  # Ensure speed doesn't go below 0.1
                    print(f"Adjusted speed from {original_speed} to {speed} for speakerId=1 and language=en")
                
                # Check if there's a config flag in the message and skip text processing
                if message.get("config", False):
                    # This is a configuration message, not a text to process
                    print("Received configuration message, skipping TTS processing")
                    continue
                
            except json.JSONDecodeError:
                # If not valid JSON, return error
                print("Error: Invalid JSON format received")
                error = ERROR_CODES["TTS_GENERATION_ERROR"]
                await websocket.send_json({
                    "errorCode": error["errorCode"], 
                    "message": "Invalid JSON format. Expected format: {\"language\": \"en\", \"audioFormat\": \"opus\", \"saveLog\": false, \"saveAudioFile\": false, \"speakerId\": 1, \"text\": \"Your text here\"}"
                })
                continue
            
            # Check if text is empty or only whitespace
            if not text or text.isspace():
                print("Error: Empty text received")
                error = ERROR_CODES["TTS_GENERATION_ERROR"]
                await websocket.send_json({
                    "errorCode": error["errorCode"], 
                    "message": "Empty text received. Please provide text to convert to speech."
                })
                continue
            
            if text.startswith('[') and text.endswith(']'):
                text = text.strip('[]').strip('"\'')
            
            print(f"Processing text: {text} with language: {language}, format: {audio_format}, saveAudioFile: {save_audio_file}")
            
            # Validate audio format
            if audio_format not in ["wav", "opus", "mp3", "raw"]:
                error = ERROR_CODES["INVALID_AUDIO_FORMAT"]
                await websocket.send_json({"errorCode": error["errorCode"], "message": error["message"]})
                continue
            
            try:
                # Validate language again in case it was changed in the message
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
                
                # Check if the streaming model is available
                if global_streaming_model is None:
                    error = ERROR_CODES["TTS_GENERATION_ERROR"]
                    await websocket.send_json({
                        "errorCode": error["errorCode"],
                        "message": "Streaming TTS model not initialized"
                    })
                    continue
                
                # Use XTTS v2 model to generate audio streams
                timestamp = int(time.time() * 1000)  # Millisecond timestamp
                
                # Create a cache key based on the reference audio path
                cache_key = reference_audio
                
                # Check if we have cached latents for this reference audio
                start_time = time.time()
                if cache_key in global_cached_latents:
                    print("Using cached speaker latents...")
                    gpt_cond_latent, speaker_embedding = global_cached_latents[cache_key]
                    cached_time = (time.time() - start_time) * 1000
                    print(f"Retrieved cached speaker latents in {cached_time:.2f} ms")
                else:
                    # Compute speaker latents if not cached
                    print("Computing speaker latents (not cached)...")
                    with torch.inference_mode():
                        gpt_cond_latent, speaker_embedding = global_streaming_model.get_conditioning_latents(audio_path=[reference_audio])
                    
                    # Cache the computed latents for future use
                    global_cached_latents[cache_key] = (gpt_cond_latent, speaker_embedding)
                    compute_time = (time.time() - start_time) * 1000
                    print(f"Speaker latents computed and cached in {compute_time:.2f} ms")
                
                # Split text by punctuation for better TTS quality
                # Define punctuation for splitting
                punctuation_markers = ['.', '!', '?', ';', ',', ':', '。', '！', '、', '？', '；', '，', '：']
                
                # Function to split text by punctuation while keeping the punctuation
                def split_by_punctuation(text):
                    segments = []
                    current_segment = ""
                    
                    for char in text:
                        current_segment += char
                        if char in punctuation_markers:
                            if current_segment.strip():  # Only add non-empty segments
                                segments.append(current_segment.strip())
                            current_segment = ""
                    
                    # Add any remaining text
                    if current_segment.strip():
                        segments.append(current_segment.strip())
                    
                    # If we have no splits (no punctuation in text), use the whole text
                    if not segments:
                        segments = [text]
                    
                    # For Chinese text, ensure first segment isn't too long for fast response
                    if detected_lang == "zh-cn" and segments and len(segments[0]) > 25:
                        # Extract a shorter first segment if it's Chinese and too long
                        # This helps get the first audio chunk to the client faster
                        first_part = segments[0][:25]
                        rest_part = segments[0][25:]
                        segments[0] = first_part
                        # Only insert the rest if it's not empty
                        if rest_part.strip():
                            segments.insert(1, rest_part)
                    
                    # Combine very short segments with the next segment
                    combined_segments = []
                    current_combined = ""
                    
                    for segment in segments:
                        # If current segment is short (less than 5 chars) or current_combined is empty
                        if len(segment) < 5 or not current_combined:
                            current_combined += " " + segment if current_combined else segment
                        else:
                            combined_segments.append(current_combined)
                            current_combined = segment
                    
                    # Add the last combined segment if it exists
                    if current_combined:
                        combined_segments.append(current_combined)
                    
                    return combined_segments
                
                # Split the text
                text_segments = split_by_punctuation(text)
                print(f"Split text into {len(text_segments)} segments by punctuation")
                
                # Start streaming inference
                print("Starting streaming inference on text segments...")
                chunk_files = []  # List to track chunk audio files
                all_chunks = []  # To collect all chunks for combining at the end if saving
                
                # Import torchaudio for high-quality audio processing
                import torchaudio
                import io
                
                # Set up streaming parameters
                stream_chunk_size = 10  # Larger chunks for better continuity
                overlap_wav_len = 3072  # Overlap for smoother transitions
                
                print(f"Using stream_chunk_size={stream_chunk_size}, overlap_wav_len={overlap_wav_len}")
                
                first_chunk = True
                first_chunk_time = 0
                chunk_counter = 0
                
                # For raw format, we'll collect all audio chunks in memory before converting to MP3
                raw_buffer = []
                
                # For MP3 format, we'll collect all audio chunks in this buffer before silence-based splitting
                mp3_buffer = []
                
                # Initialize combine_task to None to avoid reference errors
                combine_task = None
                
                # Process each text segment
                for segment_idx, segment in enumerate(text_segments):
                    print(f"Processing segment {segment_idx+1}/{len(text_segments)}: {segment[:50]}{'...' if len(segment) > 50 else ''}")
                    
                    # Process this segment through the streaming model
                    stream_chunks_iterator = global_streaming_model.inference_stream(
                        text=segment,
                        language=language,
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        stream_chunk_size=stream_chunk_size,
                        overlap_wav_len=overlap_wav_len,
                        temperature=0.1,
                        length_penalty=1.0,
                        repetition_penalty=90.0,
                        top_k=50,
                        speed=speed,
                        enable_text_splitting=True
                    )
                    
                    # For first segment only, get and send the first chunk immediately for fast response
                    if segment_idx == 0:
                        try:
                            # Get just the first chunk
                            first_audio_chunk = next(stream_chunks_iterator)
                            chunk_counter += 1
                            
                            # Track timing info for first chunk
                            chunk_time = (time.time() - start_time) * 1000  # Time in milliseconds
                            first_chunk_time = chunk_time
                            print(f"Time to first chunk: {first_chunk_time:.2f} ms")
                            first_chunk = False
                            
                            # Only save the chunk to a file if saveAudioFile is True
                            if save_audio_file:
                                # If audioFormat is raw, use .raw extension instead of .mp3
                                if audio_format == "raw":
                                    chunk_filename = f"outputs/{timestamp}-{chunk_counter}.raw"
                                else:
                                    chunk_filename = f"outputs/{timestamp}-{chunk_counter}.{audio_format}"
                                
                                chunk_files.append(chunk_filename)
                                
                                # Start a task to save the chunk in the background
                                asyncio.create_task(
                                    save_audio_chunk_background(first_audio_chunk, chunk_filename, audio_format)
                                )
                            
                            # Keep track of all chunks for later combining if needed
                            if save_audio_file or audio_format == "raw":
                                all_chunks.append(first_audio_chunk)
                            
                            # For "raw" format, collect audio chunks but don't send yet
                            if audio_format == "raw":
                                # Add to raw buffer without conversion
                                raw_buffer.append(first_audio_chunk)
                            else:
                                # Stream the chunk to the client with high quality
                                if isinstance(first_audio_chunk, torch.Tensor):
                                    # Prepare tensor for streaming
                                    chunk_audio = first_audio_chunk.squeeze().unsqueeze(0).cpu()
                                    
                                    # Stream in the requested format
                                    if audio_format == "wav":
                                        # Create in-memory WAV file
                                        wav_buffer = io.BytesIO()
                                        torchaudio.save(wav_buffer, chunk_audio, 24000, format="wav")
                                        wav_buffer.seek(0)
                                        wav_data = wav_buffer.read()
                                        
                                        # Send WAV data - first response goes out immediately
                                        await websocket.send_bytes(wav_data)
                                    elif audio_format == "mp3":
                                        # For MP3 format, collect audio data for silence-based splitting
                                        print(f"MP3 format: Adding first chunk to buffer for silence-based splitting")
                                        mp3_buffer.append(first_audio_chunk)
                                        
                                        # Try to split based on silence
                                        if len(mp3_buffer) > 1:  # Need at least some audio to analyze
                                            print(f"MP3 buffer has {len(mp3_buffer)} chunks, attempting silence-based splitting")
                                            
                                            # Combine chunks into one tensor for silence analysis
                                            print("Combining audio chunks into single tensor for analysis")
                                            combined_audio = torch.cat([c.squeeze() for c in mp3_buffer], dim=0)
                                            print(f"Combined audio shape: {combined_audio.shape}")
                                            
                                            # Convert to numpy array for processing
                                            print("Converting tensor to numpy array")
                                            numpy_audio = combined_audio.cpu().numpy()
                                            print(f"Numpy audio shape: {numpy_audio.shape}, dtype: {numpy_audio.dtype}")
                                            
                                            # Convert to AudioSegment for silence detection (16-bit PCM)
                                            print("Converting to 16-bit PCM for silence detection")
                                            pcm_audio = np.clip(numpy_audio, -1, 1)
                                            pcm_audio = (pcm_audio * 32767).astype(np.int16)
                                            print(f"PCM audio shape: {pcm_audio.shape}, dtype: {pcm_audio.dtype}")
                                            
                                            from pydub import AudioSegment
                                            from pydub.silence import split_on_silence
                                            
                                            print("Creating AudioSegment object for silence detection")
                                            audio_segment = AudioSegment(
                                                data=pcm_audio.tobytes(),
                                                sample_width=2,  # 16-bit
                                                frame_rate=24000,
                                                channels=1
                                            )
                                            print(f"AudioSegment created: {len(audio_segment)}ms duration")
                                            
                                            # Split on silence using parameters from split_wav_into_mp3_based_on_silence.py
                                            print("Splitting audio on silence with parameters: min_silence_len=80ms, silence_thresh=-28dBFS, keep_silence=60ms")
                                            chunks = split_on_silence(
                                                audio_segment,
                                                min_silence_len=80,     # 80ms: Minimum silence to be considered a pause
                                                silence_thresh=-28,     # -28 dBFS: Threshold for silence
                                                keep_silence=60         # 60ms: Keep some silence for natural sound
                                            )
                                            print(f"Split result: {len(chunks)} audio chunks detected")
                                            
                                            # If we found at least one valid split
                                            if len(chunks) > 1:
                                                print(f"Multiple chunks detected, processing first chunk of {len(chunks[0])}ms")
                                                # Convert the first chunk to MP3 and send it
                                                first_chunk = chunks[0]
                                                
                                                # Export to MP3 in memory
                                                print("Exporting first chunk to MP3 format in memory")
                                                mp3_io = io.BytesIO()
                                                first_chunk.export(mp3_io, format="mp3", bitrate="128k", parameters=["-ac", "1"])
                                                mp3_io.seek(0)
                                                mp3_data = mp3_io.read()
                                                print(f"MP3 data size: {len(mp3_data)} bytes")
                                                
                                                # Send the MP3 data
                                                print("Sending MP3 data to client")
                                                await websocket.send_bytes(mp3_data)
                                                print("MP3 data sent successfully")
                                                
                                                # Remove the processed audio from the buffer
                                                # Calculate how many samples we need to remove (first_chunk length in samples)
                                                samples_to_remove = len(first_chunk) * 24000 // 1000  # Convert ms to samples
                                                print(f"Calculating samples to remove: {samples_to_remove} samples ({len(first_chunk)}ms)")
                                                
                                                # Skip removing if we don't have enough samples
                                                if samples_to_remove > 0 and samples_to_remove < combined_audio.shape[0]:
                                                    print(f"Removing {samples_to_remove} samples from buffer")
                                                    # Create a new buffer with the remaining audio
                                                    remaining_samples = combined_audio[samples_to_remove:]
                                                    print(f"Remaining samples shape: {remaining_samples.shape}")
                                                    
                                                    # Clear the buffer and add remaining audio as a single chunk
                                                    print("Updating MP3 buffer with remaining audio")
                                                    mp3_buffer = [remaining_samples.unsqueeze(0)]
                                                    print(f"Updated MP3 buffer has {len(mp3_buffer)} chunks")
                                                else:
                                                    print(f"Cannot remove samples: samples_to_remove={samples_to_remove}, combined_audio.shape={combined_audio.shape}")
                                            else:
                                                print(f"No valid splits found, keeping audio in buffer for next iteration")
                                        else:
                                            print(f"MP3 buffer has only {len(mp3_buffer)} chunk(s), need more for analysis")
                                    else:  # opus
                                        # Use FFmpeg to encode to opus
                                        process = subprocess.Popen(
                                            [
                                                "ffmpeg",
                                                "-f", "s16le",      # 16-bit PCM input
                                                "-ar", "24000",     # Sample rate - XTTS uses 24kHz
                                                "-ac", "1",         # Mono
                                                "-i", "pipe:0",     # Read from stdin
                                                "-c:a", "libopus",
                                                "-b:a", "32k",
                                                "-application", "voip",
                                                "-vbr", "on",
                                                "-f", "opus",
                                                "pipe:1"            # Output to stdout
                                            ],
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE
                                        )
                                        
                                        # Convert the pytorch tensor to PCM data
                                        chunk_pcm = np.clip(first_audio_chunk.squeeze().cpu().numpy(), -1, 1)
                                        chunk_pcm = (chunk_pcm * 32767).astype(np.int16)
                                        
                                        # Send the PCM data to ffmpeg and get opus data
                                        opus_data, stderr = process.communicate(input=chunk_pcm.tobytes())
                                        
                                        if process.returncode != 0:
                                            print(f"FFmpeg error: {stderr.decode('utf-8', errors='ignore')}")
                                            continue
                                        
                                        # Send the opus data
                                        await websocket.send_bytes(opus_data)
                        except StopIteration:
                            # Handle case where iterator has no chunks
                            print(f"No chunks generated for first segment: {segment}")
                            continue
                    else:
                        # For non-first segments, process as usual
                        for chunk in stream_chunks_iterator:
                            chunk_counter += 1
                            
                            # Only save the chunk to a file if saveAudioFile is True
                            if save_audio_file:
                                # If audioFormat is raw, use .raw extension instead of .mp3
                                if audio_format == "raw":
                                    chunk_filename = f"outputs/{timestamp}-{chunk_counter}.raw"
                                else:
                                    chunk_filename = f"outputs/{timestamp}-{chunk_counter}.{audio_format}"
                                
                                chunk_files.append(chunk_filename)
                                
                                # Start a task to save the chunk in the background
                                asyncio.create_task(
                                    save_audio_chunk_background(chunk, chunk_filename, audio_format)
                                )
                            
                            # Keep track of all chunks for later combining if needed
                            if save_audio_file or audio_format == "raw":
                                all_chunks.append(chunk)
                            
                            # Stream the chunk to the client with high quality
                            if isinstance(chunk, torch.Tensor):
                                # Prepare tensor for streaming
                                chunk_audio = chunk.squeeze().unsqueeze(0).cpu()
                                
                                # Stream in the requested format
                                if audio_format == "wav":
                                    # Create in-memory WAV file
                                    wav_buffer = io.BytesIO()
                                    torchaudio.save(wav_buffer, chunk_audio, 24000, format="wav")
                                    wav_buffer.seek(0)
                                    wav_data = wav_buffer.read()
                                    
                                    # Send WAV data
                                    await websocket.send_bytes(wav_data)
                                elif audio_format == "mp3":
                                    # For MP3 format, collect audio data for silence-based splitting
                                    mp3_buffer.append(chunk)
                                    
                                    # Try to split based on silence
                                    if len(mp3_buffer) > 1:  # Need at least some audio to analyze
                                        # Combine chunks into one tensor for silence analysis
                                        combined_audio = torch.cat([c.squeeze() for c in mp3_buffer], dim=0)
                                        
                                        # Convert to numpy array for processing
                                        numpy_audio = combined_audio.cpu().numpy()
                                        
                                        # Convert to AudioSegment for silence detection (16-bit PCM)
                                        pcm_audio = np.clip(numpy_audio, -1, 1)
                                        pcm_audio = (pcm_audio * 32767).astype(np.int16)
                                        
                                        from pydub import AudioSegment
                                        from pydub.silence import split_on_silence
                                        
                                        audio_segment = AudioSegment(
                                            data=pcm_audio.tobytes(),
                                            sample_width=2,  # 16-bit
                                            frame_rate=24000,
                                            channels=1
                                        )
                                        
                                        # Split on silence using parameters from split_wav_into_mp3_based_on_silence.py
                                        chunks = split_on_silence(
                                            audio_segment,
                                            min_silence_len=80,     # 80ms: Minimum silence to be considered a pause
                                            silence_thresh=-28,     # -28 dBFS: Threshold for silence
                                            keep_silence=60         # 60ms: Keep some silence for natural sound
                                        )
                                        
                                        # If we found at least one valid split
                                        if len(chunks) > 1:
                                            # Convert the first chunk to MP3 and send it
                                            first_chunk = chunks[0]
                                            
                                            # Export to MP3 in memory
                                            mp3_io = io.BytesIO()
                                            first_chunk.export(mp3_io, format="mp3", bitrate="128k", parameters=["-ac", "1"])
                                            mp3_io.seek(0)
                                            mp3_data = mp3_io.read()
                                            
                                            # Send the MP3 data
                                            await websocket.send_bytes(mp3_data)
                                            
                                            # Remove the processed audio from the buffer
                                            # Calculate how many samples we need to remove (first_chunk length in samples)
                                            samples_to_remove = len(first_chunk) * 24000 // 1000  # Convert ms to samples
                                            
                                            # Skip removing if we don't have enough samples
                                            if samples_to_remove > 0 and samples_to_remove < combined_audio.shape[0]:
                                                # Create a new buffer with the remaining audio
                                                remaining_samples = combined_audio[samples_to_remove:]
                                                
                                                # Clear the buffer and add remaining audio as a single chunk
                                                mp3_buffer = [remaining_samples.unsqueeze(0)]
                                    else:  # opus
                                        # Use FFmpeg to encode to opus
                                        process = subprocess.Popen(
                                            [
                                                "ffmpeg",
                                                "-f", "s16le",      # 16-bit PCM input
                                                "-ar", "24000",     # Sample rate - XTTS uses 24kHz
                                                "-ac", "1",         # Mono
                                                "-i", "pipe:0",     # Read from stdin
                                                "-c:a", "libopus",
                                                "-b:a", "32k",
                                                "-application", "voip",
                                                "-vbr", "on",
                                                "-f", "opus",
                                                "pipe:1"            # Output to stdout
                                            ],
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE
                                        )
                                        
                                        # Convert the pytorch tensor to PCM data
                                        chunk_pcm = np.clip(chunk.squeeze().cpu().numpy(), -1, 1)
                                        chunk_pcm = (chunk_pcm * 32767).astype(np.int16)
                                        
                                        # Send the PCM data to ffmpeg and get opus data
                                        opus_data, stderr = process.communicate(input=chunk_pcm.tobytes())
                                        
                                        if process.returncode != 0:
                                            print(f"FFmpeg error: {stderr.decode('utf-8', errors='ignore')}")
                                            continue
                                        
                                        # Send the opus data
                                        await websocket.send_bytes(opus_data)
                
                # Only combine and save audio files if saveAudioFile is True
                if save_audio_file:
                    # Full audio file path (for raw format, only save the final mp3)
                    if audio_format == "raw":
                        full_audio_path = f"outputs/{timestamp}-full.mp3"
                        # For raw format, we don't need to combine chunk files as they weren't saved individually
                        if len(raw_buffer) > 0:
                            # Save the combined raw buffer to MP3 directly
                            # This will be done only once after all chunks are processed
                            all_audio = torch.cat([chunk.squeeze() for chunk in raw_buffer], dim=0)
                            combined_audio = all_audio.cpu().numpy()
                            
                            # Convert and save as MP3
                            try:
                                # Create MP3 file
                                combined_pcm = np.clip(combined_audio, -1, 1)
                                combined_pcm = (combined_pcm * 32767).astype(np.int16)
                                
                                # Save using FFmpeg with high quality settings
                                process = subprocess.Popen(
                                    [
                                        "ffmpeg",
                                        "-f", "s16le",
                                        "-ar", "24000",
                                        "-ac", "1",
                                        "-i", "pipe:0",
                                        "-c:a", "libmp3lame",
                                        "-b:a", "128k",
                                        "-q:a", "2",
                                        "-joint_stereo", "0",
                                        full_audio_path,
                                        "-y"
                                    ],
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE
                                )
                                _, stderr = process.communicate(input=combined_pcm.tobytes())
                                
                                if process.returncode == 0:
                                    print(f"Successfully saved combined raw audio to MP3: {full_audio_path}")
                                else:
                                    print(f"Error saving combined raw audio: {stderr.decode('utf-8', errors='ignore')}")
                            except Exception as e:
                                print(f"Exception saving combined raw audio: {e}")
                        
                        # For raw format, no need to run the combine task since we did it directly
                        combine_task = None
                    else:
                        # For other formats, proceed with normal combination
                        full_audio_path = f"outputs/{timestamp}-full.{audio_format}"
                        # Start a task to combine all chunk files in the background
                        combine_task = asyncio.create_task(
                            combine_audio_chunks_background(chunk_files, full_audio_path, audio_format)
                        )
                
                # Only save info log if saveLog is true
                if save_log:
                    # Create a text file with information about the chunked generation
                    info_path = f"outputs/info_{timestamp}.txt"
                    asyncio.create_task(
                        save_tts_info_background(text, language, None, len(chunk_files), info_path, chunk_files, full_audio_path)
                    )
                
                # Wait for the combination task to complete before scheduling cleanup
                try:
                    # Wait for combine task with a reasonable timeout
                    if combine_task is not None:  # Only wait if the task was created
                        await asyncio.wait_for(combine_task, timeout=30.0)
                        print(f"Audio combination completed for {full_audio_path}")
                    
                    # Print the "Time to first chunk" information again before cleanup
                    print(f"Time to first chunk: {first_chunk_time:.2f} ms")
                    
                    # Schedule WAV cleanup without waiting for it to complete
                    print("Starting cleanup process...")
                    asyncio.create_task(async_clean_wav_files())
                except asyncio.TimeoutError:
                    print(f"Warning: Audio combination timed out for {full_audio_path}")
                except Exception as e:
                    print(f"Error waiting for audio combination: {e}")
                else:
                    # Print the "Time to first chunk" information again before cleanup
                    print(f"Time to first chunk: {first_chunk_time:.2f} ms")
                    
                    # If we're not saving audio files, still run the cleanup to remove older files
                    print("Starting cleanup process...")
                    asyncio.create_task(async_clean_wav_files())
                
                # Send an empty chunk to signal completion
                await websocket.send_bytes(b'')
                
                # Send "EOT" text message to signal end of transmission
                #await websocket.send_text("EOT")
                
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

@app.websocket("/raw-stream-tts")
async def raw_stream_tts_endpoint(websocket: WebSocket, api_key: Optional[str] = Query(None)):
    """Stream TTS audio directly to client without saving files, optimized for iOS clients.
    Client can immediately play the first chunk and seamlessly play subsequent chunks without gaps."""
    
    await websocket.accept()
    
    try:
        # Receive and parse the request
        request_data = await websocket.receive_json()
        
        # Extract parameters
        text = request_data.get("text", "")
        if not text:
            await websocket.send_json({"error": "Text is required"})
            return
            
        language = request_data.get("language", "en")
        reference_audio_path = request_data.get("referenceAudio", "reference_samples/andy-liu-en-1.wav")
        
        # Optional parameters with defaults
        temperature = float(request_data.get("temperature", 0.1))
        length_penalty = float(request_data.get("lengthPenalty", 1.0))
        repetition_penalty = float(request_data.get("repetitionPenalty", 90.0))
        top_k = int(request_data.get("topK", 50))
        speed = float(request_data.get("speed", 1.0))
        enable_text_splitting = bool(request_data.get("enableTextSplitting", True))
        stream_chunk_size = int(request_data.get("streamChunkSize", 10))
        overlap_wav_len = int(request_data.get("overlapWavLen", 1024))
        
        # Get or compute speaker conditioning 
        gpt_cond_latent, speaker_embedding = global_streaming_model.get_conditioning_latents(audio_path=[reference_audio_path])
        
        # Stream audio chunks directly to client
        chunks = global_streaming_model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=stream_chunk_size,
            overlap_wav_len=overlap_wav_len,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            speed=speed,
            enable_text_splitting=enable_text_splitting
        )
        
        for i, chunk in enumerate(chunks):
            # Convert tensor to bytes
            chunk_audio = chunk.squeeze().unsqueeze(0).cpu()
            
            # Convert to opus format bytes
            opus_bytes = convert_tensor_to_opus_bytes(chunk_audio, sample_rate=24000)
            
            # Send the audio chunk directly
            await websocket.send_bytes(opus_bytes)
            
            # Log progress (server-side only)
            print(f"Sent chunk {i+1}")
        
        # No completion message and no websocket closing
        
    except WebSocketDisconnect:
        print("Client disconnected from raw-stream-tts endpoint")
    except Exception as e:
        error_message = f"Error in raw-stream-tts: {str(e)}"
        print(error_message)
        try:
            await websocket.send_json({"error": error_message})
        except:
            pass

def convert_tensor_to_opus_bytes(tensor, sample_rate=24000):
    """Convert audio tensor to opus format bytes for direct streaming"""
    import io
    import torchaudio
    
    buffer = io.BytesIO()
    torchaudio.save(buffer, tensor, sample_rate, format="opus")
    buffer.seek(0)
    return buffer.read()

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "errorCode": 0,  # 0 typically indicates success
        "message": "Server is running"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9002)

import warnings
warnings.filterwarnings("ignore", message=".*attention mask.*")
