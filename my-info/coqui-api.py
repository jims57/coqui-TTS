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

# Add this near the top of the file with other global variables
# Default number of words per TTS segment for streaming
DEFAULT_WORDS_PER_SEGMENT = 3

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
    "LANGUAGE_MISMATCH": {"errorCode": 6002, "message": "Text language doesn't match requested language"},
    "INVALID_AUDIO_FORMAT": {"errorCode": 6003, "message": "Invalid audio format. Supported formats: wav, opus"}
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
        language = "en"  # Default language
        audio_format = "opus"  # Default audio format
        words_per_segment = DEFAULT_WORDS_PER_SEGMENT  # Default words per segment
        
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
                
        print(f"API key extracted from headers: {api_key}")
        print(f"Language parameter from headers: {language}")
        print(f"Audio format parameter from headers: {audio_format}")
        print(f"Words per segment parameter: {words_per_segment}")
        
        # Validate audio format
        if audio_format not in ["wav", "opus"]:
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
                
                # Validate audio format again in case it was changed in the message
                if audio_format not in ["wav", "opus"]:
                    error = ERROR_CODES["INVALID_AUDIO_FORMAT"]
                    await websocket.send_json({"errorCode": error["errorCode"], "message": error["message"]})
                    continue
            except json.JSONDecodeError:
                # If not JSON, assume it's just text
                text = data
            
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
                
                # Process each segment sequentially
                all_audio_chunks = []
                
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
                        timestamp = int(time.time())
                        fallback_wav_path = f"outputs/fallback_{timestamp}_{i}.wav"
                        
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
                
                # Save the combined audio in the background
                timestamp = int(time.time())
                wav_path = f"outputs/output_{timestamp}.wav"
                opus_path = f"outputs/output_{timestamp}.opus"
                
                # Start a task to save the file in the background if needed
                # This can be useful for logging/debugging
                if len(segments) > 1:
                    # Generate the complete audio for saving in the background
                    try:
                        if language == "en":
                            # Use FastPitch for English text
                            with torch.inference_mode():
                                wav_array = global_tts.tts(text=text)
                        else:
                            # Use XTTS v2 for other supported languages
                            with torch.inference_mode():
                                wav_array = global_chinese_tts.tts(
                                    text=text,
                                    speaker_wav=sample_wav_path,
                                    language=language
                                )
                        
                        # Ensure wav is a properly formatted numpy array
                        wav_array = ensure_numpy_array(wav_array)
                        
                        # Start the background task with the full audio array
                        asyncio.create_task(
                            save_audio_file_background(wav_array, wav_path, opus_path, audio_format)
                        )
                    except Exception as e:
                        print(f"Error generating full audio for background save: {e}")
                        # If we can't generate the full audio, don't try to save it
                        pass
                
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
    audioFormat: Optional[str] = "opus"  # Default to opus, can be wav or opus

# Helper function to convert wav to opus using FFmpeg
def convert_wav_to_opus(wav_path, opus_path):
    try:
        # Use FFmpeg to convert WAV to Opus
        subprocess.run(
            [
                "ffmpeg", 
                "-i", wav_path,
                "-c:a", "libopus",
                "-b:a", "32k",
                "-application", "voip",
                "-vbr", "on",
                opus_path,
                "-y"  # Overwrite if exists
            ],
            check=True,
            capture_output=True
        )
        return True
    except Exception as e:
        print(f"Error converting WAV to Opus: {e}")
        return False

# Update the /tts endpoint to properly handle single-dimensional arrays
@app.post("/tts")
async def generate_audio_http(
    request: TTSRequest,
    api_key: APIKey = Depends(get_api_key)
):
    overall_start = time.time()
    validation_start = time.time()
    try:
        # Log usage for the API key
        print(f"API request from user: {API_KEYS[api_key]['user']}")
        
        text = request.text
        language = request.language.lower() if request.language else "en"
        audio_format = request.audioFormat.lower() if request.audioFormat else "opus"
        
        # Validate audio format
        if audio_format not in ["wav", "opus"]:
            error = ERROR_CODES["INVALID_AUDIO_FORMAT"]
            return {"errorCode": error["errorCode"], "message": error["message"]}
        
        if text.startswith('[') and text.endswith(']'):
            text = text.strip('[]').strip('"\'')
            
        print(f"Processing text: {text} with language: {language}, format: {audio_format}")
        
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
        
        validation_time = time.time() - validation_start
        print(f"DEBUG: Validation time: {validation_time * 1000:.2f} ms")
        
        timestamp = int(time.time())
        
        # Generate audio directly in memory
        tts_start = time.time()
        if language == "en":
            print("Using FastPitch model for English")
            with torch.inference_mode():
                wav = global_tts.tts(text=text)
        else:
            print(f"Using XTTS model with language: {language}")
            
            if not os.path.exists(sample_wav_path):
                error = ERROR_CODES["REFERENCE_AUDIO_NOT_FOUND"]
                return {"errorCode": error["errorCode"], "message": error["message"]}
                
            with torch.inference_mode():
                wav = global_chinese_tts.tts(
                    text=text,
                    speaker_wav=sample_wav_path,
                    language=language
                )
        
        tts_time = time.time() - tts_start
        print(f"DEBUG: TTS generation time: {tts_time * 1000:.2f} ms")
        
        # Normalize and convert to 16-bit PCM
        normalize_start = time.time()
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)
        normalize_time = time.time() - normalize_start
        print(f"DEBUG: Normalization time: {normalize_time * 1000:.2f} ms")
        
        # Start a background task for cleanup - don't wait for it
        cleanup_start = time.time()
        asyncio.create_task(async_clean_wav_files())
        cleanup_time = time.time() - cleanup_start
        print(f"DEBUG: Cleanup task creation time: {cleanup_time * 1000:.2f} ms")
        
        # Prepare response with minimal latency
        format_start = time.time()
        if audio_format == "wav":
            # Create WAV file in memory
            wav_io_start = time.time()
            wav_io = io.BytesIO()
            wavfile.write(wav_io, 22050, wav)
            wav_io.seek(0)
            wav_io_time = time.time() - wav_io_start
            print(f"DEBUG: WAV in-memory conversion time: {wav_io_time * 1000:.2f} ms")
            
            response_start = time.time()
            response = StreamingResponse(
                wav_io,
                media_type="audio/wav",
                headers={"Content-Disposition": f"attachment; filename=tts_output_{timestamp}.wav"}
            )
            response_time = time.time() - response_start
            print(f"DEBUG: StreamingResponse creation time (WAV): {response_time * 1000:.2f} ms")
            
        elif audio_format == "opus":
            # Start FFmpeg process
            ffmpeg_start = time.time()
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
            
            # Send wav data to ffmpeg and get opus data
            opus_data, stderr = process.communicate(input=wav.tobytes())
            ffmpeg_time = time.time() - ffmpeg_start
            print(f"DEBUG: FFmpeg opus conversion time: {ffmpeg_time * 1000:.2f} ms")
            
            if process.returncode != 0:
                print(f"FFmpeg error: {stderr.decode('utf-8', errors='ignore')}")
                error = ERROR_CODES["TTS_GENERATION_ERROR"]
                return {"errorCode": error["errorCode"], "message": f"{error['message']}: FFmpeg encoding failed"}
            
            response_start = time.time()
            response = StreamingResponse(
                io.BytesIO(opus_data),
                media_type="audio/opus",
                headers={"Content-Disposition": f"attachment; filename=tts_output_{timestamp}.opus"}
            )
            response_time = time.time() - response_start
            print(f"DEBUG: StreamingResponse creation time (Opus): {response_time * 1000:.2f} ms")
        
        format_time = time.time() - format_start
        print(f"DEBUG: Total format conversion time: {format_time * 1000:.2f} ms")
        
        total_time = time.time() - overall_start
        print(f"DEBUG: Total endpoint processing time: {total_time * 1000:.2f} ms")
        print(f"DEBUG: Time until response ready: {total_time * 1000:.2f} ms")
        
        # Confirm we're using in-memory processing and not file I/O
        print("DEBUG: Using pure in-memory audio processing, no file I/O")
        
        return response
            
    except Exception as e:
        error_time = time.time() - overall_start
        print(f"DEBUG: Error occurred after {error_time * 1000:.2f} ms")
        print(f"TTS Error: {str(e)}")
        error = ERROR_CODES["TTS_GENERATION_ERROR"]
        return {"errorCode": error["errorCode"], "message": error["message"], "details": str(e)}

# Update the ensure_numpy_array function to handle numpy.int16 and other scalar types
def ensure_numpy_array(audio_array):
    # Handle numpy scalar types (like numpy.int16)
    if isinstance(audio_array, np.number):
        return np.array([float(audio_array)])
    
    # Handle integer or scalar value case
    if isinstance(audio_array, (int, float)) or (isinstance(audio_array, np.ndarray) and audio_array.shape == ()):
        return np.array([float(audio_array)])
    
    if isinstance(audio_array, list):
        if len(audio_array) == 0:  # Empty list
            return np.array([])
        elif len(audio_array) == 1:
            # If the single element is a scalar, convert it properly
            if isinstance(audio_array[0], (int, float, np.number)):
                return np.array([float(audio_array[0])])
            return audio_array[0]
        else:
            # Check if any array is zero-dimensional or scalar
            for i, arr in enumerate(audio_array):
                if isinstance(arr, (int, float, np.number)) or (isinstance(arr, np.ndarray) and (not arr.shape or arr.size == 0)):
                    # Convert scalar/zero-dimensional arrays to 1D arrays
                    audio_array[i] = np.array([float(arr)]) if isinstance(arr, (int, float, np.number)) else np.array([])
            
            # Filter out empty arrays
            non_empty_arrays = [arr for arr in audio_array if hasattr(arr, 'size') and arr.size > 0]
            
            if not non_empty_arrays:
                return np.array([])
            elif len(non_empty_arrays) == 1:
                return non_empty_arrays[0]
            else:
                try:
                    return np.concatenate(non_empty_arrays)
                except ValueError as e:
                    # If concatenation fails, return the first non-empty array
                    print(f"Warning: Could not concatenate arrays: {e}, returning first array")
                    return non_empty_arrays[0]
    
    # If audio_array is not a numpy array or doesn't have shape attribute
    if not isinstance(audio_array, np.ndarray):
        try:
            # Try to convert to numpy array
            return np.array(audio_array, dtype=float)
        except Exception as e:
            print(f"Error converting to numpy array: {e}")
            # Return a default empty audio array as fallback
            return np.array([0.0], dtype=float)
    
    return audio_array

# Helper function to save audio files in the background
async def save_audio_file_background(wav_array, wav_path, opus_path, audio_format):
    try:
        print(f"Saving audio to {wav_path}, type: {type(wav_array)}")
        
        # Handle scalar numpy types
        if isinstance(wav_array, np.number):
            wav_array = np.array([float(wav_array)])
        
        # Ensure wav_array is a numpy array
        if not isinstance(wav_array, np.ndarray):
            if isinstance(wav_array, (list, tuple)):
                wav_array = np.array(wav_array, dtype=float)
            else:
                wav_array = np.array([float(wav_array)])
        
        # Ensure it's a 1D array
        if wav_array.ndim == 0:
            wav_array = np.array([wav_array.item()], dtype=float)
        
        # Normalize and convert to 16-bit PCM for saving
        wav_array = np.clip(wav_array, -1, 1)
        wav_array = (wav_array * 32767).astype(np.int16)
        
        print(f"Saving wav array shape: {wav_array.shape}")
        
        # Save WAV file
        sample_rate = 22050
        wavfile.write(wav_path, sample_rate, wav_array)
        
        # If opus format, convert WAV to Opus
        if audio_format == "opus":
            convert_wav_to_opus(wav_path, opus_path)
            
            # Delete the temporary WAV file
            if os.path.exists(wav_path):
                os.remove(wav_path)
                
        # Schedule cleanup after saving the file
        asyncio.create_task(async_clean_wav_files())
    except Exception as e:
        print(f"Error saving audio file: {e}")

# Update the HTTP stream endpoint
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
        audio_format = request.audioFormat.lower() if request.audioFormat else "opus"
        
        # Validate audio format
        if audio_format not in ["wav", "opus"]:
            error = ERROR_CODES["INVALID_AUDIO_FORMAT"]
            return JSONResponse(
                status_code=400,
                content={"errorCode": error["errorCode"], "message": error["message"]}
            )
        
        if text.startswith('[') and text.endswith(']'):
            text = text.strip('[]').strip('"\'')
            
        print(f"Processing text: {text} with language: {language}, format: {audio_format}")
        
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
        
        # As a failsafe, generate the wav file first and read it back
        timestamp = int(time.time())
        wav_path = f"outputs/output_{timestamp}.wav"
        opus_path = f"outputs/output_{timestamp}.opus"
        
        try:
            # Generate wav file first to avoid array handling issues
            if language == "en":
                global_tts.tts_to_file(text=text, file_path=wav_path)
            else:
                global_chinese_tts.tts_to_file(
                    text=text,
                    file_path=wav_path,
                    speaker_wav=sample_wav_path,
                    language=language
                )
            
            # Read the wav file
            sample_rate, wav_array = wavfile.read(wav_path)
            wav_array = wav_array.astype(np.float32) / 32767.0  # Convert to float [-1,1]
        except Exception as e:
            print(f"Error in tts_to_file: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "errorCode": ERROR_CODES["TTS_GENERATION_ERROR"]["errorCode"],
                    "message": ERROR_CODES["TTS_GENERATION_ERROR"]["message"],
                    "details": str(e)
                }
            )
        
        # Binary stream generator for WAV format
        async def wav_stream_generator():
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
            
            # We already have the wav_array from the file
            wav = wav_array
            
            # Normalize and convert to 16-bit PCM
            wav = np.clip(wav, -1, 1)
            wav = (wav * 32767).astype(np.int16)
            
            # Make sure wav is a 1D array
            if wav.ndim == 0:
                wav = np.array([wav.item()], dtype=np.int16)
            
            # Stream audio chunks
            chunk_size = 1024  # Samples per chunk
            total_samples = len(wav)
            
            for i in range(0, total_samples, chunk_size):
                chunk = wav[i:min(i + chunk_size, total_samples)]
                chunk_bytes = chunk.tobytes()
                data_buffer.extend(chunk_bytes)
                yield chunk_bytes
            
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
        
        # Binary stream generator for Opus format
        async def opus_stream_generator():
            # We already have the wav file, so convert to opus
            
            if not os.path.exists(wav_path):
                error = ERROR_CODES["TTS_GENERATION_ERROR"]
                raise Exception(f"{error['message']}: WAV file not found")
            
            # Convert the WAV file to Opus
            convert_wav_to_opus(wav_path, opus_path)
            
            if not os.path.exists(opus_path):
                error = ERROR_CODES["TTS_GENERATION_ERROR"]
                raise Exception(f"{error['message']}: Opus conversion failed")
            
            # Read the opus file and stream it in chunks
            with open(opus_path, 'rb') as opus_file:
                opus_data = opus_file.read()
            
            # Stream the opus data in chunks
            opus_chunk_size = 1024  # Bytes per chunk
            for i in range(0, len(opus_data), opus_chunk_size):
                opus_chunk = opus_data[i:min(i + opus_chunk_size, len(opus_data))]
                yield opus_chunk
            
            # Schedule WAV cleanup without waiting for it to complete
            asyncio.create_task(async_clean_wav_files())
        
        # Choose the appropriate generator based on format
        if audio_format == "wav":
            generator = wav_stream_generator()
            media_type = "audio/wav"
            filename = f"tts_output_{int(time.time())}.wav"
        else:  # opus
            generator = opus_stream_generator()
            media_type = "audio/opus"
            filename = f"tts_output_{int(time.time())}.opus"
        
        # Return the streaming response
        return StreamingResponse(
            generator,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
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

@app.get("/test")
async def test_endpoint():
    """
    Simple test endpoint that returns a JSON response.
    Used to measure round-trip time from client to server and back.
    """
    return {
        "status": "success",
        "timestamp": time.time(),
        "message": "API is operational"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9002)

import warnings
warnings.filterwarnings("ignore", message=".*attention mask.*")