import datetime
import builtins
import asyncio
import json
import os
import threading
import traceback
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, cast

import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from fastapi import Body, FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect, WebSocketState

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.modular.streamer import AudioStreamer
from vibevoice.text_normalization import normalize_polish

try:
    from .jobs import JobManager
except ImportError:  # uruchomione jako zwykly modul, nie jako pakiet web
    from jobs import JobManager

import copy

BASE = Path(__file__).parent
SAMPLE_RATE = 24_000


def get_timestamp():
    timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc
    ).astimezone(
        datetime.timezone(datetime.timedelta(hours=8))
    ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return timestamp

# Jeden przebieg modelu jest ograniczony oknem pozycji dekodera
# (max_position_embeddings). Dluzszy tekst dzielimy na fragmenty generowane
# kolejno tym samym glosem, a laczny czas nagrania ograniczamy do
# MAX_LENGTH_MULTIPLIER-krotnosci tego, co miesci sie w jednym przebiegu.
MAX_LENGTH_MULTIPLIER = int(os.environ.get("VIBEVOICE_MAX_LENGTH_MULTIPLIER", "5"))
CHUNK_MAX_CHARS = int(os.environ.get("VIBEVOICE_CHUNK_MAX_CHARS", "1500"))
SPEECH_TOKENS_PER_SECOND = 7.5


_POLISH_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


def should_normalize_polish(mode: str, voice: Optional[str], text: str) -> bool:
    if mode in ("off", "0", "false", "none"):
        return False
    if mode == "pl":
        return True
    return bool((voice or "").lower().startswith("pl-") or any(ch in _POLISH_CHARS for ch in text))


def prepare_text(text: str, mode: str, voice: Optional[str], phonetic: bool) -> str:
    text = (text or "").replace("’", "'").strip()
    if should_normalize_polish(mode, voice, text):
        return normalize_polish(text, phonetic=phonetic)
    return text


OUTPUT_DIR = Path(os.environ.get("VIBEVOICE_OUTPUT_DIR", BASE.parent.parent / "outputs" / "jobs"))


def split_text_for_tts(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list:
    """Dzieli tekst na fragmenty <= max_chars: po akapitach, potem po zdaniach."""
    import re

    chunks: list = []
    current = ""

    def push(piece: str) -> None:
        nonlocal current
        piece = piece.strip()
        if not piece:
            return
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current} {piece}".strip() if current else piece

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            push(paragraph)
            continue
        for sentence in re.findall(r"[^.!?…]+[.!?…]*", paragraph):
            sentence = sentence.strip()
            while len(sentence) > max_chars:
                cut = sentence.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                push(sentence[:cut])
                sentence = sentence[cut:].strip()
            push(sentence)
    if current:
        chunks.append(current)
    return chunks


class StreamingTTSService:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        inference_steps: int = 5,
    ) -> None:
        # Keep model_path as string for HuggingFace repo IDs (Path() converts / to \ on Windows)
        self.model_path = model_path
        self.inference_steps = inference_steps
        self.sample_rate = SAMPLE_RATE

        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self.default_voice_key: Optional[str] = None
        self.generation_lock = threading.Lock()
        self._voice_cache: Dict[str, Tuple[object, Path, str]] = {}

        if device == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            device = "mps"        
        if device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            device = "cpu"
        self.device = device
        self._torch_device = torch.device(device)

    def load(self) -> None:
        print(f"[startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        
        # Decide dtype & attention
        if self.device == "mps":
            load_dtype = torch.float32
            device_map = None
            attn_impl_primary = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = 'cuda'
            attn_impl_primary = "flash_attention_2"
        else:
            load_dtype = torch.float32
            device_map = 'cpu'
            attn_impl_primary = "sdpa"
        print(f"Using device: {device_map}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        # Load model
        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl_primary,
            )
            
            if self.device == "mps":
                self.model.to("mps")
        except Exception as e:
            if attn_impl_primary == 'flash_attention_2':
                print("Error loading the model. Trying to use SDPA. However, note that only flash_attention_2 has been fully tested, and using SDPA may result in lower audio quality.")
                
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=self.device,
                    attn_implementation='sdpa',
                )
                print("Load model with SDPA successfully ")
            else:
                raise e

        self.model.eval()

        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        self.voice_presets = self._load_voice_presets()
        preset_name = os.environ.get("VOICE_PRESET")
        self.default_voice_key = self._determine_voice_key(preset_name)
        self._ensure_voice_cached(self.default_voice_key)

    def _load_voice_presets(self) -> Dict[str, Path]:
        voices_dir = BASE.parent / "voices" / "streaming_model"
        if not voices_dir.exists():
            raise RuntimeError(f"Voices directory not found: {voices_dir}")

        presets: Dict[str, Path] = {}
        for pt_path in voices_dir.rglob("*.pt"):
            presets[pt_path.stem] = pt_path

        if not presets:
            raise RuntimeError(f"No voice preset (.pt) files found in {voices_dir}")

        print(f"[startup] Found {len(presets)} voice presets")
        return dict(sorted(presets.items()))

    def _determine_voice_key(self, name: Optional[str]) -> str:
        if name and name in self.voice_presets:
            return name

        default_key = "en-Carter_man"
        if default_key in self.voice_presets:
            return default_key

        first_key = next(iter(self.voice_presets))
        print(f"[startup] Using fallback voice preset: {first_key}")
        return first_key

    def _ensure_voice_cached(self, key: str) -> Tuple[object, Path, str]:
        if key not in self.voice_presets:
            raise RuntimeError(f"Voice preset {key!r} not found")

        if key not in self._voice_cache:
            preset_path = self.voice_presets[key]
            print(f"[startup] Loading voice preset {key} from {preset_path}")
            print(f"[startup] Loading prefilled prompt from {preset_path}")
            with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
                prefilled_outputs = torch.load(
                    preset_path,
                    map_location=self._torch_device,
                    weights_only=True,
                )
            self._voice_cache[key] = prefilled_outputs

        return self._voice_cache[key]

    def _get_voice_resources(self, requested_key: Optional[str]) -> Tuple[str, object, Path, str]:
        key = requested_key if requested_key and requested_key in self.voice_presets else self.default_voice_key
        if key is None:
            key = next(iter(self.voice_presets))
            self.default_voice_key = key

        prefilled_outputs = self._ensure_voice_cached(key)
        return key, prefilled_outputs

    def _prepare_inputs(self, text: str, prefilled_outputs: object):
        if not self.processor or not self.model:
            raise RuntimeError("StreamingTTSService not initialized")

        processor_kwargs = {
            "text": text.strip(),
            "cached_prompt": prefilled_outputs,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }

        processed = self.processor.process_input_with_cached_prompt(**processor_kwargs)

        prepared = {
            key: value.to(self._torch_device) if hasattr(value, "to") else value
            for key, value in processed.items()
        }
        return prepared

    def _run_generation(
        self,
        inputs,
        audio_streamer: AudioStreamer,
        errors,
        cfg_scale: float,
        do_sample: bool,
        temperature: float,
        top_p: float,
        refresh_negative: bool,
        prefilled_outputs,
        stop_event: threading.Event,
    ) -> None:
        try:
            self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else 1.0,
                    "top_p": top_p if do_sample else 1.0,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=stop_event.is_set,
                verbose=False,
                refresh_negative=refresh_negative,
                all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
            )
        except Exception as exc:  # pragma: no cover - diagnostic logging
            errors.append(exc)
            traceback.print_exc()
            audio_streamer.end()

    def _stream_single(
        self,
        text: str,
        cfg_scale: float = 1.5,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_p: float = 0.9,
        refresh_negative: bool = True,
        inference_steps: Optional[int] = None,
        voice_key: Optional[str] = None,
        log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        text = text.replace("’", "'")
        selected_voice, prefilled_outputs = self._get_voice_resources(voice_key)

        def emit(event: str, **payload: Any) -> None:
            if log_callback:
                try:
                    log_callback(event, **payload)
                except Exception as exc:
                    print(f"[log_callback] Error while emitting {event}: {exc}")

        steps_to_use = self.inference_steps
        if inference_steps is not None:
            try:
                parsed_steps = int(inference_steps)
                if parsed_steps > 0:
                    steps_to_use = parsed_steps
            except (TypeError, ValueError):
                pass
        if self.model:
            self.model.set_ddpm_inference_steps(num_steps=steps_to_use)
        self.inference_steps = steps_to_use

        inputs = self._prepare_inputs(text, prefilled_outputs)
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: list = []
        stop_signal = stop_event or threading.Event()

        thread = threading.Thread(
            target=self._run_generation,
            kwargs={
                "inputs": inputs,
                "audio_streamer": audio_streamer,
                "errors": errors,
                "cfg_scale": cfg_scale,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "refresh_negative": refresh_negative,
                "prefilled_outputs": prefilled_outputs,
                "stop_event": stop_signal,
            },
            daemon=True,
        )
        thread.start()

        generated_samples = 0

        try:
            stream = audio_streamer.get_stream(0)
            for audio_chunk in stream:
                if torch.is_tensor(audio_chunk):
                    audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.reshape(-1)

                peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                if peak > 1.0:
                    audio_chunk = audio_chunk / peak

                generated_samples += int(audio_chunk.size)
                emit(
                    "model_progress",
                    generated_sec=generated_samples / self.sample_rate,
                    chunk_sec=audio_chunk.size / self.sample_rate,
                )

                chunk_to_yield = audio_chunk.astype(np.float32, copy=False)

                yield chunk_to_yield
        finally:
            stop_signal.set()
            audio_streamer.end()
            thread.join()
            if errors:
                emit("generation_error", message=str(errors[0]))
                raise errors[0]

    def max_total_seconds(self) -> float:
        """Laczny limit nagrania: MAX_LENGTH_MULTIPLIER x pojemnosc jednego przebiegu."""
        positions = 8192
        try:
            positions = int(self.model.config.decoder_config.max_position_embeddings)
        except Exception:
            pass
        return MAX_LENGTH_MULTIPLIER * positions / SPEECH_TOKENS_PER_SECOND

    def stream(
        self,
        text: str,
        log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
        **kwargs: Any,
    ) -> Iterator[np.ndarray]:
        # Jeden GPU = jedna generacja naraz (tryb na zywo i kolejka zadan
        # dziela ta sama blokade, wiec nigdy sie nie zderza).
        with self.generation_lock:
            yield from self._stream_locked(text, log_callback=log_callback, stop_event=stop_event, **kwargs)

    def _stream_locked(
        self,
        text: str,
        log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
        **kwargs: Any,
    ) -> Iterator[np.ndarray]:
        chunks = split_text_for_tts(text)
        if not chunks:
            return
        stop_signal = stop_event or threading.Event()
        limit_samples = int(self.max_total_seconds() * self.sample_rate)
        total_samples = 0
        for index, chunk in enumerate(chunks):
            if stop_signal.is_set():
                break
            if log_callback and len(chunks) > 1:
                try:
                    log_callback("chunk_started", index=index + 1, total=len(chunks), chars=len(chunk))
                except Exception:
                    pass
            # Kazdy fragment dostaje wlasny sygnal stopu, bo _stream_single
            # ustawia go po zakonczeniu; zewnetrzny stop przekazujemy dalej.
            chunk_stop = threading.Event()
            for audio in self._stream_single(chunk, log_callback=log_callback, stop_event=chunk_stop, **kwargs):
                if stop_signal.is_set():
                    chunk_stop.set()
                    break
                remaining = limit_samples - total_samples
                if remaining <= 0:
                    chunk_stop.set()
                    break
                if audio.size > remaining:
                    audio = audio[:remaining]
                total_samples += int(audio.size)
                yield audio
            if total_samples >= limit_samples:
                if log_callback:
                    try:
                        log_callback("max_length_reached", seconds=total_samples / self.sample_rate)
                    except Exception:
                        pass
                break

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        chunk = np.clip(chunk, -1.0, 1.0)
        pcm = (chunk * 32767.0).astype(np.int16)
        return pcm.tobytes()


app = FastAPI()


@app.on_event("startup")
async def _startup() -> None:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH not set in environment")

    device = os.environ.get("MODEL_DEVICE", "cuda")
    
    service = StreamingTTSService(
        model_path=model_path,
        device=device
    )
    service.load()

    app.state.tts_service = service
    app.state.model_path = model_path
    app.state.device = device
    app.state.websocket_lock = asyncio.Lock()
    app.state.jobs = JobManager(
        root=OUTPUT_DIR,
        sample_rate=service.sample_rate,
        generate=service.stream,
        split_text=split_text_for_tts,
        prepare_text=lambda payload: prepare_text(
            str(payload.get("text") or ""),
            str(payload.get("normalize") or "auto"),
            payload.get("voice"),
            bool(payload.get("phonetic", True)),
        ),
        max_seconds=service.max_total_seconds,
    )
    print(f"[startup] Model ready. Jobs dir: {OUTPUT_DIR}")


def streaming_tts(text: str, **kwargs) -> Iterator[np.ndarray]:
    service: StreamingTTSService = app.state.tts_service
    yield from service.stream(text, **kwargs)

@app.websocket("/stream")
async def websocket_stream(ws: WebSocket) -> None:
    await ws.accept()
    text = ws.query_params.get("text", "")
    print(f"Client connected, text={text!r}")
    cfg_param = ws.query_params.get("cfg")
    steps_param = ws.query_params.get("steps")
    voice_param = ws.query_params.get("voice")

    try:
        cfg_scale = float(cfg_param) if cfg_param is not None else 1.5
    except ValueError:
        cfg_scale = 1.5
    if cfg_scale <= 0:
        cfg_scale = 1.5
    try:
        inference_steps = int(steps_param) if steps_param is not None else None
        if inference_steps is not None and inference_steps <= 0:
            inference_steps = None
    except ValueError:
        inference_steps = None

    service: StreamingTTSService = app.state.tts_service
    lock: asyncio.Lock = app.state.websocket_lock

    if lock.locked() or service.generation_lock.locked():
        busy_message = {
            "type": "log",
            "event": "backend_busy",
            "data": {"message": "Model jest zajęty (inne generowanie lub zadanie z kolejki). Spróbuj za chwilę albo dodaj tekst do kolejki."},
            "timestamp": get_timestamp(),
        }
        print("Please wait for the other requests to complete.")
        try:
            await ws.send_text(json.dumps(busy_message))
        except Exception:
            pass
        await ws.close(code=1013, reason="Service busy")
        return

    acquired = False
    try:
        await lock.acquire()
        acquired = True

        log_queue: "Queue[Dict[str, Any]]" = Queue()

        def enqueue_log(event: str, **data: Any) -> None:
            log_queue.put({"event": event, "data": data})

        async def flush_logs() -> None:
            while True:
                try:
                    entry = log_queue.get_nowait()
                except Empty:
                    break
                message = {
                    "type": "log",
                    "event": entry.get("event"),
                    "data": entry.get("data", {}),
                    "timestamp": get_timestamp(),
                }
                try:
                    await ws.send_text(json.dumps(message))
                except Exception:
                    break

        # Polska normalizacja tekstu: liczby, daty, godziny, wieki, skroty
        # slownie (+ fonetyka obcych slow). "auto" = dla polskiego glosu
        # albo tekstu z polskimi znakami.
        normalize_param = (ws.query_params.get("normalize") or "auto").lower()
        phonetic = ws.query_params.get("phonetic", "1") not in ("0", "false", "off")
        if should_normalize_polish(normalize_param, voice_param, text):
            original_text = text
            text = normalize_polish(text, phonetic=phonetic)
            enqueue_log("text_normalized", language="pl", phonetic=phonetic, text=text,
                        changed=text != original_text)

        enqueue_log(
            "backend_request_received",
            text_length=len(text or ""),
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            voice=voice_param,
        )

        stop_signal = threading.Event()

        iterator = streaming_tts(
            text,
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            voice_key=voice_param,
            log_callback=enqueue_log,
            stop_event=stop_signal,
        )
        sentinel = object()
        first_ws_send_logged = False

        # Serwer sam nic nie czyta z gniazda, wiec bez tego nie zauwazylby
        # zamkniecia karty - generowalby dalej w proznie, blokujac GPU.
        async def watch_disconnect() -> None:
            try:
                while True:
                    message = await ws.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
            except Exception:
                pass
            stop_signal.set()

        watcher = asyncio.create_task(watch_disconnect())

        await flush_logs()

        try:
            while ws.client_state == WebSocketState.CONNECTED and not stop_signal.is_set():
                await flush_logs()
                chunk = await asyncio.to_thread(next, iterator, sentinel)
                if chunk is sentinel:
                    break
                chunk = cast(np.ndarray, chunk)
                payload = service.chunk_to_pcm16(chunk)
                await ws.send_bytes(payload)
                if not first_ws_send_logged:
                    first_ws_send_logged = True
                    enqueue_log("backend_first_chunk_sent")
                await flush_logs()
        except WebSocketDisconnect:
            print("Client disconnected (WebSocketDisconnect)")
            enqueue_log("client_disconnected")
            stop_signal.set()
        except Exception as e:
            print(f"Error in websocket stream: {e}")
            traceback.print_exc()
            enqueue_log("backend_error", message=str(e))
            stop_signal.set()
        finally:
            stop_signal.set()
            watcher.cancel()
            enqueue_log("backend_stream_complete")
            await flush_logs()
            try:
                iterator_close = getattr(iterator, "close", None)
                if callable(iterator_close):
                    iterator_close()
            except Exception:
                pass
            # clear the log queue
            while not log_queue.empty():
                try:
                    log_queue.get_nowait()
                except Empty:
                    break
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.close()
            except Exception as e:
                print(f"Error closing websocket: {e}")
            print("WS handler exit")
    finally:
        if acquired:
            lock.release()


@app.get("/")
def index():
    return FileResponse(BASE / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/classic")
def classic():
    """Pierwotny interfejs demo (tylko tryb na zywo)."""
    return FileResponse(BASE / "classic.html")


@app.get("/config")
def get_config():
    service: StreamingTTSService = app.state.tts_service
    voices = sorted(service.voice_presets.keys())
    return {
        "voices": voices,
        "default_voice": service.default_voice_key,
        "max_seconds": service.max_total_seconds(),
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "sample_rate": service.sample_rate,
    }


# ---------------------------------------------------------------------------
# API zadan (niezawodne generowanie do pliku) i narzedzia
# ---------------------------------------------------------------------------

def _jobs() -> JobManager:
    return app.state.jobs


@app.get("/api/health")
def api_health():
    service: StreamingTTSService = app.state.tts_service
    gpu = None
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "memory_used_gb": round((total - free) / 1024 ** 3, 2),
                "memory_total_gb": round(total / 1024 ** 3, 2),
            }
        except Exception:
            gpu = {"name": torch.cuda.get_device_name(0)}
    return {
        "ok": service.model is not None,
        "device": service.device,
        "gpu": gpu,
        "busy": service.generation_lock.locked(),
        "jobs": _jobs().status(),
        "time": get_timestamp(),
    }


@app.post("/api/normalize")
def api_normalize(payload: Dict[str, Any] = Body(...)):
    text = str(payload.get("text") or "")
    mode = str(payload.get("normalize") or "pl")
    result = prepare_text(text, mode, payload.get("voice"), bool(payload.get("phonetic", True)))
    chunks = split_text_for_tts(result)
    return {"text": result, "changed": result != text.strip(), "chunks": len(chunks), "chars": len(result)}


@app.get("/api/jobs")
def api_jobs():
    return {"jobs": _jobs().list()}


@app.post("/api/jobs", status_code=201)
def api_job_create(payload: Dict[str, Any] = Body(...)):
    try:
        return _jobs().create(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = _jobs().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Nie ma takiego zadania.")
    return job


def _job_action(fn, job_id: str):
    try:
        return fn(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/cancel")
def api_job_cancel(job_id: str):
    return _job_action(_jobs().cancel, job_id)


@app.post("/api/jobs/{job_id}/resume")
def api_job_resume(job_id: str):
    return _job_action(_jobs().resume, job_id)


@app.delete("/api/jobs/{job_id}")
def api_job_delete(job_id: str):
    _job_action(_jobs().delete, job_id)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/audio")
def api_job_audio(job_id: str, download: int = 0):
    path = _jobs().audio_path(job_id)
    job = _jobs().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Nie ma takiego zadania.")
    filename = f"vibevoice-{job_id}.wav"
    if path:
        return FileResponse(
            path, media_type="audio/wav",
            filename=filename if download else None,
            headers={"Cache-Control": "no-cache"},
        )
    data = _jobs().partial_wav(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Nagranie jeszcze nie powstało.")
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return Response(content=data, media_type="audio/wav", headers=headers)


@app.get("/api/jobs/{job_id}/text")
def api_job_text(job_id: str):
    path = _jobs().text_path(job_id)
    if not path:
        raise HTTPException(status_code=404, detail="Brak tekstu zadania.")
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename=f"vibevoice-{job_id}.txt")

