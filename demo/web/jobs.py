"""Kolejka zadan generowania mowy zapisywana na dysku.

Niezawodnosc:
- kazde zadanie ma katalog outputs/jobs/<id>/ z meta.json (stan), audio.pcm
  (PCM16 dopisywany po KAZDYM ukonczonym fragmencie) i - po zakonczeniu -
  audio.wav;
- zadanie nie zalezy od przegladarki: zamkniecie karty, zerwane polaczenie
  czy odswiezenie strony nic nie przerywa;
- po restarcie serwera niedokonczone zadania wracaja do kolejki i ruszaja od
  pierwszego nieukonczonego fragmentu;
- fragment, ktory sie nie uda, jest ponawiany (do MAX_ATTEMPTS razy, przy
  bledzie pamieci GPU z czyszczeniem cache); po wyczerpaniu prob zadanie ma
  status "error" i mozna je wznowic recznie.
"""
from __future__ import annotations

import datetime
import json
import os
import queue
import shutil
import struct
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

MAX_ATTEMPTS = 3
ACTIVE = ("queued", "running")


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def wav_header(num_bytes: int, sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 36 + num_bytes) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", num_bytes)
    )


class JobManager:
    def __init__(
        self,
        root: Path,
        sample_rate: int,
        generate: Callable[..., Any],
        split_text: Callable[[str], List[str]],
        prepare_text: Callable[[Dict[str, Any]], str],
        max_seconds: Callable[[], float],
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self._generate = generate          # generate(text, voice_key, cfg_scale, inference_steps, stop_event) -> iterator[np.ndarray]
        self._split = split_text
        self._prepare = prepare_text       # payload -> tekst po normalizacji
        self._max_seconds = max_seconds
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._cancel: Dict[str, threading.Event] = {}
        self._running: Optional[str] = None
        self._load_existing()
        self._worker = threading.Thread(target=self._loop, name="tts-jobs", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ dysk

    def _dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _save(self, job: Dict[str, Any]) -> None:
        job["updated_at"] = _now()
        path = self._dir(job["id"]) / "meta.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    def _load_existing(self) -> None:
        resume = os.environ.get("VIBEVOICE_AUTO_RESUME", "1") not in ("0", "false")
        for meta in sorted(self.root.glob("*/meta.json")):
            try:
                job = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:
                print(f"[jobs] Pomijam uszkodzony plik {meta}")
                continue
            self._jobs[job["id"]] = job
            if job.get("status") in ACTIVE:
                if resume:
                    job["status"] = "queued"
                    job["note"] = "Wznowione po restarcie serwera."
                    self._save(job)
                    self._queue.put(job["id"])
                    print(f"[jobs] Wznawiam zadanie {job['id']} od fragmentu {job.get('chunks_done', 0) + 1}")
                else:
                    job["status"] = "interrupted"
                    self._save(job)

    # ------------------------------------------------------------------ API

    def create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = str(payload.get("text") or "").strip()
        if not text:
            raise ValueError("Tekst jest pusty.")
        final_text = self._prepare(payload)
        chunks = self._split(final_text)
        if not chunks:
            raise ValueError("Po normalizacji nie zostało nic do przeczytania.")
        job_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        title = str(payload.get("title") or "").strip() or " ".join(text.split())[:70]
        job = {
            "id": job_id,
            "title": title[:120],
            "created_at": _now(),
            "status": "queued",
            "text": text,
            "final_text": final_text,
            "voice": payload.get("voice") or None,
            "cfg_scale": float(payload.get("cfg_scale") or 1.5),
            "inference_steps": int(payload.get("inference_steps") or 5),
            "normalize": payload.get("normalize") or "auto",
            "phonetic": bool(payload.get("phonetic", True)),
            "pause_ms": max(0, min(3000, int(payload.get("pause_ms") or 350))),
            "chunks": chunks,
            "chunks_done": 0,
            "chunk_samples": [],
            "chunk_seconds": [],
            "seconds": 0.0,
            "error": "",
            "note": "",
        }
        self._dir(job_id).mkdir(parents=True, exist_ok=True)
        (self._dir(job_id) / "text.txt").write_text(final_text, encoding="utf-8")
        with self._lock:
            self._jobs[job_id] = job
            self._save(job)
        self._queue.put(job_id)
        return self.public(job)

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j["created_at"], reverse=True)
            return [self.public(j, brief=True) for j in jobs]

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return self.public(job) if job else None

    def public(self, job: Dict[str, Any], brief: bool = False) -> Dict[str, Any]:
        out = {k: v for k, v in job.items() if not (brief and k in ("text", "final_text", "chunks"))}
        out["chunks_total"] = len(job.get("chunks", []))
        out["queue_position"] = self._queue_position(job["id"]) if job.get("status") == "queued" else 0
        out["has_audio"] = job.get("chunks_done", 0) > 0
        return out

    def _queue_position(self, job_id: str) -> int:
        with self._queue.mutex:
            items = list(self._queue.queue)
        return items.index(job_id) + 1 if job_id in items else 0

    def cancel(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._require(job_id)
            if job["status"] in ACTIVE:
                if job["status"] == "queued":
                    job["status"] = "cancelled"
                    self._save(job)
                event = self._cancel.get(job_id)
                if event:
                    event.set()
            return self.public(job)

    def resume(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._require(job_id)
            if job["status"] in ACTIVE:
                return self.public(job)
            if job["status"] == "done":
                raise ValueError("Zadanie jest już ukończone.")
            job["status"] = "queued"
            job["error"] = ""
            job["note"] = f"Wznowione od fragmentu {job['chunks_done'] + 1}."
            self._save(job)
        self._queue.put(job_id)
        return self.public(job)

    def delete(self, job_id: str) -> None:
        with self._lock:
            job = self._require(job_id)
            if job["status"] == "running":
                raise ValueError("Najpierw zatrzymaj zadanie.")
            job["status"] = "deleted"
            self._jobs.pop(job_id, None)
        shutil.rmtree(self._dir(job_id), ignore_errors=True)

    def audio_path(self, job_id: str) -> Optional[Path]:
        path = self._dir(job_id) / "audio.wav"
        return path if path.exists() else None

    def partial_wav(self, job_id: str) -> Optional[bytes]:
        pcm = self._dir(job_id) / "audio.pcm"
        if not pcm.exists():
            return None
        data = pcm.read_bytes()
        data = data[: len(data) - (len(data) % 2)]
        return wav_header(len(data), self.sample_rate) + data

    def text_path(self, job_id: str) -> Optional[Path]:
        path = self._dir(job_id) / "text.txt"
        return path if path.exists() else None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            counts: Dict[str, int] = {}
            for job in self._jobs.values():
                counts[job["status"]] = counts.get(job["status"], 0) + 1
        return {"running": self._running, "queued": self._queue.qsize(), "counts": counts}

    def _require(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError("Nie ma takiego zadania.")
        return job

    # ------------------------------------------------------------------ worker

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            with self._lock:
                job = self._jobs.get(job_id)
                if not job or job["status"] != "queued":
                    continue
                job["status"] = "running"
                job["started_at"] = job.get("started_at") or _now()
                self._running = job_id
                cancel = self._cancel[job_id] = threading.Event()
                self._save(job)
            try:
                self._run(job, cancel)
            except Exception as exc:  # pragma: no cover - ostatnia linia obrony
                traceback.print_exc()
                with self._lock:
                    job["status"] = "error"
                    job["error"] = str(exc)
                    self._save(job)
            finally:
                with self._lock:
                    self._running = None
                    self._cancel.pop(job_id, None)

    def _run(self, job: Dict[str, Any], cancel: threading.Event) -> None:
        directory = self._dir(job["id"])
        pcm_path = directory / "audio.pcm"
        # Plik PCM moze miec ogon z przerwanego fragmentu - przycinamy go do
        # dlugosci wynikajacej z zapisanych, ukonczonych fragmentow.
        expected = sum(job.setdefault("chunk_samples", [])) * 2
        if pcm_path.exists() and pcm_path.stat().st_size != expected:
            with open(pcm_path, "r+b") as fh:
                fh.truncate(min(expected, pcm_path.stat().st_size))
        limit = self._max_seconds()
        pause = np.zeros(int(self.sample_rate * job.get("pause_ms", 350) / 1000), dtype=np.int16)

        while job["chunks_done"] < len(job["chunks"]):
            if cancel.is_set():
                with self._lock:
                    job["status"] = "cancelled"
                    self._save(job)
                return
            if job["seconds"] >= limit:
                with self._lock:
                    job["note"] = f"Osiągnięto limit długości nagrania ({int(limit // 60)} min)."
                break
            index = job["chunks_done"]
            text = job["chunks"][index]
            audio = None
            last_error = ""
            for attempt in range(1, MAX_ATTEMPTS + 1):
                if cancel.is_set():
                    break
                try:
                    started = time.time()
                    pieces = []
                    for piece in self._generate(
                        text,
                        voice_key=job.get("voice"),
                        cfg_scale=job.get("cfg_scale", 1.5),
                        inference_steps=job.get("inference_steps"),
                        stop_event=cancel,
                    ):
                        pieces.append(np.asarray(piece, dtype=np.float32).reshape(-1))
                    if cancel.is_set():
                        break
                    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
                    if audio.size < self.sample_rate * 0.2:
                        raise RuntimeError("Model zwrócił puste nagranie fragmentu.")
                    print(f"[jobs] {job['id']} fragment {index + 1}/{len(job['chunks'])} "
                          f"{audio.size / self.sample_rate:.1f}s w {time.time() - started:.1f}s")
                    break
                except Exception as exc:
                    audio = None
                    last_error = f"{type(exc).__name__}: {exc}"
                    print(f"[jobs] {job['id']} fragment {index + 1} próba {attempt} nieudana: {last_error}")
                    traceback.print_exc()
                    _free_gpu_memory()
                    time.sleep(min(5, attempt * 1.5))
            if cancel.is_set():
                continue
            if audio is None:
                with self._lock:
                    job["status"] = "error"
                    job["error"] = f"Fragment {index + 1} nie udał się po {MAX_ATTEMPTS} próbach: {last_error}"
                    self._save(job)
                return
            pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
            if index + 1 < len(job["chunks"]) and pause.size:
                pcm = np.concatenate([pcm, pause])
            with open(pcm_path, "ab") as fh:
                fh.write(pcm.tobytes())
                fh.flush()
                os.fsync(fh.fileno())
            with self._lock:
                job["chunk_samples"].append(int(pcm.size))
                job["chunk_seconds"].append(round(pcm.size / self.sample_rate, 3))
                job["seconds"] = round(sum(job["chunk_seconds"]), 3)
                job["chunks_done"] = index + 1
                self._save(job)

        data = pcm_path.read_bytes() if pcm_path.exists() else b""
        wav_tmp = directory / "audio.wav.tmp"
        wav_tmp.write_bytes(wav_header(len(data), self.sample_rate) + data)
        os.replace(wav_tmp, directory / "audio.wav")
        with self._lock:
            job["status"] = "done"
            job["finished_at"] = _now()
            self._save(job)


def _free_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
