"""
A_Module.py - Hybrid Audio Subsystem (macOS CoreAudio + Windows Fallback)
Fast-Whisper On-Device STT, Native afplay / Pygame Caching, and Dynamic Prompts.
"""

import os
import time
import platform
import tempfile
import threading
import subprocess
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from config import SystemConfig

# Detect operating system
IS_MAC = (platform.system() == "Darwin")

if not IS_MAC:
    import pygame
    pygame.mixer.init()

class AudioEngine:
    def __init__(self):
        print(f"[*] Initializing Audio Engine Subsystems (Platform: {'macOS' if IS_MAC else 'Windows/Linux'})...")
        self._is_speaking = False
        self.audio_lock = threading.Lock()

        # 1. Initialize Microphone Listener with Snappy 1.5s Pause Threshold
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = getattr(SystemConfig, "SPEECH_PAUSE_THRESHOLD", 1.5)
        self.recognizer.energy_threshold = getattr(SystemConfig, "MIC_ENERGY_THRESHOLD", 100)
        self.recognizer.dynamic_energy_threshold = True

        # Pre-calibrate microphone once at boot (Auto-routes CoreAudio default on Mac)
        mic_idx = None if IS_MAC else getattr(SystemConfig, "MICROPHONE_DEVICE_INDEX", None)
        try:
            with sr.Microphone(device_index=mic_idx) as source:
                print(f"[*] Calibrating microphone (Device: {mic_idx or 'System Default'})...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.6)
                print(f"[+] Ambient energy baseline established at: {self.recognizer.energy_threshold:.1f}")
        except Exception as e:
            print(f"[!] Warning: Initial mic calibration skipped: {e}")

        # 2. Load Local Fast-Whisper Model (CTranslate2 int8 Quantization + 4 CPU Threads)
        self.whisper_model = None
        self._init_faster_whisper()
        self.warmup()

    def _init_faster_whisper(self):
        """Loads faster-whisper model on CPU using multi-threaded acceleration."""
        try:
            from faster_whisper import WhisperModel
            model_name = getattr(SystemConfig, "FASTER_WHISPER_MODEL", "medium")
            device_type = getattr(SystemConfig, "FASTER_WHISPER_DEVICE", "cpu")
            compute_type = getattr(SystemConfig, "FASTER_WHISPER_COMPUTE", "int8")
            threads = getattr(SystemConfig, "FASTER_WHISPER_CPU_THREADS", 4)

            print(f"[*] Loading Fast-Whisper [{model_name} | {compute_type} | Threads={threads}] on {device_type.upper()}...")
            self.whisper_model = WhisperModel(
                model_name,
                device=device_type,
                compute_type=compute_type,
                cpu_threads=threads
            )
            print(f"[+] Fast-Whisper [{model_name}] armed with multi-threaded CPU acceleration.")
        except Exception as e:
            print(f"[!] Warning: faster-whisper initialization issue ({e}). Fallback armed.")
            self.whisper_model = None

    def warmup(self):
        """Pre-allocates CTranslate2 memory buffers on startup."""
        if self.whisper_model is not None:
            try:
                dummy_pcm = np.zeros(1600, dtype=np.float32)
                self.whisper_model.transcribe(dummy_pcm, beam_size=1)
                print("[+] Fast-Whisper memory buffers pre-warmed.")
            except Exception:
                pass

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def _play_audio_file(self, filepath: str, block: bool = True):
        """Plays audio using native CoreAudio (afplay) on macOS, or Pygame on Windows."""
        if not os.path.exists(filepath):
            print(f"[!] Warning: Audio file not found at: {filepath}")
            return

        def _worker():
            with self.audio_lock:
                self._is_speaking = True
                try:
                    if IS_MAC:
                        # Native macOS CoreAudio execution (Zero SDL2 collisions)
                        subprocess.run(["/usr/bin/afplay", filepath], check=True)
                    else:
                        # Windows fallback
                        pygame.mixer.music.load(filepath)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.04)
                        pygame.mixer.music.unload()
                except Exception as e:
                    print(f"[!] Audio Playback Failure: {e}")
                finally:
                    self._is_speaking = False

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        if block:
            thread.join()

    # -------------------------------------------------------------------------
    # ON-TOPIC MODE: Instant Playback from Local Template Cache (.mp3)
    # -------------------------------------------------------------------------
    def play_template(self, template_key: str, lang: str = "th", block: bool = True):
        """Plays pre-rendered audio templates directly from disk with zero latency."""
        filename = SystemConfig.TEMPLATES.get(lang, {}).get(template_key)
        if not filename:
            print(f"[!] Template key '{template_key}' missing for language '{lang}'.")
            return

        filepath = os.path.join(SystemConfig.AUDIO_TEMPLATE_DIR, filename)
        print(f"[*] [Audio Cache]: Playing '{template_key}' ({lang}) -> {filename}")
        self._play_audio_file(filepath, block=block)

    # -------------------------------------------------------------------------
    # DYNAMIC PROBE MODE: Quick gTTS Synthesis for Non-PII Steering Prompts
    # -------------------------------------------------------------------------
    def speak_dynamic(self, text: str, lang: str = "th", block: bool = True):
        """Synthesizes dynamic conversational prompts on the fly."""
        print(f"[*] [Dynamic gTTS]: Synthesizing prompt -> \"{text}\"")
        def _worker():
            temp_path = None
            try:
                tts = gTTS(text=text, lang=lang, slow=False)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    temp_path = f.name
                tts.save(temp_path)
                self._play_audio_file(temp_path, block=True)
            except Exception as e:
                print(f"[!] Dynamic TTS Error: {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        if block:
            thread.join()

    # -------------------------------------------------------------------------
    # FAST SPEECH INGESTION: Dynamic Auto-Detection & Clean Decoding
    # -------------------------------------------------------------------------
    def listen(self, language: str = "th") -> str:
        """Captures microphone audio and decodes via multi-threaded Fast-Whisper."""
        transcript = ""
        mic_idx = None if IS_MAC else getattr(SystemConfig, "MICROPHONE_DEVICE_INDEX", None)

        with sr.Microphone(device_index=mic_idx) as source:
            print(f"[*] Mic Listening [pause_threshold={self.recognizer.pause_threshold}s]...")
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=SystemConfig.SPEECH_TIMEOUT,
                    phrase_time_limit=SystemConfig.SPEECH_PHRASE_LIMIT
                )

                if self.whisper_model is not None:
                    raw_pcm = audio.get_raw_data(convert_rate=16000, convert_width=2)
                    audio_float32 = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

                    # Automatic Volume Normalization
                    max_amp = np.max(np.abs(audio_float32))
                    if max_amp > 0.01:
                        audio_float32 = (audio_float32 / (max_amp + 1e-6)) * 0.9

                    # Dynamic Language Routing: Auto-detect in Standby, or target active language
                    if language in ["auto", None, ""]:
                        lang_code = None  # Dynamic detection for "Hello" vs "สวัสดี"
                    else:
                        lang_code = "th" if language.startswith("th") else "en"

                    # Natural Decoding: No fragile prompt-stuffing hacks
                    segments, _ = self.whisper_model.transcribe(
                        audio_float32,
                        beam_size=2,
                        language=lang_code,
                        vad_filter=True
                    )
                    transcript = " ".join([seg.text for seg in segments]).strip()
                else:
                    stt_lang = "th-TH" if language.startswith("th") else "en-US"
                    transcript = self.recognizer.recognize_google(audio, language=stt_lang)

                print(f"[+] Decoded Speech: \"{transcript}\"")

            except sr.WaitTimeoutError:
                print("[-] Silence timeout: No speech detected.")
            except sr.UnknownValueError:
                print("[-] Audio captured but speech was unintelligible.")
            except Exception as e:
                print(f"[!] Audio Engine Listen Exception: {e}")

        return transcript.strip()

    # -------------------------------------------------------------------------
    # WAKE-WORD DETECTION
    # -------------------------------------------------------------------------
    @staticmethod
    def detect_wake_word(utterance: str) -> tuple[bool, str]:
        """Detects wake words in ambient listening."""
        text = utterance.lower()
        for w in SystemConfig.WAKE_WORDS_TH:
            if w in text:
                return True, "th"
        for w in SystemConfig.WAKE_WORDS_EN:
            if w in text:
                return True, "en"
        return False, "th"

    def release(self):
        if not IS_MAC:
            pygame.mixer.quit()