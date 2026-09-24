"""
A_Module.py - Hybrid Audio Subsystem (Step 3)
Local Fast-Whisper STT, Pygame Audio Template Caching, and Zero-PII Dynamic gTTS Pivots.
"""

import os
import time
import tempfile
import threading
import numpy as np
import pygame
import speech_recognition as sr
from gtts import gTTS
from config import SystemConfig

class AudioEngine:
    def __init__(self):
        print("[*] Initializing Audio Engine Subsystems...")
        
        # 1. Initialize Pygame Mixer for Native, Non-Blocking Audio Playback
        pygame.mixer.init()
        self._is_speaking = False
        self.audio_lock = threading.Lock()

        # 2. Initialize Microphone Listener with 2.5s Silence Threshold
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = SystemConfig.SPEECH_PAUSE_THRESHOLD
        self.recognizer.energy_threshold = SystemConfig.MIC_ENERGY_THRESHOLD
        self.recognizer.dynamic_energy_threshold = True

        # 3. Load Local Fast-Whisper Model (CTranslate2 int8 Quantization)
        self.whisper_model = None
        self._init_faster_whisper()

    def _init_faster_whisper(self):
        """Loads faster-whisper on-device model for local, privacy-compliant STT."""
        try:
            from faster_whisper import WhisperModel
            print(f"[*] Loading Fast-Whisper [{SystemConfig.FASTER_WHISPER_MODEL} | {SystemConfig.FASTER_WHISPER_COMPUTE}] on local compute...")
            self.whisper_model = WhisperModel(
                SystemConfig.FASTER_WHISPER_MODEL,
                device="auto",
                compute_type=SystemConfig.FASTER_WHISPER_COMPUTE
            )
            print("[+] Fast-Whisper armed and ready on-device.")
        except Exception as e:
            print(f"[!] Warning: faster-whisper not available ({e}). Using Google Speech API fallback.")
            self.whisper_model = None

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def _play_audio_file(self, filepath: str, block: bool = True):
        """Plays an audio file via pygame without external media player popups."""
        if not os.path.exists(filepath):
            print(f"[!] Warning: Audio file not found at: {filepath}")
            return

        def _worker():
            with self.audio_lock:
                self._is_speaking = True
                try:
                    pygame.mixer.music.load(filepath)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
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
        """
        Plays pre-rendered audio templates directly from disk.
        Zero latency. Zero cloud API calls. Zero PII exposure.
        """
        filename = SystemConfig.TEMPLATES.get(lang, {}).get(template_key)
        if not filename:
            print(f"[!] Template key '{template_key}' missing for language '{lang}'.")
            return

        filepath = os.path.join(SystemConfig.AUDIO_TEMPLATE_DIR, filename)
        print(f"[*] [Audio Cache]: Playing '{template_key}' ({lang}) -> {filename}")
        self._play_audio_file(filepath, block=block)

    # -------------------------------------------------------------------------
    # OFF-TOPIC MODE: Dynamic gTTS Synthesis for Zero-PII Steering Pivots
    # -------------------------------------------------------------------------
    def speak_dynamic(self, text: str, lang: str = "th", block: bool = True):
        """
        Synthesizes dynamic text via gTTS on the fly.
        Used strictly for system conversational steering (Zero-PII guarantee).
        """
        print(f"[*] [Dynamic gTTS]: Synthesizing pivot -> \"{text}\"")
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
    # SPEECH INGESTION: Dynamic Pause Threshold (2.5s) + Local Fast-Whisper
    # -------------------------------------------------------------------------
    def listen(self, language: str = "th") -> str:
        """
        Captures microphone stream, waits for 2.5s silence, and decodes via Fast-Whisper.
        """
        transcript = ""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print(f"[*] Mic Listening [pause_threshold={self.recognizer.pause_threshold}s, energy={self.recognizer.energy_threshold}]...")
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=SystemConfig.SPEECH_TIMEOUT,
                    phrase_time_limit=SystemConfig.SPEECH_PHRASE_LIMIT
                )

                # Route to Local Fast-Whisper
                if self.whisper_model is not None:
                    # Extract 16kHz 16-bit mono PCM into float32 array
                    raw_pcm = audio.get_raw_data(convert_rate=16000, convert_width=2)
                    audio_float32 = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

                    lang_code = "th" if language.startswith("th") else "en"
                    segments, _ = self.whisper_model.transcribe(
                        audio_float32,
                        beam_size=1,
                        language=lang_code,
                        vad_filter=True
                    )
                    transcript = " ".join([seg.text for seg in segments]).strip()
                else:
                    # Cloud Fallback if Fast-Whisper is uninstalled
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
    # WAKE-WORD VERIFICATION
    # -------------------------------------------------------------------------
    @staticmethod
    def detect_wake_word(utterance: str) -> tuple[bool, str]:
        """
        Checks if an utterance triggers the wake-word gate.
        Returns (is_triggered, detected_language).
        """
        text = utterance.lower()
        for w in SystemConfig.WAKE_WORDS_TH:
            if w in text:
                return True, "th"
        for w in SystemConfig.WAKE_WORDS_EN:
            if w in text:
                return True, "en"
        return False, "th"

    def release(self):
        pygame.mixer.quit()