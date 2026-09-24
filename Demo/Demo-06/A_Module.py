"""
A_Module.py - Hybrid Audio Subsystem (Performance Tuned)
Local Fast-Whisper STT (4-Core CPU Accelerated), Pygame Audio Templates, and Fast gTTS Pivots.
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

        # 2. Initialize Microphone Listener with Snappy 1.5s Pause Threshold
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = SystemConfig.SPEECH_PAUSE_THRESHOLD
        self.recognizer.energy_threshold = SystemConfig.MIC_ENERGY_THRESHOLD
        self.recognizer.dynamic_energy_threshold = True

        # Pre-calibrate microphone once at boot (saves 0.4s per turn later)
        try:
            with sr.Microphone(device_index=SystemConfig.MICROPHONE_DEVICE_INDEX) as source:
                print("[*] Calibrating microphone for ambient room noise once...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
                print(f"[+] Ambient energy baseline established at: {self.recognizer.energy_threshold:.1f}")
        except Exception as e:
            print(f"[!] Warning: Initial mic calibration skipped: {e}")

        # 3. Load Local Fast-Whisper Model (CTranslate2 int8 Quantization + 4 CPU Threads)
        self.whisper_model = None
        self._init_faster_whisper()
        self.warmup()

    def _init_faster_whisper(self):
        """Loads faster-whisper model on CPU using 4 parallel worker threads."""
        try:
            from faster_whisper import WhisperModel
            print(f"[*] Loading Fast-Whisper [{SystemConfig.FASTER_WHISPER_MODEL} | {SystemConfig.FASTER_WHISPER_COMPUTE} | Threads={SystemConfig.FASTER_WHISPER_CPU_THREADS}] on CPU...")
            self.whisper_model = WhisperModel(
                SystemConfig.FASTER_WHISPER_MODEL,
                device=SystemConfig.FASTER_WHISPER_DEVICE,
                compute_type=SystemConfig.FASTER_WHISPER_COMPUTE,
                cpu_threads=SystemConfig.FASTER_WHISPER_CPU_THREADS
            )
            print("[+] Fast-Whisper armed with multi-threaded CPU acceleration.")
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
        """Plays an audio file via pygame directly in memory without OS media player modals."""
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
    # FAST SPEECH INGESTION: 1.5s Pause Threshold + Local Fast-Whisper
    # -------------------------------------------------------------------------
    def listen(self, language: str = "th") -> str:
        """Captures microphone audio and decodes via multi-threaded Fast-Whisper."""
        transcript = ""
        with sr.Microphone(device_index=SystemConfig.MICROPHONE_DEVICE_INDEX) as source:
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

                    # Medical Vocabulary Priming Hint
                    medical_prompt = "ผู้ป่วยมาคัดกรองที่โรงพยาบาล แจ้งอาการไม่สบาย เจ็บหน้าอก แน่นหน้าอก ปวดท้อง มีไข้ ปวดหัว เวียนหัว คลื่นไส้ อาเจียน หนาวสั่น หายใจลำบาก หกโมงเช้า เมื่อวาน สองวัน ติดต่อเจ้าหน้าที่" if language == "th" else "Hospital emergency triage intake, chief complaint, chest pain, abdominal pain, fever, headache, dizziness, yesterday, hours, Contact, nurse"

                    lang_code = "th" if language.startswith("th") else "en"
                    segments, _ = self.whisper_model.transcribe(
                        audio_float32,
                        beam_size=2,
                        language=lang_code,
                        initial_prompt=medical_prompt,
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
        pygame.mixer.quit()