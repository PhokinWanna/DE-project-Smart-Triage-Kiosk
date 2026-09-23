"""
A_Module.py - Audio Engine
Lightweight Speech Recognition (Google Speech API with 2.5s pause_threshold) + gTTS Playback.
"""

import os
import time
import tempfile
import threading
import platform
import speech_recognition as sr
from gtts import gTTS
from config import SystemConfig

class AudioEngine:
    def __init__(self):
        print("[*] Initializing Speech Recognistion & Audio Playback Subsystems...")
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = SystemConfig.SPEECH_PAUSE_THRESHOLD
        self.recognizer.energy_threshold = SystemConfig.SPEECH_ENERGY_THRESHOLD
        self.recognizer.dynamic_energy_threshold = True
        self._is_speaking = False

    def speak(self, text: str, lang: str = "th"):
        """
        Asynchronously converts text to audio via gTTS and plays it back
        without blocking the caller or freezing camera acquisition.
        """
        def _worker():
            self._is_speaking = True
            temp_path = None
            try:
                tts = gTTS(text=text, lang=lang, slow=False)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    temp_path = f.name
                tts.save(temp_path)

                # Cross-platform audio playback dispatch
                current_os = platform.system()
                if current_os == "Darwin":      # macOS
                    os.system(f"afplay '{temp_path}'")
                elif current_os == "Windows":    # Windows
                    os.system(f'powershell -c "(New-Object Media.SoundPlayer \'{temp_path}\').PlaySync()" 2>nul || start /min wmplayer "{temp_path}"')
                else:                           # Linux / Raspberry Pi
                    os.system(f"mpg123 -q '{temp_path}' 2>/dev/null || aplay '{temp_path}' 2>/dev/null")
            except Exception as e:
                print(f"[!] Audio Playback Error: {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                self._is_speaking = False

        threading.Thread(target=_worker, daemon=True).start()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def listen(self, language: str = "th-TH") -> str:
        """
        Listens via microphone using Google Speech Recognition.
        Respects the 2.5s silence pause_threshold to avoid premature cutoffs.
        """
        transcript = ""
        with sr.Microphone() as source:
            # Quick ambient noise adjustment
            self.recognizer.adjust_for_ambient_noise(source, duration=0.6)
            print(f"[*] Listening (lang={language}, pause_threshold={self.recognizer.pause_threshold}s)...")
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=SystemConfig.SPEECH_TIMEOUT,
                    phrase_time_limit=SystemConfig.SPEECH_PHRASE_LIMIT
                )
                transcript = self.recognizer.recognize_google(audio, language=language)
                print(f"[+] Transcribed Utterance: \"{transcript}\"")
            except sr.WaitTimeoutError:
                print("[-] Listen timed out: Patient remained silent.")
            except sr.UnknownValueError:
                print("[-] Speech unrecognizable.")
            except sr.RequestError as e:
                print(f"[!] Google Speech API Network Error: {e}")
        
        return transcript.strip()

    def release(self):
        pass