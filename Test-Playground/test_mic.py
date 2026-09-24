"""
test_mic.py - Multi-Mic Target Diagnostic Tool
Tests individual microphone indices with live VU meters and transcription.
"""

import time
import numpy as np
import pyaudio
import speech_recognition as sr
from faster_whisper import WhisperModel

def test_microphone_hardware():
    p = pyaudio.PyAudio()
    print("=" * 65)
    print("      STARK INDUSTRIES AUDIO HARDWARE SELECTOR & TESTER      ")
    print("=" * 65)

    # 1. Clean Global Device Enumeration
    input_devices = {}
    print("\n Detected Audio Input Hardware:")
    for i in range(p.get_device_count()):
        try:
            dev = p.get_device_info_by_index(i)
            if dev.get('maxInputChannels') > 0:
                name = dev.get('name')
                print(f"  [Index {i}] {name}")
                input_devices[i] = name
        except Exception:
            pass

    if not input_devices:
        print("[!] No recording devices detected.")
        p.terminate()
        return

    # 2. Select Device Index
    print("\n" + "-" * 65)
    choice = input("Enter the Index number of the mic you want to test (e.g. 1, 2, or 3): ").strip()
    target_idx = int(choice) if choice.isdigit() and int(choice) in input_devices else 1
    print(f"[*] Testing Device [Index {target_idx}]: {input_devices.get(target_idx)}")
    print("-" * 65)

    # 3. Live 6-Second Sound Meter
    print("\n Live Sound Meter (Speak or tap your mic now!):")
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        input_device_index=target_idx,
        frames_per_buffer=1024
    )

    start = time.time()
    try:
        while time.time() - start < 6.0:
            data = stream.read(1024, exception_on_overflow=False)
            pcm = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(pcm.astype(np.float32)**2))
            meter_len = int(min(rms / 100, 40))
            bars = "█" * meter_len + "░" * (40 - meter_len)
            print(f"\r  Signal Level: [{bars}] {int(rms):4d}", end="", flush=True)
            time.sleep(0.05)
    finally:
        stream.stop_stream()
        stream.close()
    print("\n  [✓] VU Meter test complete.")

    # 4. Countdown + 4-Second Recording Test
    print("\n" + "=" * 65)
    print(" Get ready to speak into your USB mic:")
    for count in range(3, 0, -1):
        print(f"  Starting in {count}...", flush=True)
        time.sleep(1.0)
    print("  >>> [SPEAK NOW!] Say 'สวัสดีครับ' or 'Hello' clearly! <<<", flush=True)

    r = sr.Recognizer()
    with sr.Microphone(device_index=target_idx) as source:
        r.adjust_for_ambient_noise(source, duration=0.3)
        audio = r.record(source, duration=4.0)

    print("  [✓] Recording captured. Analyzing audio stream...")

    # Save to WAV so you can play it back and verify
    wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
    with open("test_recording.wav", "wb") as f:
        f.write(wav_bytes)
    print("  [i] Saved audio to 'test_recording.wav' (Double-click to listen).")

    # Extract PCM and check signal strengths
    raw_pcm = audio.get_raw_data(convert_rate=16000, convert_width=2)
    audio_f32 = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

    max_amplitude = np.max(np.abs(audio_f32))
    print(f"  [*] Peak Recorded Amplitude: {max_amplitude:.4f}")

    if max_amplitude < 0.02:
        print("  [!] WARNING: Audio was nearly silent! Check mic volume slider in Windows.")
    else:
        # Boost volume to optimal level for Whisper
        audio_f32 = audio_f32 / (max_amplitude + 1e-6) * 0.9

    print("  [*] Decoding with Fast-Whisper on CPU...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_f32, beam_size=3)
    transcript = " ".join([s.text for s in segments]).strip()

    print("\n" + "=" * 65)
    print(f"  [DETECTED LANGUAGE]: {info.language} (Probability: {info.language_probability:.2f})")
    print(f"  [DECODED RESULT]   : \"{transcript}\"")
    print("=" * 65)

    p.terminate()

if __name__ == "__main__":
    test_microphone_hardware()