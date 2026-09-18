# Change Detail
<<<<<<< HEAD

*Enhance from Demo-03*

*Key Improvements:*

- Reconstructuring Code's Architecture.
- Lightweight Startup: Initial boot time drops from 25 seconds down to ~1 second by replacing Whisper/PyTorch with speech_recognition + gTTS.
- Dynamic Speech Ingestion: Natural speech listening now runs with pause_threshold = 2.5 instead of a hard 7-second cutoff.
- Clinical Guardrails: Restores Demo-03's false-positive filters (crossed arms, thinking pose, scratch persistence) and Scenario A/B/C cross-validation.
- CNN Skin Analysis: Integrates a skin_classifier.tflite scaffold targeting Normal vs. Red/Flushing (with Pallor removed).
- Dispatch: Data is transmitted to the nurse dashboard only after the case is complete. Patients who decline the PDPA notice are routed immediately to the manual desk.
=======
***Enhance from Demo-03***

***Key Improvements:***

- Reconstructuring Code's Architecture.
                    
- Lightweight Startup: Initial boot time drops from 25 seconds down to ~1 second by replacing Whisper/PyTorch with speech_recognition + gTTS.

- Dynamic Speech Ingestion: Natural speech listening now runs with pause_threshold = 2.5 instead of a hard 7-second cutoff.

- Clinical Guardrails: Restores Demo-03's false-positive filters (crossed arms, thinking pose, scratch persistence) and Scenario A/B/C cross-validation.

- CNN Skin Analysis: Integrates a skin_classifier.tflite scaffold targeting Normal vs. Red/Flushing (with Pallor removed).

- Dispatch: Data is transmitted to the nurse dashboard only after the case is complete. Patients who decline the PDPA notice are routed immediately to the manual desk.
>>>>>>> 5746577930f32deff477db86811a3bd4f0d589e5
