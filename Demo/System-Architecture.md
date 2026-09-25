# System Architecture

┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   1. OFFLINE TRAINING & MODEL OPTIMIZATION PIPELINE                    │
│                                                                                        │
│  [Dataset Collection: Normal Facial Pigmentation & Clinical Erythema/Flushing]         │
│        │                                                                               │
│  [Landmark Preprocessing: MediaPipe Face Mesh ──► Bilateral Cheek ROI Extraction]      │
│        │                                                                               │
│  [Color Space Transformation: sRGB ──► CIELAB (L*a*b*) ──► Data Augmentation]          │
│        │                                                                               │
│  [Deep Learning: 4-Layer Convolutional Neural Network (CNN) in TensorFlow/Keras]       │
│        │                                                                               │
│  [Model Compression: TensorFlow Lite (TFLite) FlatBuffer Export & INT8 Quantization]   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Deploys skin_model.tflite
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                    2. ONLINE REAL-TIME KIOSK INFERENCE PIPELINE                        │
│                                                                                        │
│  [Sensory Ingestion: 1080p Optical Webcam (30 FPS) + Directional Microphone (16 kHz)]  │
│        │                                                                               │
│  [Multi-Core Concurrency & Parallel Execution Engine (Apple M1 Mac / 8GB RAM)]         │
│   ├── Thread 1 (Vision Engine): MediaPipe Pose + Pain Geometry + CIELAB CNN Inference  │
│   ├── Thread 2 (Audio Engine): VAD + Faster-Whisper ASR + Audio Template Caching       │
│   └── Thread 3 (Reasoning Engine): Multimodal Fused Context ──► 4-Bit Llama 3.2 (Ollama│
│        │                                                                               │
│  [Clinical Handoff: Spool Outbox (CASE-xxxx.json) ──► Node.js / SQL Nurse Dashboard]   │
└────────────────────────────────────────────────────────────────────────────────────────┘