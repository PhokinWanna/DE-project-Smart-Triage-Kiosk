# Multimodal Pre-Triage Assistance Using Computer Vision and a Local Large Language Model: A Controlled Simulation Study

A smart, multi-modal medical pre-triage kiosk system developed as part of the Digital Engineering Project (977-302). This repository demonstrates the integration of edge computing, computer vision, and local Large Language Models (LLMs) to streamline patient intake and assessment.

---

## Overview & System Architecture
Provide a brief, 2-3 sentence summary of the project's core functionality here. Explain how the system assists with medical pre-triage through vision and voice.

![System Architecture](Doc/DE%20Project-2/Additional/6630613024_PhokinWanna-DEII-Report-V4.jpg) <!-- Or path to an image file like docs/architecture.png -->

---

## Key Features
- **Skin Condition Assessment:** Computer vision tools designed to analyze visual symptoms.
- **Local LLM Reasoning:** Patient interaction powered by a localized LLM (e.g., Llama) running entirely on-premise for strict data privacy.
- **Multimodal Inputs:** Support for voice (microphone) and visual tracking for an intuitive triage kiosk simulation.
- **Controlled Simulation:** A fully configured testing playground containing benchmarking scripts for model performance evaluation.

---

## Tech Stack & Architecture
- **AI & Core Frameworks:** Python, Ollama / Llama (Local LLM Deployment), CNNs / Vision Models
- **Signal & Hardware Interface:** OpenCV (Gesture/Vision), SoundDevice/SpeechRecognition (Microphone Interaction)
- **Environment Management:** Virtual environments with deterministic package mapping via `requirements.txt`.

---

## Project Structure
```text
├── docs/                 # System architecture, documentation, and research papers
├── src/                  # Core production source code
│   ├── models/           # LLM interaction handlers and CNN vision scripts
│   └── utils/            # Helper scripts for gesture tracking and microphone input
├── tests/                # Benchmarking and playground scripts (formerly Test-Playground)
├── .gitignore            # Git exclusion rules
├── README.md             # Project documentation
└── requirements.txt      # Dependency specification
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.com) (configured with your choice of local Llama model)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com
   cd DE-project-Smart-Triage-Kiosk
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset & Simulation Data
The structural directories for `Train-dataset` contain sample structures used during model calibration. Large binary dataset objects and local development environment folders are intentionally omitted via `.gitignore` to maintain a production-ready codebase.
