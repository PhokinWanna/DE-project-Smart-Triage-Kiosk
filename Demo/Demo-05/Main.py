"""
Main.py - Master Kiosk Orchestrator (Step 5)
Wake-Word Gated FSM, Continuous Acute Preemption, and Transactional Outbox Spooler.
"""

import os
import cv2
import time
import json
import queue
import base64
import threading
import requests

from config import SystemConfig, KioskState, ExitReason
from V_Module import VisionEngine
from A_Module import AudioEngine
from R_Module import ReasoningEngine


class ConveyorBeltWorker(threading.Thread):
    """
    Background Outbox Spooler (The Conveyor Belt).
    Decouples kiosk front-end interactions from backend network latency.
    """
    def __init__(self, spool_dir: str, endpoint: str):
        super().__init__(daemon=True)
        self.spool_dir = spool_dir
        self.endpoint = endpoint
        self.queue = queue.Queue()
        self.running = True
        os.makedirs(self.spool_dir, exist_ok=True)

    def enqueue_case(self, case_container: dict):
        """Drops case into in-memory queue and saves a persistent copy to disk."""
        case_id = case_container.get("case_id", f"CASE-{int(time.time())}")
        disk_path = os.path.join(self.spool_dir, f"{case_id}.json")

        try:
            with open(disk_path, "w", encoding="utf-8") as f:
                json.dump(case_container, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[!] Warning: Failed to spool case to disk: {e}")

        self.queue.put((case_id, disk_path, case_container))
        print(f"[+] [Conveyor Belt]: Enqueued {case_id} for asynchronous transmission.")

    def run(self):
        while self.running:
            try:
                case_id, disk_path, payload = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            transmitted = False
            while not transmitted and self.running:
                try:
                    res = requests.post(self.endpoint, json=payload, timeout=SystemConfig.HTTP_TIMEOUT)
                    if res.status_code in [200, 201]:
                        print(f"[✓] [Conveyor Belt]: Case {case_id} dispatched to Nurse Dashboard (HTTP {res.status_code}).")
                        transmitted = True
                        if os.path.exists(disk_path):
                            os.remove(disk_path)
                    else:
                        print(f"[!] [Conveyor Belt]: Dashboard returned status {res.status_code}. Retrying in {SystemConfig.CONVEYOR_RETRY_INTERVAL}s...")
                        time.sleep(SystemConfig.CONVEYOR_RETRY_INTERVAL)
                except requests.exceptions.RequestException:
                    print(f"[-] [Conveyor Belt]: Nurse Station offline. Holding case in spool queue. Retrying in {SystemConfig.CONVEYOR_RETRY_INTERVAL}s...")
                    time.sleep(SystemConfig.CONVEYOR_RETRY_INTERVAL)

            self.queue.task_done()


class SmartReceptionTriageKiosk:
    def __init__(self):
        print(">> Powering on Stark Industries Smart Reception Triage Kiosk...")
        self.vision = VisionEngine()
        self.audio = AudioEngine()
        self.reasoning = ReasoningEngine()

        # Arm the Conveyor Belt Outbox Worker
        self.conveyor_belt = ConveyorBeltWorker(
            spool_dir=SystemConfig.OUTBOX_SPOOL_DIR,
            endpoint=SystemConfig.NURSE_DASHBOARD_ENDPOINT
        )
        self.conveyor_belt.start()

        # State Machine Flags
        self.current_state = KioskState.PREPARE_STANDBY
        self.active_language = "th"
        self.running = True
        self.current_strikes = 0

        # Thread-safe Perception Stream
        self.latest_telemetry = {}
        self.telemetry_lock = threading.Lock()

        # Active Session Data Container
        self.case_container = {}
        self._reset_case_container()

    def _reset_case_container(self):
        """Initializes a clean, immutable clinical container for an encounter."""
        self.current_strikes = 0
        self.case_container = {
            "case_id": f"CASE-{int(time.time())}",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "language": self.active_language,
            "pdpa_consented": False,
            "exit_reason": None,
            "conversation": [],
            "cumulative_visual_signs": {
                "chest_clutching": False,
                "abdominal_clutching": False,
                "head_clutching": False
            },
            "facial_skin_status": "NORMAL",
            "visual_evidence_b64": None,
            "triage_verdict": None
        }

    def _video_stream_worker(self):
        """Dedicated background thread maintaining dual camera feeds at 30 FPS."""
        cap = cv2.VideoCapture(SystemConfig.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, SystemConfig.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SystemConfig.FRAME_HEIGHT)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Process through V_Module (30 FPS Pose + 7s Throttled CNN)
            telemetry = self.vision.process_frame(frame)

            with self.telemetry_lock:
                self.latest_telemetry = telemetry

            # Render dual display windows
            if SystemConfig.SHOW_DUAL_WINDOWS:
                cv2.imshow("Kiosk - Clean Patient View", telemetry["clean_view"])
                cv2.imshow("Kiosk - Diagnostic Skeleton View", telemetry["skeleton_view"])
            else:
                cv2.imshow("Smart Triage Kiosk", telemetry["clean_view"])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cap.release()
        cv2.destroyAllWindows()

    def _update_cumulative_visuals(self):
        """Thread-safe update of visual signs accumulated over the session."""
        with self.telemetry_lock:
            t = self.latest_telemetry
            for key in ["chest_clutching", "abdominal_clutching", "head_clutching"]:
                if t.get(key, False):
                    self.case_container["cumulative_visual_signs"][key] = True

            self.case_container["facial_skin_status"] = t.get("skin_status", "NORMAL")

            # Capture visual evidence snapshot
            if "skeleton_view" in t and self.case_container["visual_evidence_b64"] is None:
                _, buf = cv2.imencode('.jpg', t["skeleton_view"])
                self.case_container["visual_evidence_b64"] = base64.b64encode(buf).decode('utf-8')

    def run(self):
        """Master FSM Lifecycle Loop."""
        # 1. Start continuous 30 FPS video thread
        video_thread = threading.Thread(target=self._video_stream_worker, daemon=True)
        video_thread.start()

        print("[*] All subsystems nominal. Kiosk entering PREPARE_STANDBY.")

        try:
            while self.running:
                # -------------------------------------------------------------
                # 1. State: PREPARE_STANDBY (Wake-Word Gate)
                # -------------------------------------------------------------
                if self.current_state == KioskState.PREPARE_STANDBY:
                    with self.telemetry_lock:
                        patient_in_zone = self.latest_telemetry.get("patient_present", False)

                    if patient_in_zone:
                        print("[*] Person in triage zone. Listening for wake trigger ('สวัสดี' / 'Hello')...")
                        # Lightweight ambient listen for wake-word
                        ambient_speech = self.audio.listen(language="th")
                        triggered, detected_lang = self.audio.detect_wake_word(ambient_speech)

                        if triggered:
                            print(f"[+] Wake word verified ({detected_lang}). Launching intake encounter...")
                            self._reset_case_container()
                            self.active_language = detected_lang
                            self.case_container["language"] = self.active_language
                            self.current_state = KioskState.GREETING_LANG_SELECT

                # -------------------------------------------------------------
                # 2. State: GREETING_LANG_SELECT
                # -------------------------------------------------------------
                elif self.current_state == KioskState.GREETING_LANG_SELECT:
                    # Instant playback of cached audio template
                    self.audio.play_template("greeting", lang=self.active_language, block=True)

                    lang_utterance = self.audio.listen(language=self.active_language)
                    print(f"[Language Negotiation Input]: \"{lang_utterance}\"")

                    # Lock case language
                    if any(w in lang_utterance.lower() for w in ["english", "eng", "hello"]):
                        self.active_language = "en"
                    else:
                        self.active_language = "th"

                    self.case_container["language"] = self.active_language
                    self.current_state = KioskState.PDPA_CONSENT

                # -------------------------------------------------------------
                # 3. State: PDPA_CONSENT (Privacy Gatekeeper)
                # -------------------------------------------------------------
                elif self.current_state == KioskState.PDPA_CONSENT:
                    # Instant playback of cached PDPA audio template
                    self.audio.play_template("pdpa_notice", lang=self.active_language, block=True)

                    consent_reply = self.audio.listen(language=self.active_language)
                    self.case_container["conversation"].append({"role": "pdpa_response", "text": consent_reply})

                    # Evaluate consent refusal
                    deny_tokens = ["ไม่", "ปฏิเสธ", "no", "deny", "disagree", "not accept"]
                    if any(t in consent_reply.lower() for t in deny_tokens):
                        print("[-] Patient denied PDPA consent. Transitioning to MANUAL_ROUTING...")
                        self.case_container["exit_reason"] = ExitReason.PDPA_DENIED
                        self.current_state = KioskState.MANUAL_ROUTING
                    else:
                        self.case_container["pdpa_consented"] = True
                        self.current_state = KioskState.SYMPTOM_INTERVIEW

                # -------------------------------------------------------------
                # 3b. State: MANUAL_ROUTING (Patient Refused PDPA)
                # -------------------------------------------------------------
                elif self.current_state == KioskState.MANUAL_ROUTING:
                    self.audio.play_template("pdpa_denied", lang=self.active_language, block=True)
                    time.sleep(2.0)
                    print("[*] Patient rerouted to human desk. Resetting kiosk to PREPARE_STANDBY.\n" + "="*50)
                    self.current_state = KioskState.PREPARE_STANDBY

                # -------------------------------------------------------------
                # 4. State: SYMPTOM_INTERVIEW (Multimodal Clinical Encounter)
                # -------------------------------------------------------------
                elif self.current_state == KioskState.SYMPTOM_INTERVIEW:
                    # Play cached inquiry template
                    self.audio.play_template("inquiry", lang=self.active_language, block=True)

                    interview_active = True
                    while interview_active and self.running:
                        # 1. Listen with 2.5s silence pause threshold (Local Fast-Whisper)
                        patient_speech = self.audio.listen(language=self.active_language)
                        self.case_container["conversation"].append({"role": "patient", "text": patient_speech})

                        # Update accumulated visual telemetry
                        self._update_cumulative_visuals()

                        # 2. Check for Immediate Acute Preemption (Cardiac Arrest / Syncope / Collapse)
                        is_preempted, preempt_esi = self.reasoning.check_acute_preemption(
                            patient_speech,
                            self.case_container["cumulative_visual_signs"]
                        )
                        if is_preempted:
                            print(f"[!] EMERGENCY INTERRUPT TRIGGERED: Fast-tracking ESI Level {preempt_esi}!")
                            self.case_container["exit_reason"] = ExitReason.EMERGENCY_INTERRUPT
                            interview_active = False
                            self.current_state = KioskState.FINALIZING_CASE
                            break

                        # 3. Check Conversational Coherence & 3-Strike Monitor
                        is_off_topic, self.current_strikes, pivot_script = self.reasoning.check_relevance_and_pivot(
                            patient_speech,
                            self.current_strikes,
                            lang=self.active_language
                        )

                        if is_off_topic:
                            if self.current_strikes >= SystemConfig.MAX_OFFTOPIC_STRIKES:
                                print("[!] 3-Strike Rule Tripped: Altered Mental Status (AMS) Diagnosed.")
                                self.case_container["exit_reason"] = ExitReason.AMS_3_STRIKES
                                interview_active = False
                                self.current_state = KioskState.FINALIZING_CASE
                                break
                            else:
                                # Speak dynamic Zero-PII steering pivot
                                self.audio.speak_dynamic(pivot_script, lang=self.active_language, block=True)
                                continue

                        # 4. On-Topic Answer: Valid symptom gathered
                        print("[+] On-Topic Clinical Symptom successfully registered.")
                        self.case_container["exit_reason"] = ExitReason.NORMAL_COMPLETE
                        interview_active = False
                        self.current_state = KioskState.FINALIZING_CASE

                # -------------------------------------------------------------
                # 5. State: FINALIZING_CASE (Clinical Synthesis & SOAP Generation)
                # -------------------------------------------------------------
                elif self.current_state == KioskState.FINALIZING_CASE:
                    print(f"[*] Packaging case [{self.case_container['case_id']}] (Exit Reason: {self.case_container['exit_reason'].value})...")
                    self._update_cumulative_visuals()

                    # Holistic Clinical Synthesis
                    verdict = self.reasoning.evaluate_final_case(
                        conversation_history=self.case_container["conversation"],
                        visual_signs=self.case_container["cumulative_visual_signs"],
                        skin_status=self.case_container["facial_skin_status"],
                        exit_reason=self.case_container["exit_reason"],
                        language=self.active_language
                    )
                    self.case_container["triage_verdict"] = verdict
                    print(f"[VERDICT]: ESI Level {verdict.get('esi_level')} ({verdict.get('urgency')})")
                    self.current_state = KioskState.DISPATCH_AND_RESET

                # -------------------------------------------------------------
                # 6. State: DISPATCH_AND_RESET (Conveyor Belt Transport)
                # -------------------------------------------------------------
                elif self.current_state == KioskState.DISPATCH_AND_RESET:
                    exit_r = self.case_container["exit_reason"]

                    # 1. Play appropriate closing audio template
                    if exit_r == ExitReason.EMERGENCY_INTERRUPT:
                        self.audio.play_template("emergency_alert", lang=self.active_language, block=True)
                    elif exit_r == ExitReason.AMS_3_STRIKES:
                        self.audio.play_template("ams_alert", lang=self.active_language, block=True)
                    else:
                        self.audio.play_template("wrapup", lang=self.active_language, block=True)

                    # 2. Drop the container onto the Conveyor Belt (<1 millisecond)
                    self.conveyor_belt.enqueue_case(self.case_container)

                    # 3. Clean reset to standby
                    print("[*] Encounter finalized. Resetting to PREPARE_STANDBY.\n" + "="*55)
                    self.current_state = KioskState.PREPARE_STANDBY

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n[!] Emergency shutoff signal received...")
        finally:
            self.running = False
            self.conveyor_belt.running = False
            video_thread.join()
            self.vision.release()
            self.audio.release()
            print("[+] Kiosk successfully and safely offline.")

if __name__ == "__main__":
    kiosk = SmartReceptionTriageKiosk()
    kiosk.run()