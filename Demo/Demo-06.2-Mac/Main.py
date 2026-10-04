# """
# Main.py - Master Kiosk Orchestrator (macOS Cocoa Main-Thread Compliant)
# Main-Thread GUI Rendering, Background FSM Triage, and Asynchronous Outbox.
# """
# import os
# import cv2
# import time
# import json
# import queue
# import base64
# import threading
# import requests
# from config import SystemConfig, KioskState, ExitReason
# from V_Module import VisionEngine
# from A_Module import AudioEngine
# from R_Module import ReasoningEngine

# class ConveyorBeltWorker(threading.Thread):
#     """
#     Background Outbox Spooler (The Conveyor Belt).
#     Ensures network latency never blocks front-end kiosk operations.
#     """
#     def __init__(self, spool_dir: str, endpoint: str):
#         super().__init__(daemon=True)
#         self.spool_dir = spool_dir
#         self.endpoint = endpoint
#         self.queue = queue.Queue()
#         self.running = True
#         os.makedirs(self.spool_dir, exist_ok=True)

#     def enqueue_case(self, case_container: dict):
#         """Saves local persistent backup and queues payload for background dispatch."""
#         case_id = case_container.get("case_id", f"CASE-{int(time.time())}")
#         disk_path = os.path.join(self.spool_dir, f"{case_id}.json")

#         try:
#             with open(disk_path, "w", encoding="utf-8") as f:
#                 json.dump(case_container, f, ensure_ascii=False, indent=2)
#         except Exception as e:
#             print(f"[!] Warning: Failed to spool case to local disk: {e}")

#         self.queue.put((case_id, disk_path, case_container))
#         print(f"[+] [Conveyor Belt]: Case {case_id} queued for background transmission.")

#     def run(self):
#         while self.running:
#             try:
#                 case_id, disk_path, payload = self.queue.get(timeout=1.0)
#             except queue.Empty:
#                 continue

#             transmitted = False
#             while not transmitted and self.running:
#                 try:
#                     res = requests.post(self.endpoint, json=payload, timeout=SystemConfig.HTTP_TIMEOUT)
#                     if res.status_code in [200, 201]:
#                         print(f"[✓] [Conveyor Belt]: Case {case_id} successfully delivered to Nurse Dashboard.")
#                         transmitted = True
#                         if os.path.exists(disk_path):
#                             os.remove(disk_path)
#                     else:
#                         print(f"[!] [Conveyor Belt]: Dashboard returned status {res.status_code}. Retrying in {SystemConfig.CONVEYOR_RETRY_INTERVAL}s...")
#                         time.sleep(SystemConfig.CONVEYOR_RETRY_INTERVAL)
#                 except requests.exceptions.RequestException:
#                     print(f"[-] [Conveyor Belt]: Nurse Station offline. Holding case in spool. Retrying in {SystemConfig.CONVEYOR_RETRY_INTERVAL}s...")
#                     time.sleep(SystemConfig.CONVEYOR_RETRY_INTERVAL)

#             self.queue.task_done()

# class SmartReceptionTriageKiosk:
#     def __init__(self):
#         print(">> Initializing Stark Industries Smart Reception Triage Kiosk...")
#         self.vision = VisionEngine()
#         self.audio = AudioEngine()
#         self.reasoning = ReasoningEngine()

#         # VRAM Keep-Alive Heartbeat Timer
#         self.last_heartbeat = time.time()

#         # Arm the Conveyor Belt Outbox Worker
#         self.conveyor_belt = ConveyorBeltWorker(
#             spool_dir=SystemConfig.OUTBOX_SPOOL_DIR,
#             endpoint=SystemConfig.NURSE_DASHBOARD_ENDPOINT
#         )
#         self.conveyor_belt.start()

#         # FSM State & Session Properties
#         self.current_state = KioskState.PREPARE_STANDBY
#         self.active_language = "th"
#         self.running = True
#         self.current_strikes = 0

#         # Thread-safe Perception Telemetry
#         self.latest_telemetry = {}
#         self.telemetry_lock = threading.Lock()

#         # Active Case Container
#         self.case_container = {}
#         self._reset_case_container()

#     def _reset_case_container(self):
#         """Initializes a fresh, clinical container for the encounter."""
#         self.current_strikes = 0
#         self.case_container = {
#             "case_id": f"CASE-{int(time.time())}",
#             "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
#             "language": self.active_language,
#             "pdpa_consented": False,
#             "exit_reason": None,
#             "conversation": [],
#             "cumulative_visual_signs": {
#                 "chest_clutching": False,
#                 "abdominal_clutching": False,
#                 "head_clutching": False
#             },
#             "facial_skin_status": "NORMAL",
#             "visual_evidence_b64": None,
#             "triage_verdict": None
#         }

#     def _update_cumulative_visuals(self):
#         """Aggregates kinetic and facial telemetry into the current case container."""
#         with self.telemetry_lock:
#             t = self.latest_telemetry
#             for key in ["chest_clutching", "abdominal_clutching", "head_clutching"]:
#                 if t.get(key, False):
#                     self.case_container["cumulative_visual_signs"][key] = True

#             self.case_container["facial_skin_status"] = t.get("skin_status", "NORMAL")

#             # Capture visual evidence snapshot for the nurse dashboard
#             if "skeleton_view" in t and self.case_container["visual_evidence_b64"] is None:
#                 _, buf = cv2.imencode('.jpg', t["skeleton_view"])
#                 self.case_container["visual_evidence_b64"] = base64.b64encode(buf).decode('utf-8')

#     def _fsm_worker(self):
#         """
#         Background Triage FSM Worker.
#         Executes dialog turn-taking, speech listening, and LLM reasoning
#         without blocking the main thread's 30 FPS camera feed.
#         """
#         time.sleep(1.0) # Allow camera display to initialize
#         print("[*] Triage Reception Agent is active in background.")

#         while self.running:
#             try:
#                 # -------------------------------------------------------------
#                 # 1. State: PREPARE_STANDBY (Wake-Word Gate & Heartbeat)
#                 # -------------------------------------------------------------
#                 if self.current_state == KioskState.PREPARE_STANDBY:
#                     if (time.time() - self.last_heartbeat) >= 240.0:
#                         print("[*] [Heartbeat]: Refreshing VRAM residency...")
#                         self.reasoning.warmup()
#                         self.last_heartbeat = time.time()

#                     with self.telemetry_lock:
#                         patient_in_zone = self.latest_telemetry.get("patient_present", False)

#                     if patient_in_zone:
#                         print("[*] Patient in triage zone. Listening for speech or wake word...")
#                         ambient_speech = self.audio.listen(language="th")
#                         triggered, detected_lang = self.audio.detect_wake_word(ambient_speech)

#                         if triggered or len(ambient_speech) > 0:
#                             print(f"[+] Patient engagement confirmed (speech: '{ambient_speech}'). Launching intake...")
#                             self._reset_case_container()
#                             self.active_language = detected_lang if triggered else "th"
#                             self.case_container["language"] = self.active_language
#                             self.current_state = KioskState.GREETING_LANG_SELECT

#                 # -------------------------------------------------------------
#                 # 2. State: GREETING_LANG_SELECT
#                 # -------------------------------------------------------------
#                 elif self.current_state == KioskState.GREETING_LANG_SELECT:
#                     self.audio.play_template("greeting", lang=self.active_language, block=True)
#                     lang_reply = self.audio.listen(language=self.active_language)
#                     print(f"[Language Preference]: \"{lang_reply}\"")

#                     if any(w in lang_reply.lower() for w in ["english", "eng", "hello"]):
#                         self.active_language = "en"
#                     else:
#                         self.active_language = "th"

#                     self.case_container["language"] = self.active_language
#                     self.current_state = KioskState.PDPA_CONSENT

#                 # -------------------------------------------------------------
#                 # 3. State: PDPA_CONSENT
#                 # -------------------------------------------------------------
#                 elif self.current_state == KioskState.PDPA_CONSENT:
#                     self.audio.play_template("pdpa_notice", lang=self.active_language, block=True)
#                     consent_reply = self.audio.listen(language=self.active_language)
#                     self.case_container["conversation"].append({"role": "pdpa_response", "text": consent_reply})

#                     deny_tokens = ["ไม่", "ปฏิเสธ", "no", "deny", "disagree", "not consent"]
#                     if any(t in consent_reply.lower() for t in deny_tokens):
#                         print("[-] Patient refused PDPA consent. Transitioning to MANUAL_ROUTING...")
#                         self.case_container["exit_reason"] = ExitReason.PDPA_DENIED.value
#                         self.current_state = KioskState.MANUAL_ROUTING
#                     else:
#                         self.case_container["pdpa_consented"] = True
#                         self.current_state = KioskState.SYMPTOM_INTERVIEW

#                 # -------------------------------------------------------------
#                 # 3b. State: MANUAL_ROUTING
#                 # -------------------------------------------------------------
#                 elif self.current_state == KioskState.MANUAL_ROUTING:
#                     self.audio.play_template("pdpa_denied", lang=self.active_language, block=True)
#                     time.sleep(2.0)
#                     print("[*] Patient routed to human desk. Resetting to PREPARE_STANDBY.\n" + "="*55)
#                     self.current_state = KioskState.PREPARE_STANDBY

#                 # -------------------------------------------------------------
#                 # 4. State: SYMPTOM_INTERVIEW (Interactive Micro-Dialogue)
#                 # -------------------------------------------------------------
#                 elif self.current_state == KioskState.SYMPTOM_INTERVIEW:
#                     self.audio.play_template("inquiry", lang=self.active_language, block=True)

#                     interview_active = True
#                     consecutive_pauses = 0

#                     while interview_active and self.running:
#                         chunk = self.audio.listen(language=self.active_language)
#                         self._update_cumulative_visuals()

#                         # Silence Handling
#                         if len(chunk) == 0:
#                             consecutive_pauses += 1
#                             if consecutive_pauses == 1:
#                                 self.audio.play_template("still_listening", lang=self.active_language, block=True)
#                                 continue
#                             elif consecutive_pauses == 2:
#                                 self.audio.play_template("anything_else", lang=self.active_language, block=True)
#                                 continue
#                             else:
#                                 print("[*] Extended silence verified. Finalizing symptom collection.")
#                                 self.case_container["exit_reason"] = ExitReason.NORMAL_COMPLETE.value
#                                 interview_active = False
#                                 break

#                         consecutive_pauses = 0
#                         self.case_container["conversation"].append({"role": "patient", "text": chunk})

#                         # Acute Emergency Preemption Check
#                         is_preempted, preempt_esi = self.reasoning.check_acute_preemption(
#                             chunk,
#                             self.case_container["cumulative_visual_signs"]
#                         )
#                         if is_preempted:
#                             print(f"[!] ACUTE EMERGENCY PREEMPTION TRIGGERED: ESI Level {preempt_esi}!")
#                             self.case_container["exit_reason"] = ExitReason.EMERGENCY_INTERRUPT.value
#                             interview_active = False
#                             break

#                         # Explicit Stopping Intent Check
#                         if self.reasoning.is_stopping_phrase(chunk, lang=self.active_language):
#                             print("[*] Patient explicitly requested completion.")
#                             self.case_container["exit_reason"] = ExitReason.NORMAL_COMPLETE.value
#                             interview_active = False
#                             break

#                         # Clinical Turn Evaluation
#                         turn_eval = self.reasoning.evaluate_turn_and_probe(
#                             utterance=chunk,
#                             history=self.case_container["conversation"],
#                             current_strikes=self.current_strikes,
#                             lang=self.active_language
#                         )
#                         self.current_strikes = turn_eval["strikes"]

#                         if turn_eval["ams_triggered"]:
#                             print("[!] 3-Strike Rule Tripped: Altered Mental Status (AMS) Diagnosed.")
#                             self.case_container["exit_reason"] = ExitReason.AMS_3_STRIKES.value
#                             interview_active = False
#                             break

#                         elif turn_eval["action"] == "STEER_PIVOT":
#                             self.audio.speak_dynamic(turn_eval["speech_output"], lang=self.active_language, block=True)
#                             self.case_container["conversation"].append({"role": "kiosk_pivot", "text": turn_eval["speech_output"]})
#                             continue

#                         elif turn_eval["action"] == "PROBE_QUESTION":
#                             self.audio.speak_dynamic(turn_eval["speech_output"], lang=self.active_language, block=True)
#                             self.case_container["conversation"].append({"role": "kiosk_probe", "text": turn_eval["speech_output"]})
#                             continue

#                         elif turn_eval["action"] == "DETAILS_SUFFICIENT":
#                             self.audio.play_template("anything_else", lang=self.active_language, block=True)
#                             final_chunk = self.audio.listen(language=self.active_language)

#                             if len(final_chunk) > 0 and not self.reasoning.is_stopping_phrase(final_chunk, lang=self.active_language):
#                                 self.case_container["conversation"].append({"role": "patient", "text": final_chunk})

#                             self.case_container["exit_reason"] = ExitReason.NORMAL_COMPLETE.value
#                             interview_active = False
#                             break

#                     self.current_state = KioskState.FINALIZING_CASE

#                 # -------------------------------------------------------------
#                 # 5. State: FINALIZING_CASE (Clinical Synthesis)
#                 # -------------------------------------------------------------
#                 elif self.current_state == KioskState.FINALIZING_CASE:
#                     raw_exit = self.case_container.get("exit_reason", ExitReason.NORMAL_COMPLETE.value)
#                     exit_str = raw_exit.value if hasattr(raw_exit, "value") else str(raw_exit)

#                     print(f"[*] Packaging case [{self.case_container['case_id']}] (Exit: {exit_str})...")
#                     self._update_cumulative_visuals()

#                     verdict = self.reasoning.evaluate_final_case(
#                         conversation_history=self.case_container["conversation"],
#                         visual_signs=self.case_container["cumulative_visual_signs"],
#                         skin_status=self.case_container["facial_skin_status"],
#                         exit_reason=raw_exit,
#                         language=self.active_language
#                     )
#                     self.case_container["triage_verdict"] = verdict
#                     print(f"[VERDICT]: ESI Level {verdict.get('esi_level')} ({verdict.get('urgency')})")
#                     self.current_state = KioskState.DISPATCH_AND_RESET

#                 # -------------------------------------------------------------
#                 # 6. State: DISPATCH_AND_RESET
#                 # -------------------------------------------------------------
#                 elif self.current_state == KioskState.DISPATCH_AND_RESET:
#                     raw_exit = self.case_container.get("exit_reason", ExitReason.NORMAL_COMPLETE.value)
#                     exit_str = raw_exit.value if hasattr(raw_exit, "value") else str(raw_exit)

#                     if exit_str == ExitReason.EMERGENCY_INTERRUPT.value:
#                         self.audio.play_template("emergency_alert", lang=self.active_language, block=True)
#                     elif exit_str == ExitReason.AMS_3_STRIKES.value:
#                         self.audio.play_template("ams_alert", lang=self.active_language, block=True)
#                     else:
#                         self.audio.play_template("wrapup", lang=self.active_language, block=True)

#                     self.conveyor_belt.enqueue_case(self.case_container)
#                     print("[*] Case finalized. Kiosk returning to PREPARE_STANDBY.\n" + "="*55)
#                     self.current_state = KioskState.PREPARE_STANDBY

#                 time.sleep(0.05)

#             except Exception as e:
#                 print(f"[!] FSM Encounter Exception: {e}")
#                 time.sleep(1.0)

#     def run(self):
#         """
#         Main-Thread Execution Loop.
#         macOS Cocoa requires all NSWindow / cv2.imshow calls on the main thread.
#         """
#         # Launch FSM & Dialogue loop in a background daemon thread
#         fsm_thread = threading.Thread(target=self._fsm_worker, daemon=True)
#         fsm_thread.start()

#         cap = cv2.VideoCapture(SystemConfig.CAMERA_INDEX)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, SystemConfig.FRAME_WIDTH)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SystemConfig.FRAME_HEIGHT)

#         print("[*] Camera capture active on Main Thread.")

#         try:
#             while self.running:
#                 ret, frame = cap.read()
#                 if not ret:
#                     time.sleep(0.01)
#                     continue

#                 # Process perception
#                 telemetry = self.vision.process_frame(frame)
#                 with self.telemetry_lock:
#                     self.latest_telemetry = telemetry

#                 # Render display windows on Main Thread (Cocoa Safe)
#                 if SystemConfig.SHOW_DUAL_WINDOWS:
#                     cv2.imshow("Kiosk - Clean Patient View", telemetry["clean_view"])
#                     cv2.imshow("Kiosk - Diagnostic Skeleton View", telemetry["skeleton_view"])
#                 else:
#                     cv2.imshow("Smart Triage Kiosk", telemetry["clean_view"])

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     self.running = False
#                     break

#         except KeyboardInterrupt:
#             print("\n[!] Powering down systems gracefully...")
#         finally:
#             self.running = False
#             self.conveyor_belt.running = False
#             cap.release()
#             cv2.destroyAllWindows()
#             self.vision.release()
#             self.audio.release()
#             print("[+] Kiosk successfully and safely offline.")

# if __name__ == "__main__":
#     kiosk = SmartReceptionTriageKiosk()
#     kiosk.run()


"""
Main.py - Master Kiosk Orchestrator (Agentic Action Architecture)
Executes Agent Actions, Local JPEG Evidence Storage, and Proactive Visual Inquiries.
"""

import os
import cv2
import time
import json
import queue
import threading
import requests

from config import SystemConfig, KioskState, ExitReason
from V_Module import VisionEngine
from A_Module import AudioEngine
from R_Module import ReasoningEngine


class ConveyorBeltWorker(threading.Thread):
    """Background Outbox Spooler (The Conveyor Belt)."""
    def __init__(self, spool_dir: str, endpoint: str):
        super().__init__(daemon=True)
        self.spool_dir = spool_dir
        self.endpoint = endpoint
        self.queue = queue.Queue()
        self.running = True
        os.makedirs(self.spool_dir, exist_ok=True)

    def enqueue_case(self, case_container: dict):
        case_id = case_container.get("case_id", f"CASE-{int(time.time())}")
        disk_path = os.path.join(self.spool_dir, f"{case_id}.json")

        try:
            with open(disk_path, "w", encoding="utf-8") as f:
                json.dump(case_container, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[!] Warning: Failed to spool case to local disk: {e}")

        self.queue.put((case_id, disk_path, case_container))
        print(f"[+] [Conveyor Belt]: Case {case_id} queued for background transmission.")

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
                        print(f"[✓] [Conveyor Belt]: Case {case_id} delivered to Nurse Dashboard.")
                        transmitted = True
                        if os.path.exists(disk_path):
                            os.remove(disk_path)
                    else:
                        print(f"[!] [Conveyor Belt]: Dashboard status {res.status_code}. Retrying in {SystemConfig.CONVEYOR_RETRY_INTERVAL}s...")
                        time.sleep(SystemConfig.CONVEYOR_RETRY_INTERVAL)
                except requests.exceptions.RequestException:
                    print(f"[-] [Conveyor Belt]: Dashboard unreachable. Retrying in {SystemConfig.CONVEYOR_RETRY_INTERVAL}s...")
                    time.sleep(SystemConfig.CONVEYOR_RETRY_INTERVAL)

            self.queue.task_done()


class SmartReceptionTriageKiosk:
    def __init__(self):
        print(">> Initializing Stark Industries Smart Reception Triage Kiosk (Agentic Mode)...")
        self.vision = VisionEngine()
        self.audio = AudioEngine()
        self.reasoning = ReasoningEngine()

        self.last_heartbeat = time.time()

        # Arm the Conveyor Belt Outbox Worker
        self.conveyor_belt = ConveyorBeltWorker(
            spool_dir=SystemConfig.OUTBOX_SPOOL_DIR,
            endpoint=SystemConfig.NURSE_DASHBOARD_ENDPOINT
        )
        self.conveyor_belt.start()

        # Evidence folder setup
        self.evidence_dir = getattr(SystemConfig, "EVIDENCE_DIR", os.path.join(SystemConfig.BASE_DIR, "evidence"))
        os.makedirs(self.evidence_dir, exist_ok=True)

        self.current_state = KioskState.PREPARE_STANDBY
        self.active_language = "th"
        self.running = True
        self.current_strikes = 0

        self.latest_telemetry = {}
        self.telemetry_lock = threading.Lock()

        self.case_container = {}
        self._reset_case_container()

    def _reset_case_container(self):
        """Initializes a fresh clinical case container."""
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
            "visual_evidence_path": None,
            "triage_verdict": None
        }

    def _update_cumulative_visuals(self):
        """Updates sustained gestures and captures clean JPEG evidence without Base64 clutter."""
        with self.telemetry_lock:
            t = self.latest_telemetry
            
            # Use SUSTAINED gestures (>=3.0s) to eliminate the passing-hand latching bug!
            sustained = t.get("sustained_gestures", {})
            for key in ["chest_clutching", "abdominal_clutching", "head_clutching"]:
                if sustained.get(key, False):
                    self.case_container["cumulative_visual_signs"][key] = True

            self.case_container["facial_skin_status"] = t.get("skin_status", "NORMAL")

            # Save clean JPEG to disk
            if "clean_view" in t and self.case_container["visual_evidence_path"] is None:
                img_path = os.path.join(self.evidence_dir, f"{self.case_container['case_id']}.jpg")
                cv2.imwrite(img_path, t["clean_view"])
                self.case_container["visual_evidence_path"] = img_path
                print(f"[i] Captured JPEG Evidence: {img_path}")

    def _fsm_worker(self):
        """Background Agentic FSM Worker."""
        time.sleep(1.0)
        print("[*] Autonomous Reception Agent is active in background.")

        while self.running:
            try:
                # -------------------------------------------------------------
                # 1. State: PREPARE_STANDBY
                # -------------------------------------------------------------
                if self.current_state == KioskState.PREPARE_STANDBY:
                    if (time.time() - self.last_heartbeat) >= 240.0:
                        self.reasoning.warmup()
                        self.last_heartbeat = time.time()

                    with self.telemetry_lock:
                        patient_in_zone = self.latest_telemetry.get("patient_present", False)

                    if patient_in_zone:
                        print("[*] Patient in triage zone. Listening for speech or wake word...")
                        ambient_speech = self.audio.listen(language="th")
                        triggered, detected_lang = self.audio.detect_wake_word(ambient_speech)

                        if triggered or len(ambient_speech) > 0:
                            print(f"[+] Patient engagement confirmed. Launching intake...")
                            self._reset_case_container()
                            self.active_language = detected_lang if triggered else "th"
                            self.case_container["language"] = self.active_language
                            self.current_state = KioskState.GREETING_LANG_SELECT

                # -------------------------------------------------------------
                # 2. State: GREETING_LANG_SELECT
                # -------------------------------------------------------------
                elif self.current_state == KioskState.GREETING_LANG_SELECT:
                    self.audio.play_template("greeting", lang=self.active_language, block=True)
                    lang_reply = self.audio.listen(language=self.active_language)
                    print(f"[Language Preference]: \"{lang_reply}\"")

                    if any(w in lang_reply.lower() for w in ["english", "eng", "hello"]):
                        self.active_language = "en"
                    else:
                        self.active_language = "th"

                    self.case_container["language"] = self.active_language
                    self.current_state = KioskState.PDPA_CONSENT

                # -------------------------------------------------------------
                # 3. State: PDPA_CONSENT
                # -------------------------------------------------------------
                elif self.current_state == KioskState.PDPA_CONSENT:
                    self.audio.play_template("pdpa_notice", lang=self.active_language, block=True)
                    consent_reply = self.audio.listen(language=self.active_language)
                    self.case_container["conversation"].append({"role": "pdpa_response", "text": consent_reply})

                    # Check for refusal
                    deny_tokens = ["ไม่", "ปฏิเสธ", "no", "deny", "disagree", "not consent"]
                    if any(t in consent_reply.lower() for t in deny_tokens):
                        print("[-] Patient refused PDPA consent. Transitioning to MANUAL_ROUTING...")
                        self.case_container["exit_reason"] = ExitReason.PDPA_DENIED.value
                        self.current_state = KioskState.MANUAL_ROUTING
                    else:
                        self.case_container["pdpa_consented"] = True
                        self.current_state = KioskState.SYMPTOM_INTERVIEW

                # -------------------------------------------------------------
                # 3b. State: MANUAL_ROUTING
                # -------------------------------------------------------------
                elif self.current_state == KioskState.MANUAL_ROUTING:
                    self.audio.play_template("pdpa_denied", lang=self.active_language, block=True)
                    time.sleep(2.0)
                    print("[*] Patient routed to human desk. Resetting to PREPARE_STANDBY.\n" + "="*55)
                    self.current_state = KioskState.PREPARE_STANDBY

                # -------------------------------------------------------------
                # 4. State: SYMPTOM_INTERVIEW (Agentic Action Loop)
                # -------------------------------------------------------------
                elif self.current_state == KioskState.SYMPTOM_INTERVIEW:
                    # PROACTIVE VISUAL INITIATION:
                    # If patient is actively clutching something upon arrival, ask about it directly!
                    with self.telemetry_lock:
                        dom_gesture = self.latest_telemetry.get("dominant_current_gesture", "NONE")

                    proactive_prompt = self.reasoning.get_proactive_visual_prompt(dom_gesture, lang=self.active_language)

                    if proactive_prompt:
                        print(f"[*] [Proactive Visual Cue]: Triggering inquiry for {dom_gesture}")
                        self.audio.speak_dynamic(proactive_prompt, lang=self.active_language, block=True)
                        self.case_container["conversation"].append({"role": "kiosk_proactive", "text": proactive_prompt})
                    else:
                        self.audio.play_template("inquiry", lang=self.active_language, block=True)

                    interview_active = True
                    consecutive_pauses = 0

                    while interview_active and self.running:
                        chunk = self.audio.listen(language=self.active_language)
                        self._update_cumulative_visuals()

                        # Silence Handling
                        if len(chunk) == 0:
                            consecutive_pauses += 1
                            if consecutive_pauses == 1:
                                self.audio.play_template("still_listening", lang=self.active_language, block=True)
                                continue
                            elif consecutive_pauses == 2:
                                self.audio.play_template("anything_else", lang=self.active_language, block=True)
                                continue
                            else:
                                print("[*] Extended silence verified. Finalizing encounter.")
                                self.case_container["exit_reason"] = ExitReason.NORMAL_COMPLETE.value
                                interview_active = False
                                break

                        consecutive_pauses = 0
                        self.case_container["conversation"].append({"role": "patient", "text": chunk})

                        # Get current telemetry snapshot for Llama
                        with self.telemetry_lock:
                            v_telemetry = self.latest_telemetry

                        # DISPATCH TO AGENT CONTROLLER (Llama 3.2 In Charge)
                        action_frame = self.reasoning.dispatch_agent_turn(
                            utterance=chunk,
                            history=self.case_container["conversation"],
                            visual_telemetry=v_telemetry,
                            current_strikes=self.current_strikes,
                            active_lang=self.active_language
                        )

                        action = action_frame.get("action", "CONTINUE_TRIAGE")
                        spoken_resp = action_frame.get("spoken_response", "")
                        self.current_strikes = action_frame.get("strikes", self.current_strikes)

                        print(f"[*] [Agent Decision]: Action={action} | Spoken=\"{spoken_resp}\"")

                        # ACTION 1: Switch Language Mid-Conversation
                        if action == "SWITCH_LANGUAGE":
                            new_lang = action_frame.get("target_language", "en")
                            print(f"[!] AGENT COMMAND: Dynamically switching language from {self.active_language} -> {new_lang}")
                            self.active_language = new_lang
                            self.case_container["language"] = self.active_language
                            self.audio.speak_dynamic(spoken_resp, lang=self.active_language, block=True)
                            self.case_container["conversation"].append({"role": "kiosk_action", "text": spoken_resp})
                            continue

                        # ACTION 2: Cancel and Exit
                        elif action == "CANCEL_AND_EXIT":
                            print("[!] AGENT COMMAND: Patient requested early cancellation.")
                            self.audio.speak_dynamic(spoken_resp, lang=self.active_language, block=True)
                            self.case_container["exit_reason"] = "USER_CANCELLED"
                            interview_active = False
                            break

                        # ACTION 3: Call Human Nurse
                        elif action == "CALL_HUMAN_NURSE":
                            print("[!] AGENT COMMAND: Summoning human triage staff.")
                            self.audio.speak_dynamic(spoken_resp, lang=self.active_language, block=True)
                            self.case_container["exit_reason"] = "HUMAN_NURSE_SUMMONED"
                            interview_active = False
                            break

                        # ACTION 4: Emergency Abort (Acute Decompensation)
                        elif action == "EMERGENCY_ABORT":
                            print("[!] AGENT COMMAND: High-Acuity Emergency Abort triggered!")
                            self.audio.speak_dynamic(spoken_resp, lang=self.active_language, block=True)
                            self.case_container["exit_reason"] = ExitReason.EMERGENCY_INTERRUPT.value
                            interview_active = False
                            break

                        # ACTION 5: 3-Strike Cognitive Failure (AMS)
                        elif action == "TRIGGER_AMS":
                            print("[!] 3-Strike Rule Tripped: Altered Mental Status (AMS).")
                            self.audio.play_template("ams_alert", lang=self.active_language, block=True)
                            self.case_container["exit_reason"] = ExitReason.AMS_3_STRIKES.value
                            interview_active = False
                            break

                        # ACTION 6: Normal Dialogue Turn (OPQRST Probing / Bedside Pivot)
                        else:
                            self.audio.speak_dynamic(spoken_resp, lang=self.active_language, block=True)
                            self.case_container["conversation"].append({"role": "kiosk_agent", "text": spoken_resp})
                            continue

                    self.current_state = KioskState.FINALIZING_CASE

                # -------------------------------------------------------------
                # 5. State: FINALIZING_CASE (SOAP & ESI Synthesis)
                # -------------------------------------------------------------
                elif self.current_state == KioskState.FINALIZING_CASE:
                    raw_exit = self.case_container.get("exit_reason", ExitReason.NORMAL_COMPLETE.value)
                    exit_str = raw_exit.value if hasattr(raw_exit, "value") else str(raw_exit)

                    print(f"[*] Packaging case [{self.case_container['case_id']}] (Exit: {exit_str})...")
                    self._update_cumulative_visuals()

                    verdict = self.reasoning.evaluate_final_case(
                        conversation_history=self.case_container["conversation"],
                        visual_signs=self.case_container["cumulative_visual_signs"],
                        skin_status=self.case_container["facial_skin_status"],
                        exit_reason=raw_exit,
                        language=self.active_language
                    )
                    self.case_container["triage_verdict"] = verdict
                    print(f"[VERDICT]: ESI Level {verdict.get('esi_level')} ({verdict.get('urgency')})")
                    self.current_state = KioskState.DISPATCH_AND_RESET

                # -------------------------------------------------------------
                # 6. State: DISPATCH_AND_RESET
                # -------------------------------------------------------------
                elif self.current_state == KioskState.DISPATCH_AND_RESET:
                    raw_exit = self.case_container.get("exit_reason", ExitReason.NORMAL_COMPLETE.value)
                    exit_str = raw_exit.value if hasattr(raw_exit, "value") else str(raw_exit)

                    if exit_str == ExitReason.EMERGENCY_INTERRUPT.value:
                        self.audio.play_template("emergency_alert", lang=self.active_language, block=True)
                    elif exit_str == ExitReason.AMS_3_STRIKES.value:
                        self.audio.play_template("ams_alert", lang=self.active_language, block=True)
                    elif exit_str not in ["USER_CANCELLED", "HUMAN_NURSE_SUMMONED"]:
                        self.audio.play_template("wrapup", lang=self.active_language, block=True)

                    # Enqueue to background Conveyor Belt (<1ms)
                    self.conveyor_belt.enqueue_case(self.case_container)
                    print("[*] Case finalized. Kiosk returning to PREPARE_STANDBY.\n" + "="*55)
                    self.current_state = KioskState.PREPARE_STANDBY

                time.sleep(0.05)

            except Exception as e:
                print(f"[!] FSM Encounter Exception: {e}")
                time.sleep(1.0)

    def run(self):
        """Main-Thread GUI Capture Loop (macOS Cocoa Compliant)."""
        fsm_thread = threading.Thread(target=self._fsm_worker, daemon=True)
        fsm_thread.start()

        cap = cv2.VideoCapture(SystemConfig.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, SystemConfig.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SystemConfig.FRAME_HEIGHT)

        print("[*] Camera capture active on Main Thread.")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                telemetry = self.vision.process_frame(frame)
                with self.telemetry_lock:
                    self.latest_telemetry = telemetry

                if SystemConfig.SHOW_DUAL_WINDOWS:
                    cv2.imshow("Kiosk - Clean Patient View", telemetry["clean_view"])
                    cv2.imshow("Kiosk - Diagnostic Skeleton View", telemetry["skeleton_view"])
                else:
                    cv2.imshow("Smart Triage Kiosk", telemetry["clean_view"])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

        except KeyboardInterrupt:
            print("\n[!] Powering down systems gracefully...")
        finally:
            self.running = False
            self.conveyor_belt.running = False
            cap.release()
            cv2.destroyAllWindows()
            self.vision.release()
            self.audio.release()
            print("[+] Kiosk successfully and safely offline.")


if __name__ == "__main__":
    kiosk = SmartReceptionTriageKiosk()
    kiosk.run()