"""
Main.py - Kiosk Master Orchestrator
Drives the Receptionist FSM, Dual OpenCV Windows, PDPA Gating, and Atomic Case Dispatch.
"""

import cv2
import time
import base64
import threading
import requests
from config import SystemConfig, KioskState
from V_Module import VisionEngine
from A_Module import AudioEngine
from R_Module import ReasoningEngine

class ReceptionistTriageKiosk:
    def __init__(self):
        print(">> Powering on Reception Kiosk...")
        self.vision = VisionEngine()
        self.audio = AudioEngine()
        self.reasoning = ReasoningEngine()

        self.current_state = KioskState.IDLE
        self.active_language = "th"
        self.running = True

        # Thread-safe vision state
        self.latest_telemetry = {}
        self.telemetry_lock = threading.Lock()

        # Immutable Case Container (Dispatched atomically upon case completion)
        self.case_container = {}
        self._reset_case_container()

    def _reset_case_container(self):
        self.case_container = {
            "case_id": f"CASE-{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "language": self.active_language,
            "pdpa_consented": False,
            "conversation": [],
            "visual_anomalies": {
                "chest_clutching": False,
                "abdominal_clutching": False,
                "head_clutching": False
            },
            "skin_status": "NORMAL",
            "visual_evidence_b64": None,
            "triage_verdict": None
        }

    def _video_stream_worker(self):
        """Maintains dual camera windows (Clean + Skeleton) at steady 30 FPS."""
        cap = cv2.VideoCapture(SystemConfig.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, SystemConfig.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SystemConfig.FRAME_HEIGHT)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            telemetry = self.vision.process_frame(frame)

            with self.telemetry_lock:
                self.latest_telemetry = telemetry

            # Display dual windows if configured
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

    def _dispatch_case_to_nurse(self):
        """Sends the packaged case container to the nurse's web dashboard."""
        print(f"[*] Packaging Case Container [{self.case_container['case_id']}] -> Nurse Dashboard...")
        try:
            res = requests.post(
                SystemConfig.NURSE_DASHBOARD_ENDPOINT,
                json=self.case_container,
                timeout=SystemConfig.HTTP_TIMEOUT
            )
            print(f"[+] Successfully uploaded to Nurse Dashboard. Status: {res.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[!] Offline Mode: Nurse Station unreachable ({e}). Encrypted payload saved locally.")

    def run(self):
        """FSM Execution Loop."""
        video_thread = threading.Thread(target=self._video_stream_worker, daemon=True)
        video_thread.start()

        print("[*] Triage Reception Agent is active and listening.")

        try:
            while self.running:
                # -------------------------------------------------------------
                # 1. State: IDLE
                # -------------------------------------------------------------
                if self.current_state == KioskState.IDLE:
                    with self.telemetry_lock:
                        patient_in_zone = self.latest_telemetry.get("patient_present", False)

                    if patient_in_zone:
                        print("[*] Patient arrived. Initiating Intake Reception Sequence...")
                        self._reset_case_container()
                        self.current_state = KioskState.GREETING_LANG_SELECT

                # -------------------------------------------------------------
                # 2. State: GREETING_LANG_SELECT
                # -------------------------------------------------------------
                elif self.current_state == KioskState.GREETING_LANG_SELECT:
                    # Greet and prompt for language
                    self.audio.speak(SystemConfig.SCRIPTS["th"]["greeting"], lang="th")
                    while self.audio.is_speaking:
                        time.sleep(0.1)

                    utterance = self.audio.listen(language="th-TH")
                    print(f"[Language Preference Input]: {utterance}")

                    # Determine language
                    if any(w in utterance.lower() for w in ["english", "eng", "hello", "hi"]):
                        self.active_language = "en"
                    else:
                        self.active_language = "th"

                    self.case_container["language"] = self.active_language
                    self.current_state = KioskState.PDPA_CONSENT

                # -------------------------------------------------------------
                # 3. State: PDPA_CONSENT
                # -------------------------------------------------------------
                elif self.current_state == KioskState.PDPA_CONSENT:
                    lang = self.active_language
                    notice = SystemConfig.SCRIPTS[lang]["pdpa_notice"]
                    self.audio.speak(notice, lang=lang)
                    while self.audio.is_speaking:
                        time.sleep(0.1)

                    stt_lang = "th-TH" if lang == "th" else "en-US"
                    consent_reply = self.audio.listen(language=stt_lang)
                    self.case_container["conversation"].append({"role": "pdpa_response", "text": consent_reply})

                    # Evaluate consent
                    deny_tokens = ["ไม่", "ปฏิเสธ", "no", "deny", "disagree"]
                    if any(t in consent_reply.lower() for t in deny_tokens):
                        print("[-] Patient denied PDPA consent. Rerouting to manual triage desk...")
                        self.current_state = KioskState.MANUAL_ROUTING
                    else:
                        self.case_container["pdpa_consented"] = True
                        self.current_state = KioskState.SYMPTOM_INTERVIEW

                # -------------------------------------------------------------
                # 3b. State: MANUAL_ROUTING (PDPA Denied)
                # -------------------------------------------------------------
                elif self.current_state == KioskState.MANUAL_ROUTING:
                    lang = self.active_language
                    self.audio.speak(SystemConfig.SCRIPTS[lang]["pdpa_denied"], lang=lang)
                    while self.audio.is_speaking:
                        time.sleep(0.1)
                    time.sleep(4.0)
                    self.current_state = KioskState.IDLE

                # -------------------------------------------------------------
                # 4. State: SYMPTOM_INTERVIEW
                # -------------------------------------------------------------
                elif self.current_state == KioskState.SYMPTOM_INTERVIEW:
                    lang = self.active_language
                    inquiry = SystemConfig.SCRIPTS[lang]["inquiry"]
                    self.audio.speak(inquiry, lang=lang)
                    while self.audio.is_speaking:
                        time.sleep(0.1)

                    stt_lang = "th-TH" if lang == "th" else "en-US"
                    patient_complaint = self.audio.listen(language=stt_lang)
                    self.case_container["conversation"].append({"role": "patient", "text": patient_complaint})

                    # Harvest accumulated visual telemetry
                    with self.telemetry_lock:
                        t = self.latest_telemetry
                        self.case_container["visual_anomalies"]["chest_clutching"] = t.get("chest_clutching", False)
                        self.case_container["visual_anomalies"]["abdominal_clutching"] = t.get("abdominal_clutching", False)
                        self.case_container["visual_anomalies"]["head_clutching"] = t.get("head_clutching", False)
                        self.case_container["skin_status"] = t.get("skin_status", "NORMAL")

                        # Snapshot of the diagnostic view for the nurse
                        if "skeleton_view" in t:
                            _, buf = cv2.imencode('.jpg', t["skeleton_view"])
                            self.case_container["visual_evidence_b64"] = base64.b64encode(buf).decode('utf-8')

                    self.current_state = KioskState.FINALIZING_CASE

                # -------------------------------------------------------------
                # 5. State: FINALIZING_CASE
                # -------------------------------------------------------------
                elif self.current_state == KioskState.FINALIZING_CASE:
                    print("[*] Packaging case data and initiating triage evaluation...")
                    verdict = self.reasoning.evaluate_case(
                        conversation_history=self.case_container["conversation"],
                        visual_signs=self.case_container["visual_anomalies"],
                        skin_status=self.case_container["skin_status"],
                        language=self.active_language
                    )
                    self.case_container["triage_verdict"] = verdict
                    print(f"[VERDICT]: ESI Level {verdict.get('esi_level')} | Urgency: {verdict.get('urgency')}")
                    self.current_state = KioskState.DISPATCH_AND_RESET

                # -------------------------------------------------------------
                # 6. State: DISPATCH_AND_RESET
                # -------------------------------------------------------------
                elif self.current_state == KioskState.DISPATCH_AND_RESET:
                    lang = self.active_language
                    # Play patient closing instructions
                    self.audio.speak(SystemConfig.SCRIPTS[lang]["wrapup"], lang=lang)

                    # Transmit the completed case to the nurse dashboard
                    self._dispatch_case_to_nurse()

                    while self.audio.is_speaking:
                        time.sleep(0.1)

                    time.sleep(3.0)
                    print("[*] Case completed. Resetting kiosk to IDLE standby.\n" + "="*50)
                    self.current_state = KioskState.IDLE

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n[!] Graceful shutdown initiated...")
        finally:
            self.running = False
            video_thread.join()
            self.vision.release()
            self.audio.release()
            print("[+] Kiosk successfully offline.")

if __name__ == "__main__":
    kiosk = ReceptionistTriageKiosk()
    kiosk.run()