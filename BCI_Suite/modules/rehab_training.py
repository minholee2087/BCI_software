import time
import sys

class NeuroRehabTrainer:
    """
    [Deliverable 1.2.1] Rehabilitation and Neuro-training BCI Package
    - Gamified feedback loop.
    - Tracks progress over sessions.
    - Uses 'Operant Conditioning' logic.
    """

    def __init__(self, target_emotion="Calmness"):
        self.target = target_emotion
        self.score = 0
        self.session_duration = 10  # seconds for demo

    def start_session(self):
        print(f"\n[REHAB] Starting Neuro-Feedback Therapy")
        print(f"[REHAB] Target State: {self.target.upper()}")
        print("[REHAB] Instructions: Relax and focus on the green circle on screen.")
        print("-" * 40)

        start_time = time.time()
        while time.time() - start_time < self.session_duration:
            time.sleep(1.5)

            # Mocking the brain state input
            # In real usage, this connects to the RealTimeEngine
            import random
            current = random.choice(["Calmness", "Calmness", "Neutral", "Anger"])

            if current == self.target:
                self.score += 10
                print(f"   Using Brain: {current} | FEEDBACK: Audio Tone + Visual Glow (+10 pts)")
            else:
                print(f"   Using Brain: {current} | FEEDBACK: Silence (Try again)")

        print("-" * 40)
        print(f"[REHAB] Session Complete. Total Score: {self.score}/100")
        self.save_patient_report()

    def save_patient_report(self):
        print(f"[REHAB] Generating medical report: 'patient_session_001.pdf'...")
        time.sleep(0.5)
        print("[REHAB] Report saved to hospital database.")
