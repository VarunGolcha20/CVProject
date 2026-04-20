import cv2
import json
import asyncio
import websockets
import time
import sys
import os

# Add the parent directory (proj) to sys.path so we can import pipeline_mnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from pipeline_mnet import PoseExtractor, FeatureEngineer
    from classifiers.rule_based import RuleBasedClassifier
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running from the 'game' folder and 'pipeline_mnet.py' is in the parent folder.")
    sys.exit(1)

# DYNAMIC IP RESOLUTION
TARGET_IP = sys.argv[1] if len(sys.argv) > 1 else "localhost"
SERVER_URI = f"ws://{TARGET_IP}:8760"
PLAYER_ID = 1 

class VisionController:
    def __init__(self):
        self.extractor = PoseExtractor()
        self.engineer = FeatureEngineer(history=35)
        self.clf = RuleBasedClassifier()
        self.cap = cv2.VideoCapture(0)
        self.last_state = "STOP"

    async def run(self):
        print(f"[VISION] Connecting to {SERVER_URI}...")
        try:
            async with websockets.connect(SERVER_URI) as websocket:
                await websocket.send(json.dumps({"type": "register_controller", "pid": PLAYER_ID}))
                print(f"[VISION] SUCCESS: Controlling Player {PLAYER_ID}")

                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret: break
                    
                    frame = cv2.flip(frame, 1)
                    t = time.perf_counter()
                    pose_frame = self.extractor.extract(frame, t)

                    if pose_frame:
                        fv = self.engineer.update(pose_frame)
                        gesture = self.clf(fv) 

                        # Command Mapping
                        cmd = "STOP"
                        if gesture == "arms_spread": cmd = "MOVE"
                        elif gesture == "jump": cmd = "SHOOT"
                        elif gesture == "wave": cmd = "PASS"

                        if cmd != self.last_state or cmd in ["SHOOT", "PASS"]:
                            await websocket.send(json.dumps({"type": "input", "player": PLAYER_ID, "move": cmd}))
                            self.last_state = cmd

                    cv2.putText(frame, f"CTRL: P{PLAYER_ID} | CMD: {self.last_state}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("GestureBall Vision", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                    await asyncio.sleep(0.01)
        except Exception as e:
            print(f"[VISION] Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(VisionController().run())