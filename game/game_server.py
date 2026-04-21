import asyncio, json, math, time, websockets
import cv2, base64
import numpy as np
import os
import sys
import torch
import mediapipe as mp
from collections import deque, defaultdict

# Ensure parent directory is in path for classifiers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your new ST-GCN architecture and joint list
try:
    from stgcn import GestureSTGCN, CORE_JOINTS, GESTURE_CLASSES
except ImportError:
    print("[ERROR] Could not find stgcn.py. Ensure it's in the root or classifiers folder.")
    GestureSTGCN = None

W, H = 900, 560 
GOAL_H, TICK_RATE = 160, 60
GY1, GY2 = (H - GOAL_H) // 2, (H + GOAL_H) // 2
ROTATION_SPEED = 0.3 
MOVE_GRACE = 0.5 
GAME_DURATION = 300.0
DIRECTIONS = [4.71, 5.49, 0, 0.785, 1.57, 2.35, 3.14, 3.92]

class GameServer:
    def __init__(self):
        # --- 1. Core Game State ---
        self.players = []
        self.ball = {"x": 450, "y": 280, "vx": 0, "vy": 0, "holder": None}
        self.phase = "waiting" 
        self.score = [0, 0]
        self.time_left = GAME_DURATION
        self.clients = [] 
        self.frames = {} 
        self.processing_pids = set() 
        self.move_counters = defaultdict(int)
        self.pose_lost_counter = defaultdict(int) 
        self.MAX_LOST_FRAMES = 5 # Grace period of 3 frames before forcing IDLE
        self.cmd_latches = defaultdict(lambda: {"cmd": "STOP", "frames": 0})
        self.LATCH_DURATION = 2 # Keep SHOOT/PASS active for 10 frames (~0.16s)
        # --- 2. ST-GCN Buffering & Per-Player Filtering ---
        self.buffers = defaultdict(lambda: deque(maxlen=15))
        self.ema_skeletons = defaultdict(lambda: None) 
        self.ema_alpha = 0.4 # Jitter reduction (0.4 is a good balance for Arch/Intel Iris)
        
        # --- 3. MediaPipe Setup ---
        # In your __init__ MediaPipe setup:
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            # This can help on some Linux builds:
            min_tracking_confidence=0.5
        )

        # --- 4. ST-GCN Model Loading ---
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stgcn_best.pt"))
        try:
            self.device = torch.device("cpu")
            self.stgcn_clf = GestureSTGCN(num_classes=4)
            self.stgcn_clf.load_state_dict(torch.load(model_path, map_location=self.device))
            self.stgcn_clf.eval()
            print(f"[LOG] ST-GCN Loaded Successfully: {model_path}")
        except Exception as e:
            print(f"[ERROR] ST-GCN Load Failed: {e}")
            self.stgcn_clf = None
        
        self.setup_players(4)
        
    def setup_players(self, count):
        self.players = []
        half = count // 2
        for i in range(count):
            team = 0 if i < half else 1
            x = 200 if team == 0 else 700
            y = 100 + (i % half) * (360 // max(1, half-1)) if half > 1 else 280
            self.players.append({
                "pid": i + 1, "team": team, "x": x, "y": y,
                "vx": 0, "vy": 0, "dir_idx": 2 if team == 0 else 6, 
                "last_scan": 0, "is_moving": False, "last_move_time": time.time()
            })

    # def process_vision_frame(self, pid, b64_data):
    #     if not self.stgcn_clf: return pid, "STOP", b64_data
        
    #     try:
    #         header, encoded = b64_data.split(",", 1)
    #         nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    #         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #         frame = cv2.flip(frame, 1)

    #         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         results = self.mp_pose.process(img_rgb)
            
    #         # --- TELEMETRY PRINTS ---
    #         # This will show up in your Arch terminal
    #         if not results.pose_landmarks:
    #             print(f"[DEBUG P{pid}] MediaPipe found ZERO landmarks.")
    #         else:
    #             lms = results.pose_landmarks.landmark
    #             # Check the core joints we actually use
    #             lower_vis = [lms[i].visibility for i in [23, 24, 25, 26]]
    #             avg_v = sum(lower_vis) / 4
    #             torso = abs(lms[11].y - lms[23].y)
    #             print(f"[DEBUG P{pid}] Vis: {avg_v:.3f} | Torso: {torso:.3f}")

    #         # --- BYPASS ALL GATES ---
    #         # We are forcing has_pose to True if ANY landmarks exist
    #         current_frame_has_pose = True if results.pose_landmarks else False
    #         skeleton_step = np.zeros((3, 12)) 

    #         if current_frame_has_pose:
    #             lms = results.pose_landmarks.landmark
    #             raw_step = np.zeros((3, 12))
    #             for v, joint_idx in enumerate(CORE_JOINTS):
    #                 raw_step[0, v] = lms[joint_idx].x
    #                 raw_step[1, v] = lms[joint_idx].y
    #                 raw_step[2, v] = lms[joint_idx].visibility
                    
    #                 # Draw dots NO MATTER WHAT visibility they have
    #                 cx, cy = int(lms[joint_idx].x * frame.shape[1]), int(lms[joint_idx].y * frame.shape[0])
    #                 cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1) # Yellow dots

    #             # EMA & Normalization
    #             if self.ema_skeletons[pid] is None:
    #                 self.ema_skeletons[pid] = raw_step
    #             else:
    #                 self.ema_skeletons[pid] = (0.5 * raw_step) + (0.5 * self.ema_skeletons[pid])
                
    #             skeleton_step = self.ema_skeletons[pid].copy()
    #             m_x = (skeleton_step[0, 4] + skeleton_step[0, 5]) / 2
    #             m_y = (skeleton_step[1, 4] + skeleton_step[1, 5]) / 2
    #             skeleton_step[0, :] -= m_x
    #             skeleton_step[1, :] -= m_y

    #         # --- Update Buffer ---
    #         if current_frame_has_pose:
    #             self.buffers[pid].append(skeleton_step)
            
    #         # Default logic for the game
    #         cmd, label = "STOP", "IDLE"
    #         if len(self.buffers[pid]) == 15:
    #             # (Keep your existing ST-GCN inference here...)
    #             pass

    #         # Final feedback string
    #         status = "LOCKED" if current_frame_has_pose else "SEARCHING..."
    #         cv2.putText(frame, f"P{pid}: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
    #         _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
    #         out_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
    #         return pid, cmd, out_b64
    #     except Exception as e:
    #         print(f"[CRITICAL] {e}")
    #         return pid, "STOP", b64_data
        
    # def process_vision_frame(self, pid, b64_data):
    #     try:
    #         # 1. Decoding & Preprocessing
    #         header, encoded = b64_data.split(",", 1)
    #         nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    #         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #         frame = cv2.flip(frame, 1)

    #         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         results = self.mp_pose.process(img_rgb)
            
    #         current_frame_has_pose = False
    #         skeleton_step = np.zeros((3, 12)) 
            
    #         # --- SKELETON EXTRACTION & NORMALIZATION ---
    #         if results.pose_landmarks:
    #             lms = results.pose_landmarks.landmark
    #             avg_vis = sum([lms[i].visibility for i in [23, 24, 25, 26]]) / 4
                
    #             if avg_vis > 0.35:
    #                 current_frame_has_pose = True
    #                 self.pose_lost_counter[pid] = 0
                    
    #                 raw_step = np.zeros((3, 12))
    #                 for v, joint_idx in enumerate(CORE_JOINTS):
    #                     raw_step[0, v] = lms[joint_idx].x
    #                     raw_step[1, v] = lms[joint_idx].y
    #                     raw_step[2, v] = lms[joint_idx].visibility
                        
    #                     # Draw green visual tracking dots
    #                     cx, cy = int(lms[joint_idx].x * frame.shape[1]), int(lms[joint_idx].y * frame.shape[0])
    #                     cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    #                 # EMA Smoothing (Weight set to 0.5 for fast response)
    #                 if self.ema_skeletons[pid] is None:
    #                     self.ema_skeletons[pid] = raw_step
    #                 else:
    #                     self.ema_skeletons[pid] = (0.5 * raw_step) + (0.5 * self.ema_skeletons[pid])
                    
    #                 skeleton_step = self.ema_skeletons[pid].copy()
                    
    #                 # Mid-Hip Centering (Anchors the skeleton so movement is relative)
    #                 m_x = (skeleton_step[0, 4] + skeleton_step[0, 5]) / 2
    #                 m_y = (skeleton_step[1, 4] + skeleton_step[1, 5]) / 2
    #                 skeleton_step[0, :] -= m_x
    #                 skeleton_step[1, :] -= m_y

    #         # 2. Hysteresis (Grace Period)
    #         if not current_frame_has_pose:
    #             self.pose_lost_counter[pid] += 1
            
    #         pose_is_stable = self.pose_lost_counter[pid] < self.MAX_LOST_FRAMES

    #         if pose_is_stable:
    #             if not current_frame_has_pose and self.ema_skeletons[pid] is not None:
    #                 skeleton_step = self.ema_skeletons[pid].copy()
    #                 m_x = (skeleton_step[0, 4] + skeleton_step[0, 5]) / 2
    #                 m_y = (skeleton_step[1, 4] + skeleton_step[1, 5]) / 2
    #                 skeleton_step[0, :] -= m_x
    #                 skeleton_step[1, :] -= m_y
    #             self.buffers[pid].append(skeleton_step)
    #         else:
    #             self.buffers[pid].clear()

    #         # --- 3. VECTOR-BASED RULE ENGINE ---
    #         if self.cmd_latches[pid]["frames"] > 0:
    #             self.cmd_latches[pid]["frames"] -= 1
    #             cmd = self.cmd_latches[pid]["cmd"]
    #             label = f"{cmd} (LOCKED)"
    #         else:
    #             cmd, label = "STOP", "IDLE"

    #             if pose_is_stable and len(self.buffers[pid]) == 15:
    #                 clip = np.array(list(self.buffers[pid]))
                    
    #                 # Total motion magnitude to filter out standing still
    #                 mag = np.sum(np.abs(np.diff(clip[:, :2, :], axis=0)))
                    
    #                 # A. Extract Vectors (Comparing current frame to 6 frames ago)
    #                 # Y-axis decreases as it goes UP the screen.
    #                 l_ank_vec = clip[-1, :2, 8] - clip[-6, :2, 8]
    #                 r_ank_vec = clip[-1, :2, 9] - clip[-6, :2, 9]
                    
    #                 l_ank_speed = np.linalg.norm(l_ank_vec)
    #                 r_ank_speed = np.linalg.norm(r_ank_vec)
                    
    #                 # B. Isolate Vertical Velocity (Negative = moving UP)
    #                 l_ank_y_vel = l_ank_vec[1]
    #                 r_ank_y_vel = r_ank_vec[1]
                    
    #                 # C. Current Heights for Classification
    #                 l_y, r_y = clip[-1, 1, 8], clip[-1, 1, 9]
    #                 avg_knee_y = (clip[-1, 1, 6] + clip[-1, 1, 7]) / 2

    #                 # --- TRIGGER LOGIC ---
                    
    #                 # RULE 1: KICK DETECTION
    #                 # A legitimate kick has high speed AND an upward trajectory.
    #                 # This prevents lateral lunges (which have speed but no upward Y) from registering as kicks.
    #                 l_kicking = l_ank_speed > 0.08 and l_ank_y_vel < -0.02
    #                 r_kicking = r_ank_speed > 0.08 and r_ank_y_vel < -0.02

    #                 if l_kicking or r_kicking:
    #                     # Determine which foot is kicking
    #                     kicking_foot_y = l_y if l_kicking else r_y
    #                     kick_speed = l_ank_speed if l_kicking else r_ank_speed
                        
    #                     # RULE 2: SHOOT vs PASS
    #                     # If the foot swings higher than the knees, or is extremely fast -> SHOOT
    #                     if kicking_foot_y < avg_knee_y or kick_speed > 0.16:
    #                         cmd = "SHOOT"
    #                     else:
    #                         # If the foot stays low and has moderate speed -> PASS
    #                         cmd = "PASS"
                            
    #                     # Latch the action so the game engine registers it
    #                     self.cmd_latches[pid] = {"cmd": cmd, "frames": 10}
    #                     label = f"{cmd} TRIGGERED"
    #                     self.buffers[pid].clear()
                        
    #                 # RULE 3: JOGGING / MOVING
    #                 # If not kicking, but overall motion is above the deadzone
    #                 elif mag > 0.045: 
    #                     cmd, label = "MOVE", "MOVE (Jogging)"
                        
    #                 # RULE 4: IDLE
    #                 else:
    #                     cmd, label = "STOP", "IDLE (Neutral)"

    #         if not pose_is_stable:
    #             label, cmd = "IDLE (NO POSE)", "STOP"

    #         # Draw labels
    #         cv2.putText(frame, f"P1: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
    #         _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
    #         out_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
    #         return pid, cmd, out_b64
            
    #     except Exception as e:
    #         print(f"[ERROR] {e}")
    #         return pid, "STOP", b64_data


    def process_vision_frame(self, pid, b64_data):
        try:
            # 1. Decoding & Preprocessing
            header, encoded = b64_data.split(",", 1)
            nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame = cv2.flip(frame, 1)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_pose.process(img_rgb)
            
            current_frame_has_pose = False
            skeleton_step = np.zeros((3, 12)) 
            
            # --- SKELETON EXTRACTION & NORMALIZATION ---
            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                avg_vis = sum([lms[i].visibility for i in [23, 24, 25, 26]]) / 4
                
                if avg_vis > 0.35:
                    current_frame_has_pose = True
                    self.pose_lost_counter[pid] = 0
                    
                    raw_step = np.zeros((3, 12))
                    for v, joint_idx in enumerate(CORE_JOINTS):
                        raw_step[0, v] = lms[joint_idx].x
                        raw_step[1, v] = lms[joint_idx].y
                        raw_step[2, v] = lms[joint_idx].visibility
                        
                        cx, cy = int(lms[joint_idx].x * frame.shape[1]), int(lms[joint_idx].y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                    if self.ema_skeletons[pid] is None:
                        self.ema_skeletons[pid] = raw_step
                    else:
                        self.ema_skeletons[pid] = (0.5 * raw_step) + (0.5 * self.ema_skeletons[pid])
                    
                    skeleton_step = self.ema_skeletons[pid].copy()
                    
                    m_x = (skeleton_step[0, 4] + skeleton_step[0, 5]) / 2
                    m_y = (skeleton_step[1, 4] + skeleton_step[1, 5]) / 2
                    skeleton_step[0, :] -= m_x
                    skeleton_step[1, :] -= m_y

            # 2. Hysteresis (Grace Period)
            if not current_frame_has_pose:
                self.pose_lost_counter[pid] += 1
            
            pose_is_stable = self.pose_lost_counter[pid] < self.MAX_LOST_FRAMES

            if pose_is_stable:
                if not current_frame_has_pose and self.ema_skeletons[pid] is not None:
                    skeleton_step = self.ema_skeletons[pid].copy()
                    m_x = (skeleton_step[0, 4] + skeleton_step[0, 5]) / 2
                    m_y = (skeleton_step[1, 4] + skeleton_step[1, 5]) / 2
                    skeleton_step[0, :] -= m_x
                    skeleton_step[1, :] -= m_y
                self.buffers[pid].append(skeleton_step)
            else:
                self.buffers[pid].clear()

            # --- 3. VECTOR-BASED RULE ENGINE ---
            if self.cmd_latches[pid]["frames"] > 0:
                self.cmd_latches[pid]["frames"] -= 1
                cmd = self.cmd_latches[pid]["cmd"]
                label = f"{cmd} (LOCKED)"
            else:
                cmd, label = "STOP", "IDLE"

                if pose_is_stable and len(self.buffers[pid]) == 15:
                    clip = np.array(list(self.buffers[pid]))
                    
                    # A. KICK PHYSICS (Speed + Upward Trajectory)
                    l_ank_vec = clip[-1, :2, 8] - clip[-6, :2, 8]
                    r_ank_vec = clip[-1, :2, 9] - clip[-6, :2, 9]
                    
                    l_ank_speed = np.linalg.norm(l_ank_vec)
                    r_ank_speed = np.linalg.norm(r_ank_vec)
                    
                    l_ank_y_vel = l_ank_vec[1]
                    r_ank_y_vel = r_ank_vec[1]
                    
                    l_y, r_y = clip[-1, 1, 8], clip[-1, 1, 9]
                    avg_knee_y = (clip[-1, 1, 6] + clip[-1, 1, 7]) / 2

                    # B. JOGGING PHYSICS (Vertical Variance)
                    # We look at the standard deviation of the Y-axis for the ankles over the 15 frames.
                    # Breathing/swaying creates a variance < 0.008. Jogging creates a variance > 0.02.
                    l_ank_std = np.std(clip[:, 1, 8])
                    r_ank_std = np.std(clip[:, 1, 9])
                    is_jogging = l_ank_std > 0.020 or r_ank_std > 0.020

                    # --- TRIGGER LOGIC ---
                    
                    # RULE 1: KICK DETECTION
                    l_kicking = l_ank_speed > 0.08 and l_ank_y_vel < -0.02
                    r_kicking = r_ank_speed > 0.08 and r_ank_y_vel < -0.02

                    if l_kicking or r_kicking:
                        kicking_foot_y = l_y if l_kicking else r_y
                        kick_speed = l_ank_speed if l_kicking else r_ank_speed
                        
                        # RULE 2: SHOOT vs PASS
                        if kicking_foot_y < avg_knee_y or kick_speed > 0.16:
                            cmd = "SHOOT"
                        else:
                            cmd = "PASS"
                            
                        self.cmd_latches[pid] = {"cmd": cmd, "frames": 10}
                        label = f"{cmd} TRIGGERED"
                        self.buffers[pid].clear()
                        
                    # RULE 3: JOGGING / MOVING
                    # Only fires if a foot is actively lifting up and down
                    elif is_jogging: 
                        cmd, label = "MOVE", "MOVE (Jogging)"
                        
                    # RULE 4: IDLE
                    # If neither a kick nor a jog is detected, ignore all other noise
                    else:
                        cmd, label = "STOP", "IDLE (Neutral)"

            if not pose_is_stable:
                label, cmd = "IDLE (NO POSE)", "STOP"

            # Draw labels
            cv2.putText(frame, f"P1: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
            out_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
            return pid, cmd, out_b64
            
        except Exception as e:
            print(f"[ERROR] {e}")
            return pid, "STOP", b64_data

    async def async_vision_task(self, pid, frame_data):
        self.processing_pids.add(pid)
        loop = asyncio.get_running_loop()
        try:
            _, cmd, out_b64 = await loop.run_in_executor(None, self.process_vision_frame, pid, frame_data)
            self.frames[str(pid)] = out_b64
            self.handle_game_input(pid, cmd)
        finally:
            self.processing_pids.remove(pid)

    def handle_game_input(self, p_id, cmd):
        idx = int(p_id) - 1
        if not (0 <= idx < len(self.players)): return
        
        p = self.players[idx]
        
        if cmd == "MOVE":
            p['is_moving'] = True
            
        elif cmd == "STOP":
            if p['is_moving']: 
                p['last_move_time'] = time.time()
            p['is_moving'] = False
            
        elif cmd == "PASS":
            # STRICT CHECK: Only execute if this specific player holds the ball
            if self.ball['holder'] == p['pid']:
                self.kick(p, 14) # Base Pass Speed
                
        elif cmd == "SHOOT":
            # STRICT CHECK: Only execute if this specific player holds the ball
            if self.ball['holder'] == p['pid']:
                self.kick(p, 28) # Exactly 2x the Pass Speed

    def kick(self, p, power):
        # Calculate velocity vectors based on the player's current rotation angle
        angle = DIRECTIONS[p['dir_idx']]
        
        # Apply the power strictly in the direction of the player's arrow
        self.ball['vx'] = math.cos(angle) * power
        self.ball['vy'] = math.sin(angle) * power
        
        # Detach the ball from the player so it can move freely
        self.ball['holder'] = None

    async def handler(self, ws):
        self.clients.append(ws)
        try:
            async for msg in ws:
                data = json.loads(msg)
                m_type = data.get('type')
                if m_type == 'ready':
                    self.setup_players(int(data.get('count', 4)))
                    self.score, self.time_left, self.phase = [0, 0], GAME_DURATION, "playing"
                    self.reset_positions()
                    self.frames = {}
                    self.buffers.clear()
                    self.ema_skeletons.clear()
                elif m_type == 'video_frame':
                    pid = int(data['pid'])
                    if pid not in self.processing_pids:
                        asyncio.create_task(self.async_vision_task(pid, data['frame']))
                elif m_type == 'ping':
                    await ws.send(json.dumps({"type": "pong", "time": data['time']}))
        finally:
            if ws in self.clients:
                self.clients.remove(ws)

    def update_physics(self):
        if self.phase != "playing": return
        self.time_left -= 1/TICK_RATE
        now = time.time()
        for p in self.players:
            if not p['is_moving'] and (now - p['last_move_time'] >= MOVE_GRACE):
                if now - p['last_scan'] >= ROTATION_SPEED:
                    p['dir_idx'] = (p['dir_idx'] + 1) % 8
                    p['last_scan'] = now
            if p['is_moving']:
                angle = DIRECTIONS[p['dir_idx']]
                p['vx'], p['vy'] = math.cos(angle)*8.2, math.sin(angle)*8.2
            else:
                p['vx'] *= 0.82; p['vy'] *= 0.82
            
            p['x'] += p['vx']; p['y'] += p['vy']
            p['x'], p['y'] = max(25, min(875, p['x'])), max(25, min(535, p['y']))

        if not self.ball['holder']:
            self.ball['x'] += self.ball['vx']; self.ball['y'] += self.ball['vy']
            self.ball['vx'] *= 0.985; self.ball['vy'] *= 0.985
            if self.ball['x'] < 30 or self.ball['x'] > 870:
                if GY1 < self.ball['y'] < GY2:
                    self.score[1 if self.ball['x'] < 450 else 0] += 1
                    self.reset_positions()
                else: 
                    self.ball['vx'] *= -0.8
                    self.ball['x'] = max(30, min(870, self.ball['x']))
            if self.ball['y'] < 30 or self.ball['y'] > 530: 
                self.ball['vy'] *= -0.8
                self.ball['y'] = max(30, min(530, self.ball['y']))

        for p in self.players:
            if math.hypot(self.ball['x'] - p['x'], self.ball['y'] - p['y']) < 40:
                self.ball['holder'] = p['pid']
        if self.ball['holder']:
            h = self.players[self.ball['holder']-1]
            self.ball['x'], self.ball['y'] = h['x'], h['y']

    def reset_positions(self):
        self.ball = {"x": 450, "y": 280, "vx": 0, "vy": 0, "holder": None}
        for p in self.players: p['vx'] = p['vy'] = 0; p['is_moving'] = False

    def kick(self, p, power):
        angle = DIRECTIONS[p['dir_idx']]
        self.ball['vx'], self.ball['vy'] = math.cos(angle)*power, math.sin(angle)*power
        self.ball['holder'] = None

    async def main_loop(self):
        while True:
            self.update_physics()
            state = json.dumps({
                "type": "state", "phase": self.phase, "ball": self.ball, "score": self.score, 
                "time": max(0, self.time_left), "frames": self.frames,
                "players": [{"x":p['x'], "y":p['y'], "team":p['team'], "angle":DIRECTIONS[p['dir_idx']], "pid":p['pid']} for p in self.players]
            })
            if self.clients:
                await asyncio.gather(*[c.send(state) for c in self.clients], return_exceptions=True)
            await asyncio.sleep(1/TICK_RATE)

    async def run(self):
        async with websockets.serve(self.handler, "0.0.0.0", 8760, max_size=2**24):
            print("--- GESTUREBALL HUB ONLINE (ST-GCN) ---")
            await self.main_loop()

if __name__ == "__main__":
    asyncio.run(GameServer().run())