import asyncio, json, math, time, websockets
import cv2, base64
import numpy as np
import os
import sys

# Ensure parent directory is in path for classifiers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from classifiers.mobilenet_classifier import PretrainedImageClassifier
except ImportError:
    PretrainedImageClassifier = None

W, H = 900, 560 
GOAL_H, TICK_RATE = 160, 60
GY1, GY2 = (H - GOAL_H) // 2, (H + GOAL_H) // 2
# --- FIXED: 0.3s for rotation, 0.5s for grace period ---
ROTATION_SPEED = 0.3 
MOVE_GRACE = 0.5 
GAME_DURATION = 300.0
# Directions: N, NE, E, SE, S, SW, W, NW
DIRECTIONS = [4.71, 5.49, 0, 0.785, 1.57, 2.35, 3.14, 3.92]

class GameServer:
    def __init__(self):
        self.players = []
        self.ball = {"x": 450, "y": 280, "vx": 0, "vy": 0, "holder": None}
        self.phase = "waiting" 
        self.score = [0, 0]
        self.time_left = GAME_DURATION
        self.clients = [] 
        self.frames = {} 
        self.processing_pids = set() # Track players currently being processed

        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "mobilenet_gesture.pt"))
        try:
            self.mnet_clf = PretrainedImageClassifier(model_path=model_path) if PretrainedImageClassifier else None
            print(f"[LOG] MobileNet Loaded: {model_path}")
        except Exception as e:
            print(f"[ERROR] MobileNet Load Failed: {e}")
            self.mnet_clf = None
        
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
                "last_scan": 0, "is_moving": False, "last_move_time": time.time(), "kick_cd": 0
            })

    def process_vision_frame(self, pid, b64_data):
        """CPU-Heavy: Processes frame and returns result."""
        if not self.mnet_clf: return pid, "STOP", b64_data
        try:
            header, encoded = b64_data.split(",", 1)
            nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            label = self.mnet_clf(frame)
            
            cmd = "STOP"
            clean_label = str(label).lower().strip()
            if clean_label == "move": cmd = "MOVE"
            elif clean_label == "shoot": cmd = "SHOOT"
            elif clean_label == "pass": cmd = "PASS"

            cv2.putText(frame, f"P{pid}: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
            out_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
            
            return pid, cmd, out_b64
        except:
            return pid, "STOP", b64_data

    async def async_vision_task(self, pid, frame_data):
        """Parallel worker for vision processing."""
        self.processing_pids.add(pid)
        loop = asyncio.get_running_loop()
        try:
            _, cmd, out_b64 = await loop.run_in_executor(None, self.process_vision_frame, pid, frame_data)
            self.frames[str(pid)] = out_b64
            self.handle_game_input(pid, cmd)
        finally:
            self.processing_pids.remove(pid)

    def handle_game_input(self, p_id, cmd):
        if self.phase != "playing": return
        idx = int(p_id) - 1
        if 0 <= idx < len(self.players):
            p = self.players[idx]
            if cmd == "MOVE": 
                p['is_moving'] = True
            elif cmd == "STOP": 
                if p['is_moving']: # Only reset timer if we were actually moving
                    p['last_move_time'] = time.time()
                p['is_moving'] = False
            elif cmd == "SHOOT" and self.ball['holder'] == p['pid']: self.kick(p, 26)
            elif cmd == "PASS" and self.ball['holder'] == p['pid']: self.kick(p, 14)

    async def handler(self, ws):
        self.clients.append(ws)
        try:
            async for msg in ws:
                data = json.loads(msg)
                m_type = data.get('type')
                
                # --- PARALLEL CONTROLS: Processed instantly ---
                if m_type == 'ready':
                    self.setup_players(int(data.get('count', 4)))
                    self.score, self.time_left, self.phase = [0, 0], GAME_DURATION, "playing"
                    self.reset_positions()
                    self.frames = {}
                elif m_type == 'pause':
                    self.phase = "paused" if self.phase == "playing" else "playing"
                elif m_type == 'restart':
                    self.phase = "waiting"; self.score = [0, 0]; self.reset_positions()
                elif m_type == 'video_frame':
                    pid = int(data['pid'])
                    # DROPPING FRAMES: If still processing P1, don't queue another P1 frame
                    if pid not in self.processing_pids:
                        asyncio.create_task(self.async_vision_task(pid, data['frame']))
                elif m_type == 'ping':
                    await ws.send(json.dumps({"type": "pong", "time": data['time']}))
        finally:
            self.clients.append(ws)

    def kick(self, p, power):
        angle = DIRECTIONS[p['dir_idx']]
        self.ball['vx'], self.ball['vy'] = math.cos(angle)*power, math.sin(angle)*power
        self.ball['holder'] = None

    def update_physics(self):
        if self.phase != "playing": return
        self.time_left -= 1/TICK_RATE
        now = time.time()
        for p in self.players:
            # --- ARROW ROTATION LOGIC ---
            # Only rotate if STOPPED and 0.5s has passed since the last MOVE command
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

        # Ball Physics & Goal Check
        if not self.ball['holder']:
            self.ball['x'] += self.ball['vx']; self.ball['y'] += self.ball['vy']
            self.ball['vx'] *= 0.985; self.ball['vy'] *= 0.985
            if self.ball['x'] < 30 or self.ball['x'] > 870:
                if GY1 < self.ball['y'] < GY2:
                    self.score[1 if self.ball['x'] < 450 else 0] += 1
                    self.reset_positions()
                else: self.ball['vx'] *= -0.8; self.ball['x'] = max(30, min(870, self.ball['x']))
            if self.ball['y'] < 30 or self.ball['y'] > 530: self.ball['vy'] *= -0.8; self.ball['y'] = max(30, min(530, self.ball['y']))

        # Capture
        for p in self.players:
            if math.hypot(self.ball['x'] - p['x'], self.ball['y'] - p['y']) < 40:
                self.ball['holder'] = p['pid']
        if self.ball['holder']:
            h = self.players[self.ball['holder']-1]
            self.ball['x'], self.ball['y'] = h['x'], h['y']

    def reset_positions(self):
        self.ball = {"x": 450, "y": 280, "vx": 0, "vy": 0, "holder": None}
        for p in self.players: p['vx'] = p['vy'] = 0; p['is_moving'] = False

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
            print("--- GESTUREBALL HUB ONLINE ---")
            await self.main_loop()

if __name__ == "__main__":
    asyncio.run(GameServer().run())