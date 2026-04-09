import asyncio
import json
import math
import time
import websockets

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════
W, H = 900, 560 
FIELD = {"x": 30, "y": 40, "w": 840, "h": 480}
GOAL_H = 160
GY1, GY2 = (H - GOAL_H) // 2, (H + GOAL_H) // 2
TICK_RATE = 60

# 8 Tactical Directions (N, NE, E, SE, S, SW, W, NW)
DIRECTIONS = [4.71, 5.49, 0, 0.785, 1.57, 2.35, 3.14, 3.92]

class GameServer:
    def __init__(self):
        self.players = []
        self.ball = {
            "x": 450, "y": 280, "vx": 0, "vy": 0, 
            "holder": None, "last_capture_time": 0
        }
        self.phase = "waiting" # waiting, playing, paused
        self.score = [0, 0]
        self.time_left = 180.0
        self.browsers = set()
        self.setup_players(4) # Default start

    def setup_players(self, count):
        """Initializes teams and spawns players in formation"""
        self.players = []
        half = count // 2
        for i in range(count):
            team = 0 if i < half else 1
            x = 200 if team == 0 else 700
            # Distribute players vertically based on team size
            y = 100 + (i % half) * (360 // max(1, half-1)) if half > 1 else 280
            self.players.append({
                "pid": i + 1, "team": team, "x": x, "y": y, 
                "vx": 0, "vy": 0, "dir_idx": 2 if team == 0 else 6, 
                "last_scan": 0, "is_moving": False, "last_move_time": 0, "kick_cd": 0
            })
        print(f"[SERVER] Setup {count} players.")

    async def browser_handler(self, ws):
        self.browsers.add(ws)
        try:
            async for msg in ws:
                data = json.loads(msg)
                m_type = data.get('type')
                
                if m_type == 'ready':
                    self.setup_players(int(data.get('count', 4)))
                    self.score = [0, 0]
                    self.time_left = 180.0
                    self.reset_positions()
                    self.phase = "playing"
                
                elif m_type == 'pause':
                    if self.phase == "playing": self.phase = "paused"
                    elif self.phase == "paused": self.phase = "playing"
                
                elif m_type == 'restart':
                    self.phase = "waiting"
                    self.score = [0, 0]
                    self.time_left = 180.0
                    self.reset_positions()

                elif m_type == 'input' and self.phase == "playing":
                    p_idx = int(data['player']) - 1
                    if 0 <= p_idx < len(self.players):
                        p = self.players[p_idx]
                        move = data['move']
                        if move == "MOVE": p['is_moving'] = True
                        elif move == "STOP": 
                            p['is_moving'] = False
                            p['last_move_time'] = time.time()
                        elif move == "SHOOT" and self.ball['holder'] == p['pid']: 
                            self.kick(p, 26)
                        elif move == "PASS" and self.ball['holder'] == p['pid']: 
                            self.kick(p, 14)
        finally:
            self.browsers.discard(ws)

    def kick(self, p, power):
        angle = DIRECTIONS[p['dir_idx']]
        self.ball['vx'] = math.cos(angle) * power
        self.ball['vy'] = math.sin(angle) * power
        self.ball['holder'] = None
        p['kick_cd'] = 20 # Protects from immediate self-capture

    def update_physics(self):
        if self.phase != "playing": return
        self.time_left -= 1/TICK_RATE
        now = time.time()
        
        # 1. Update Player Movement
        for p in self.players:
            if p['kick_cd'] > 0: p['kick_cd'] -= 1
            
            # Grace period for direction rotation
            if not p['is_moving'] and (now - p['last_move_time'] >= 0.5):
                if now - p['last_scan'] >= 0.5:
                    p['dir_idx'] = (p['dir_idx'] + 1) % 8
                    p['last_scan'] = now
            
            if p['is_moving']:
                angle = DIRECTIONS[p['dir_idx']]
                p['vx'], p['vy'] = math.cos(angle)*8.2, math.sin(angle)*8.2
            else:
                p['vx'] *= 0.82; p['vy'] *= 0.82

            p['x'] += p['vx']
            p['y'] += p['vy']
            p['x'], p['y'] = max(20, min(880, p['x'])), max(20, min(540, p['y']))

        # 2. COMPETITIVE STEAL LOGIC
        # 1-second protection for the person who just stole it
        grace_active = (now - self.ball['last_capture_time'] < 1.0)
        
        candidates = []
        for p in self.players:
            d = math.hypot(self.ball['x'] - p['x'], self.ball['y'] - p['y'])
            if d < 42 and p['kick_cd'] == 0:
                # Stealer Priority: 1 (not holder), Holder Priority: 0
                priority = 1 if p['pid'] != self.ball['holder'] else 0
                candidates.append((priority, d, p))

        if candidates:
            # Sort: Priority first (Stealers), then distance (Closest)
            candidates.sort(key=lambda x: (-x[0], x[1]))
            best_p = candidates[0][2]
            
            # Allow capture if ball is free OR if 1s protection has expired for a steal
            if self.ball['holder'] is None:
                self.capture_ball(best_p, now)
            elif self.ball['holder'] != best_p['pid'] and not grace_active:
                self.capture_ball(best_p, now)

        # 3. Ball Physics
        if self.ball['holder']:
            h = self.players[self.ball['holder']-1]
            self.ball['x'], self.ball['y'] = h['x'], h['y']
        else:
            self.ball['x'] += self.ball['vx']; self.ball['y'] += self.ball['vy']
            self.ball['vx'] *= 0.985; self.ball['vy'] *= 0.985
            
            # Boundaries
            if self.ball['y'] < 20 or self.ball['y'] > 540: 
                self.ball['vy'] *= -0.8
                self.ball['y'] = max(20, min(540, self.ball['y']))
            
            if self.ball['x'] < 20 or self.ball['x'] > 880:
                if GY1 < self.ball['y'] < GY2: # GOAL!
                    idx = 1 if self.ball['x'] < 450 else 0
                    self.score[idx] += 1
                    self.reset_positions()
                else: # Bounce off end-line
                    self.ball['vx'] *= -0.8
                    self.ball['x'] = max(20, min(880, self.ball['x']))

    def capture_ball(self, player, timestamp):
        if self.ball['holder'] != player['pid']:
            self.ball['holder'] = player['pid']
            self.ball['vx'] = self.ball['vy'] = 0
            self.ball['last_capture_time'] = timestamp
            print(f"[STEAL] Ball taken by P{player['pid']}")

    def reset_positions(self):
        self.ball = {"x": 450, "y": 280, "vx": 0, "vy": 0, "holder": None, "last_capture_time": 0}
        for p in self.players:
            p['vx'] = p['vy'] = 0
            p['is_moving'] = False
            p['kick_cd'] = 0

    async def main_loop(self):
        while True:
            self.update_physics()
            state = json.dumps({
                "type": "state", "phase": self.phase, "ball": self.ball, 
                "score": self.score, "time": self.time_left,
                "players": [{"x":p['x'], "y":p['y'], "team":p['team'], "angle":DIRECTIONS[p['dir_idx']], "pid":p['pid']} for p in self.players]
            })
            if self.browsers:
                await asyncio.gather(*[b.send(state) for b in self.browsers], return_exceptions=True)
            await asyncio.sleep(1/TICK_RATE)

    async def run(self):
        async with websockets.serve(self.browser_handler, "0.0.0.0", 8760):
            print("--- GESTUREBALL SERVER RUNNING ON PORT 8760 ---")
            await self.main_loop()

if __name__ == "__main__":
    asyncio.run(GameServer().run())