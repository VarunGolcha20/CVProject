import asyncio, json, math, time, websockets

W, H = 900, 560 
FIELD = {"x": 30, "y": 40, "w": 840, "h": 480}
GOAL_H, TICK_RATE = 160, 60
GY1, GY2 = (H - GOAL_H) // 2, (H + GOAL_H) // 2
ROTATION_SPEED, GAME_DURATION = 0.3, 300.0 
MOVE_GRACE = 0.5 
DIRECTIONS = [4.71, 5.49, 0, 0.785, 1.57, 2.35, 3.14, 3.92]

class GameServer:
    def __init__(self):
        self.players = []
        self.ball = {"x": 450, "y": 280, "vx": 0, "vy": 0, "holder": None, "last_capture_time": 0}
        self.phase = "waiting" 
        self.score = [0, 0]
        self.time_left = GAME_DURATION
        self.winner = None
        self.browsers = {} # websocket: {"pid": id, "lat": 0}
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
                "last_scan": 0, "is_moving": False, "last_move_time": 0, "kick_cd": 0
            })
        print(f"[LOG] Match Init: {count} Players Created")

    async def browser_handler(self, ws):
        self.browsers[ws] = {"pid": None, "lat": 0}
        try:
            async for msg in ws:
                data = json.loads(msg)
                m_type = data.get('type')
                
                if m_type == 'ping':
                    await ws.send(json.dumps({"type": "pong", "time": data['time']}))
                    continue
                
                if m_type == 'lat_update':
                    self.browsers[ws]["lat"] = data['lat']
                    continue

                if m_type == 'ready':
                    cnt = int(data.get('count', 4))
                    self.setup_players(cnt)
                    self.score, self.time_left, self.phase = [0, 0], GAME_DURATION, "playing"
                    self.reset_positions()
                    print(f"[LOG] Kickoff Started: {cnt}p Mode")
                
                elif m_type == 'pause':
                    self.phase = "paused" if self.phase == "playing" else "playing"
                    print(f"[LOG] Game Phase: {self.phase}")

                elif m_type == 'restart':
                    self.phase = "waiting"
                    self.reset_positions()
                    print("[LOG] Game Reset to Menu")

                elif m_type == 'input':
                    p_id = int(data['player'])
                    self.browsers[ws]["pid"] = p_id
                    if self.phase == "playing":
                        p = self.players[p_id - 1]
                        if data['move'] == "MOVE": p['is_moving'] = True
                        elif data['move'] == "STOP": p['is_moving'] = False; p['last_move_time'] = time.time()
                        elif data['move'] == "SHOOT" and self.ball['holder'] == p['pid']: 
                            print(f"[LOG] P{p_id} Shoot")
                            self.kick(p, 26)
                        elif data['move'] == "PASS" and self.ball['holder'] == p['pid']: 
                            print(f"[LOG] P{p_id} Pass")
                            self.kick(p, 14)
        finally:
            del self.browsers[ws]

    def kick(self, p, power):
        angle = DIRECTIONS[p['dir_idx']]
        self.ball['vx'], self.ball['vy'] = math.cos(angle)*power, math.sin(angle)*power
        self.ball['holder'], p['kick_cd'] = None, 20

    def update_physics(self):
        if self.phase != "playing": return
        
        self.time_left -= 1/TICK_RATE
        if self.time_left <= 0:
            self.time_left = 0
            self.winner = "ALPHA" if self.score[0] > self.score[1] else ("OMEGA" if self.score[1] > self.score[0] else "DRAW")
            self.phase = "gameover"
            print(f"[LOG] Match Ended. Winner: {self.winner}")
            return

        now = time.time()
        for p in self.players:
            if p['kick_cd'] > 0: p['kick_cd'] -= 1
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
            p['x'], p['y'] = max(20, min(880, p['x'])), max(20, min(540, p['y']))

        # Steal Logic
        grace = (now - self.ball['last_capture_time'] < 1.0)
        candidates = []
        for p in self.players:
            d = math.hypot(self.ball['x'] - p['x'], self.ball['y'] - p['y'])
            if d < 42 and p['kick_cd'] == 0:
                priority = 1 if p['pid'] != self.ball['holder'] else 0
                candidates.append((priority, d, p))
        
        if candidates:
            candidates.sort(key=lambda x: (-x[0], x[1]))
            best_p = candidates[0][2]
            if self.ball['holder'] is None or (self.ball['holder'] != best_p['pid'] and not grace):
                self.ball['holder'] = best_p['pid']
                self.ball['vx'] = self.ball['vy'] = 0
                self.ball['last_capture_time'] = now
                print(f"[LOG] Ball Possession: P{best_p['pid']}")

        if self.ball['holder']:
            h = self.players[self.ball['holder']-1]
            self.ball['x'], self.ball['y'] = h['x'], h['y']
        else:
            self.ball['x'] += self.ball['vx']; self.ball['y'] += self.ball['vy']
            self.ball['vx'] *= 0.985; self.ball['vy'] *= 0.985
            if self.ball['y'] < 20 or self.ball['y'] > 540: self.ball['vy'] *= -0.8; self.ball['y'] = max(20, min(540, self.ball['y']))
            if self.ball['x'] < 20 or self.ball['x'] > 880:
                if GY1 < self.ball['y'] < GY2:
                    scorer = 1 if self.ball['x'] < 450 else 0
                    self.score[scorer] += 1
                    print(f"[LOG] GOAL! Team {'OMEGA' if scorer else 'ALPHA'} scored. Current Score: {self.score}")
                    self.reset_positions()
                else:
                    self.ball['vx'] *= -0.8; self.ball['x'] = max(20, min(880, self.ball['x']))

    def reset_positions(self):
        self.ball = {"x": 450, "y": 280, "vx": 0, "vy": 0, "holder": None, "last_capture_time": 0}
        count = len(self.players)
        half = count // 2
        for i, p in enumerate(self.players):
            team = 0 if i < half else 1
            p['x'] = 200 if team == 0 else 700
            p['y'] = 100 + (i % half) * (360 // max(1, half-1)) if half > 1 else 280
            p['vx'] = p['vy'] = 0; p['is_moving'] = False; p['kick_cd'] = 0; p['dir_idx'] = 2 if team == 0 else 6

    async def main_loop(self):
        while True:
            self.update_physics()
            latencies = {v['pid']: v['lat'] for v in self.browsers.values() if v['pid'] is not None}
            state = json.dumps({
                "type": "state", "phase": self.phase, "ball": self.ball, "score": self.score, 
                "time": self.time_left, "winner": self.winner, "lats": latencies,
                "players": [{"x":p['x'], "y":p['y'], "team":p['team'], "angle":DIRECTIONS[p['dir_idx']], "pid":p['pid']} for p in self.players]
            })
            if self.browsers:
                await asyncio.gather(*[b.send(state) for b in self.browsers.keys()], return_exceptions=True)
            await asyncio.sleep(1/TICK_RATE)

    async def run(self):
        async with websockets.serve(self.browser_handler, "0.0.0.0", 8760):
            print("--- GESTUREBALL SERVER ONLINE ---")
            await self.main_loop()

if __name__ == "__main__":
    asyncio.run(GameServer().run())