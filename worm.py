import pygame
import random
import math
import time
import array
import os
from concurrent.futures import ThreadPoolExecutor

# =========================
# INIT
# =========================
pygame.init()

SOUND_OK = True
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=1)
except pygame.error:
    SOUND_OK = False

WIDTH, HEIGHT = 900, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

NORMAL, CANNIBAL, SAFE, KING = 0, 1, 2, 3

# Core knobs
START_WORM_COUNT = 2
START_FOOD_COUNT = 9999

FRENZY_POP_THRESHOLD = 25
FRENZY_CHANCE = 0.0005
FRENZY_RADIUS = 120
FRENZY_DURATION = 150
FRENZY_MIN_GROUP = 3

KING_MUTATION_CHANCE = 0.003
KING_BIRTH_NORMAL_CHANCE = 0.98
KING_REPRO_MIN = 1
KING_REPRO_MAX = 10

DEBUG_LOG = True
LOG_EVERY_FRAMES = 30

# Spatial hash: keeps neighbor lookups cheap
CELL_SIZE = 40

def activate(x):
    return math.tanh(x)

def log(worm_id, msg):
    if DEBUG_LOG:
        print(f"[{time.strftime('%H:%M:%S')}] Worm {worm_id}: {msg}")

def make_tone(freq=440, duration=0.08, volume=0.3):
    if not SOUND_OK:
        return None
    sr = 44100
    n = int(sr * duration)
    buf = array.array("h")
    for i in range(n):
        t = i / sr
        env = 1 - (t / duration)
        val = math.sin(2 * math.pi * freq * t) * env
        buf.append(int(val * 32767 * volume))
    return pygame.mixer.Sound(buffer=buf)

SOUNDS = {
    "eat": make_tone(700, 0.05, 0.45),
    "fear": make_tone(200, 0.10, 0.45),
    "pop": make_tone(120, 0.12, 0.55),
    "frenzy": make_tone(260, 0.09, 0.35),
    "bored": make_tone(420, 0.07, 0.25),
    "spare": make_tone(900, 0.05, 0.30),
}

def play(name):
    snd = SOUNDS.get(name)
    if snd is not None:
        snd.play()

def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def kind_name(kind):
    return "N" if kind == NORMAL else "C" if kind == CANNIBAL else "S" if kind == SAFE else "K"

def king_present(worms):
    return any((not w.dead) and w.kind == KING for w in worms)

# =========================
# SPATIAL HASH
# =========================
class SpatialHash:
    def __init__(self, cell_size, width, height):
        self.cell = cell_size
        self.w = width
        self.h = height
        self.grid = {}

    def _key(self, x, y):
        return (int(x) // self.cell, int(y) // self.cell)

    def clear(self):
        self.grid.clear()

    def insert(self, idx, x, y):
        k = self._key(x, y)
        self.grid.setdefault(k, []).append(idx)

    def query(self, x, y, radius):
        cx, cy = self._key(x, y)
        r = int(math.ceil(radius / self.cell))
        out = []
        for gx in range(cx - r, cx + r + 1):
            for gy in range(cy - r, cy + r + 1):
                out.extend(self.grid.get((gx, gy), []))
        return out

# =========================
# WORM
# =========================
class Worm:
    id_counter = 0

    def __init__(self, x=None, y=None, kind=None):
        self.id = Worm.id_counter
        Worm.id_counter += 1

        self.x = x if x is not None else random.randint(0, WIDTH)
        self.y = y if y is not None else random.randint(0, HEIGHT)
        self.angle = random.random() * 2 * math.pi

        self.kind = kind if kind is not None else random.choice([NORMAL, CANNIBAL, SAFE])

        self.speed = 2.0 if self.kind != KING else 0.65
        self.size = 4.0 if self.kind != KING else 6.0

        self.turn_bias = random.uniform(0.8, 1.2)
        self.speed_bias = random.uniform(0.8, 1.2)

        self.hunger = random.uniform(0.55, 1.0)
        self.fear = 0.0
        self.boredom = random.uniform(0, 1)
        self.mercy = random.uniform(0, 1) if self.kind == SAFE else 0.0

        self.repro_timer = random.randint(500, 1000) if self.kind != KING else random.randint(KING_REPRO_MIN, KING_REPRO_MAX)

        self.starving = False
        self.starve_countdown = 0

        self.fog = False
        self.fog_timer = 0

        self.frenzy = False
        self.frenzy_timer = 0
        self.frenzy_target_id = None

        self.splitting = False
        self.split_countdown = 0

        self.dead = False
        self.frame_counter = 0
        self.last_state_log = 0

    def strength(self):
        return self.size + (1.0 - self.hunger)

    def snapshot(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "kind": self.kind,
            "size": self.size,
            "hunger": self.hunger,
            "boredom": self.boredom,
            "fear": self.fear,
            "angle": self.angle,
            "frenzy": self.frenzy,
            "frenzy_target_id": self.frenzy_target_id,
            "dead": self.dead,
            "starving": self.starving,
            "splitting": self.splitting,
        }

    def nearest_food(self, food):
        best = None
        best_d = 10**9
        for fx, fy in food:
            d = math.hypot(fx - self.x, fy - self.y)
            if d < best_d:
                best_d = d
                best = (fx, fy)
        return best, best_d

    def get_target_by_id(self, worms, target_id):
        if target_id is None:
            return None
        for w in worms:
            if (not w.dead) and w.id == target_id and (not w.starving) and (not w.splitting):
                return w
        return None

    def apply_boundary(self):
        r = max(2.0, self.size)

        if self.x < r:
            self.x = r
            self.angle = math.pi - self.angle
        elif self.x > WIDTH - r:
            self.x = WIDTH - r
            self.angle = math.pi - self.angle

        if self.y < r:
            self.y = r
            self.angle = -self.angle
        elif self.y > HEIGHT - r:
            self.y = HEIGHT - r
            self.angle = -self.angle

        self.angle = wrap_angle(self.angle)

    # -------------------------
    # PLANNING PHASE
    # -------------------------
    def perceive(self, worm_snapshots, food_positions, worm_grid, food_grid, king_on):
        """
        Read-only sensing. This is the part we parallelize.
        """
        x, y = self.x, self.y

        # local sensory vectors
        seek_x = 0.0
        seek_y = 0.0
        flee_x = 0.0
        flee_y = 0.0
        sep_x = 0.0
        sep_y = 0.0
        fear = 0.0

        # food
        food_candidates = food_grid.query(x, y, 220)
        best_food = None
        best_food_d = 10**9

        for idx in food_candidates:
            fx, fy = food_positions[idx]
            d = math.hypot(fx - x, fy - y)
            if d < best_food_d:
                best_food_d = d
                best_food = (fx, fy)

        if best_food is not None and best_food_d < 220:
            dx = best_food[0] - x
            dy = best_food[1] - y
            d = max(1.0, math.hypot(dx, dy))
            seek_x += dx / d
            seek_y += dy / d

        # worm neighborhood
        nearby = worm_grid.query(x, y, 120)
        for idx in nearby:
            ws = worm_snapshots[idx]
            if ws["id"] == self.id or ws["dead"] or ws["starving"] or ws["splitting"]:
                continue

            dx = ws["x"] - x
            dy = ws["y"] - y
            d = math.hypot(dx, dy)
            if d <= 0.0001:
                continue

            # separation: avoid bumping by default
            if d < 18:
                weight = (18 - d) / 18
                sep_x -= (dx / d) * weight
                sep_y -= (dy / d) * weight

            # danger and attraction logic
            if self.kind == NORMAL:
                if ws["kind"] == CANNIBAL:
                    fear += 1.0 / (d + 1.0)
                    flee_x -= dx / d
                    flee_y -= dy / d
                elif ws["frenzy"] and ws["frenzy_target_id"] == self.id:
                    fear += 0.8 / (d + 1.0)
                    flee_x -= dx / d
                    flee_y -= dy / d

            elif self.kind == CANNIBAL:
                if ws["kind"] == SAFE and ws["size"] + (1.0 - ws["hunger"]) >= self.strength():
                    fear += 0.8 / (d + 1.0)
                    flee_x -= dx / d
                    flee_y -= dy / d
                elif ws["kind"] in (NORMAL, SAFE, KING):
                    seek_x += 1.1 * dx / d
                    seek_y += 1.1 * dy / d

            elif self.kind == SAFE:
                if ws["kind"] == CANNIBAL:
                    seek_x += 1.0 * dx / d
                    seek_y += 1.0 * dy / d
                    if ws["size"] + (1.0 - ws["hunger"]) >= self.strength():
                        fear += 0.7 / (d + 1.0)
                        flee_x -= dx / d
                        flee_y -= dy / d

            # king makes normals/safes more anti-cannibal
            if king_on and self.kind in (NORMAL, SAFE) and ws["kind"] == CANNIBAL:
                fear *= 0.8

        return {
            "seek_x": seek_x,
            "seek_y": seek_y,
            "flee_x": flee_x,
            "flee_y": flee_y,
            "sep_x": sep_x,
            "sep_y": sep_y,
            "fear": min(1.0, fear * 0.85),
        }

    # -------------------------
    # STATE UPDATES
    # -------------------------
    def maybe_enter_fog(self):
        if (not self.fog) and random.random() < 0.002:
            self.fog = True
            self.fog_timer = random.randint(20, 80)
            log(self.id, "entered mental fog")

    def maybe_start_frenzy(self, worms):
        alive_count = sum(1 for w in worms if not w.dead)
        if self.kind != NORMAL:
            return
        if alive_count <= FRENZY_POP_THRESHOLD:
            return
        if self.frenzy:
            return
        if random.random() < FRENZY_CHANCE:
            self.frenzy = True
            self.frenzy_timer = FRENZY_DURATION
            candidates = [w for w in worms if w is not self and not w.dead and not w.starving and not w.splitting]
            self.frenzy_target_id = random.choice(candidates).id if candidates else None
            play("frenzy")
            log(self.id, f"FRENZY started target={self.frenzy_target_id}")

    def spread_frenzy(self, worms):
        if not self.frenzy:
            return
        target = self.get_target_by_id(worms, self.frenzy_target_id)
        if target is None:
            self.frenzy = False
            self.frenzy_target_id = None
            return

        for w in worms:
            if w.dead or w.kind != NORMAL or w.frenzy or w.starving or w.splitting:
                continue
            d = math.hypot(w.x - self.x, w.y - self.y)
            if d < FRENZY_RADIUS and random.random() < 0.08:
                w.frenzy = True
                w.frenzy_timer = FRENZY_DURATION
                w.frenzy_target_id = self.frenzy_target_id
                log(w.id, "joined frenzy")

    def can_eat_target(self, target):
        if target.dead or target is self or target.starving or target.splitting:
            return False

        if self.kind == NORMAL:
            return False

        if self.kind == CANNIBAL:
            if target.kind == NORMAL:
                return self.strength() >= target.strength() or random.random() < 0.5
            if target.kind == SAFE:
                return random.random() < 0.25 and self.strength() >= target.strength()
            if target.kind == KING:
                return self.strength() >= target.strength() or random.random() < 0.35
            return False

        if self.kind == SAFE:
            return target.kind == CANNIBAL and self.strength() >= target.strength()

        return False

    def start_split(self):
        self.splitting = True
        self.split_countdown = 40
        log(self.id, "starting split")

    def spawn_split_children(self):
        if self.kind == NORMAL:
            pool = [SAFE, CANNIBAL]
        elif self.kind == CANNIBAL:
            pool = [NORMAL, SAFE]
        elif self.kind == SAFE:
            pool = [NORMAL, CANNIBAL]
        else:
            pool = [NORMAL]

        if random.random() < KING_MUTATION_CHANCE:
            child_kinds = [KING, random.choice(pool)]
        else:
            child_kinds = [random.choice(pool), random.choice(pool)]

        children = []
        offsets = [(-10, -6), (10, 6)]
        for idx in range(2):
            ox, oy = offsets[idx]
            cx = clamp(self.x + ox, 0, WIDTH)
            cy = clamp(self.y + oy, 0, HEIGHT)

            child = Worm(cx, cy, child_kinds[idx])
            child.angle = self.angle + random.uniform(-0.6, 0.6)
            child.turn_bias = clamp(self.turn_bias + random.uniform(-0.1, 0.1), 0.5, 1.5)
            child.speed_bias = clamp(self.speed_bias + random.uniform(-0.1, 0.1), 0.5, 1.5)
            child.hunger = random.uniform(0.55, 1.0)
            child.repro_timer = random.randint(500, 1000) if child.kind != KING else random.randint(KING_REPRO_MIN, KING_REPRO_MAX)
            child.mercy = random.uniform(0, 1) if child.kind == SAFE else 0.0
            if child.kind == KING:
                child.speed = 0.65
                child.size = 6.0
            children.append(child)

        return children

    def spawn_king_child(self):
        child_kind = NORMAL if random.random() < KING_BIRTH_NORMAL_CHANCE else CANNIBAL
        child = Worm(self.x, self.y, child_kind)
        child.x = clamp(self.x + random.uniform(-6, 6), 0, WIDTH)
        child.y = clamp(self.y + random.uniform(-6, 6), 0, HEIGHT)
        child.angle = self.angle + random.uniform(-0.5, 0.5)
        child.turn_bias = clamp(self.turn_bias + random.uniform(-0.08, 0.08), 0.5, 1.5)
        child.speed_bias = clamp(self.speed_bias + random.uniform(-0.08, 0.08), 0.5, 1.5)
        child.hunger = random.uniform(0.6, 1.0)
        child.repro_timer = random.randint(500, 1000)
        return child

    def tick_internal_state(self):
        self.frame_counter += 1
        self.hunger = max(0.0, self.hunger - (0.0007 if self.kind == KING else 0.0010))
        self.boredom = min(1.0, self.boredom + 0.002)

        if self.hunger <= 0.0 and not self.starving and not self.splitting:
            self.starving = True
            self.starve_countdown = 40
            log(self.id, "starving")

        self.maybe_enter_fog()

    def resolve_fog(self):
        self.fog_timer -= 1
        self.angle += random.uniform(-0.02, 0.02)
        self.x += math.cos(self.angle) * 0.15
        self.y += math.sin(self.angle) * 0.15
        self.apply_boundary()
        if self.fog_timer <= 0:
            self.fog = False
            log(self.id, "recovered from fog")

    def resolve_starvation(self):
        self.starve_countdown -= 1
        self.x += random.uniform(-1.0, 1.0)
        self.y += random.uniform(-1.0, 1.0)
        self.apply_boundary()
        if self.starve_countdown <= 0:
            play("pop")
            log(self.id, "died of hunger")
            self.dead = True

    def resolve_splitting(self, newborns):
        self.split_countdown -= 1
        self.x += random.uniform(-1.6, 1.6)
        self.y += random.uniform(-1.6, 1.6)
        self.size += 0.05
        self.apply_boundary()
        if self.split_countdown <= 0:
            children = self.spawn_split_children()
            newborns.extend(children)
            play("pop")
            log(self.id, f"split into {children[0].id} and {children[1].id}")
            self.dead = True

    def resolve_king(self, food, newborns):
        self.angle += random.uniform(-0.02, 0.02)
        self.x += math.cos(self.angle) * 0.12
        self.y += math.sin(self.angle) * 0.12
        self.apply_boundary()

        # eat food
        for i in range(len(food) - 1, -1, -1):
            fx, fy = food[i]
            if math.hypot(self.x - fx, self.y - fy) < 7:
                food.pop(i)
                self.hunger = min(1.0, self.hunger + 0.35)
                play("eat")
                break

        self.repro_timer -= 1
        if self.repro_timer <= 0 and self.hunger > 0.30:
            child = self.spawn_king_child()
            newborns.append(child)
            log(self.id, f"spawned {child.id} [{kind_name(child.kind)}]")
            self.repro_timer = random.randint(KING_REPRO_MIN, KING_REPRO_MAX)

        if self.frame_counter - self.last_state_log >= LOG_EVERY_FRAMES:
            self.last_state_log = self.frame_counter
            log(self.id, f"[K] H:{self.hunger:.2f} B:{self.boredom:.2f}")

    def resolve_frenzy(self, worms):
        self.frenzy_timer -= 1
        target = self.get_target_by_id(worms, self.frenzy_target_id)
        if target is None:
            self.frenzy = False
            self.frenzy_target_id = None
            return

        if self.id == self.frenzy_target_id:
            mobs = [
                w for w in worms
                if w.kind == NORMAL and w.frenzy and not w.dead and not w.starving and not w.splitting
                and w.frenzy_target_id == self.id
            ]
            if mobs:
                vx = self.x - sum(w.x for w in mobs) / len(mobs)
                vy = self.y - sum(w.y for w in mobs) / len(mobs)
                d = max(1.0, math.hypot(vx, vy))
                self.angle = math.atan2(vy, vx)
                self.x += (vx / d) * 2.5
                self.y += (vy / d) * 2.5
                self.apply_boundary()
        else:
            dx = target.x - self.x
            dy = target.y - self.y
            ang = math.atan2(dy, dx)
            self.angle += wrap_angle(ang - self.angle) * 0.25
            self.x += math.cos(self.angle) * 3.0
            self.y += math.sin(self.angle) * 3.0
            self.apply_boundary()

        if target and not target.dead:
            near = [
                w for w in worms
                if w.kind == NORMAL and w.frenzy and not w.dead and not w.starving and not w.splitting
                and w.frenzy_target_id == target.id
                and math.hypot(w.x - target.x, w.y - target.y) < 15
            ]
            if len(near) >= FRENZY_MIN_GROUP:
                target.dead = True
                play("pop")
                log(self.id, f"group killed {target.id}")
                self.frenzy = False
                self.frenzy_target_id = None
                return

        if self.frenzy_timer <= 0:
            self.frenzy = False
            self.frenzy_target_id = None
            log(self.id, "calmed down from frenzy")

    def move_from_perception(self, p, king_on):
        # if feared or in normal mode, use the perception vectors
        seek_x = p["seek_x"]
        seek_y = p["seek_y"]
        flee_x = p["flee_x"]
        flee_y = p["flee_y"]
        sep_x = p["sep_x"]
        sep_y = p["sep_y"]
        fear = p["fear"]

        desire_x = seek_x + 1.8 * flee_x + 1.0 * sep_x
        desire_y = seek_y + 1.8 * flee_y + 1.0 * sep_y

        if fear > 0.25:
            desire_x = 0.7 * seek_x + 2.4 * flee_x + 1.0 * sep_x
            desire_y = 0.7 * seek_y + 2.4 * flee_y + 1.0 * sep_y

        if king_on and self.kind in (NORMAL, SAFE):
            desire_x += 0.8 * seek_x
            desire_y += 0.8 * seek_y

        if desire_x == 0.0 and desire_y == 0.0:
            desire_x = math.cos(self.angle)
            desire_y = math.sin(self.angle)

        target_angle = math.atan2(desire_y, desire_x)
        turn = wrap_angle(target_angle - self.angle)
        self.angle += turn * 0.18 + random.uniform(-0.06, 0.06)

        move_speed = self.speed + activate(self.hunger + self.boredom * 0.25) + random.uniform(-0.25, 0.35)
        self.x += math.cos(self.angle) * move_speed
        self.y += math.sin(self.angle) * move_speed
        self.apply_boundary()

    def resolve_food_and_bites(self, food, worms):
        # eat food, finite and non-respawning
        bite_range = 8

        for i in range(len(food) - 1, -1, -1):
            fx, fy = food[i]
            if math.hypot(self.x - fx, self.y - fy) < 7:
                food.pop(i)
                self.hunger = min(1.0, self.hunger + 0.45)
                play("eat")
                break

        # bite other worms only when actually close enough
        for w in worms:
            if w.dead or w is self or w.starving or w.splitting:
                continue

            d = math.hypot(self.x - w.x, self.y - w.y)
            if d >= bite_range:
                continue

            if self.kind == CANNIBAL and self.can_eat_target(w):
                w.dead = True
                self.hunger = min(1.0, self.hunger + 0.35)
                play("eat")
                log(self.id, f"ate {w.id}")
                break

            if self.kind == SAFE and self.can_eat_target(w):
                w.dead = True
                self.hunger = min(1.0, self.hunger + 0.35)
                play("eat")
                log(self.id, f"ate cannibal {w.id}")
                break

            if self.kind == NORMAL and w.kind == CANNIBAL:
                # last-second fear pivot
                self.angle = math.atan2(self.y - w.y, self.x - w.x)

    def update(self, perception, worms, food, newborns, king_on):
        if self.dead:
            return

        self.tick_internal_state()

        if self.starving:
            self.resolve_starvation()
            return

        if self.fog:
            self.resolve_fog()
            return

        if self.kind == KING:
            self.resolve_king(food, newborns)
            return

        if self.splitting:
            self.resolve_splitting(newborns)
            return

        # frenzy only really belongs to normals
        if self.kind == NORMAL:
            if not self.frenzy:
                self.maybe_start_frenzy(worms)
                self.spread_frenzy(worms)
            if self.frenzy:
                self.resolve_frenzy(worms)
                return

        self.fear = perception["fear"]

        # boredom can cause tiny wander spikes
        if self.boredom > 0.9 and random.random() < 0.01:
            play("bored")

        self.move_from_perception(perception, king_on)
        self.resolve_food_and_bites(food, worms)

        # reproduction
        self.repro_timer -= 1
        if self.repro_timer <= 0 and self.hunger > 0.72:
            if self.kind == KING:
                # not used, but kept consistent
                child = self.spawn_king_child()
                newborns.append(child)
            else:
                self.start_split()

        if self.frame_counter - self.last_state_log >= LOG_EVERY_FRAMES:
            self.last_state_log = self.frame_counter
            log(self.id, f"[{kind_name(self.kind)}] H:{self.hunger:.2f} F:{self.fear:.2f} B:{self.boredom:.2f}")

    def draw(self, surf):
        if self.kind == NORMAL:
            color = (255, 255, 255)
        elif self.kind == CANNIBAL:
            color = (255, 0, 0)
        elif self.kind == SAFE:
            color = (255, 255, 0)
        else:
            color = (0, 255, 255)

        radius = max(2, int(self.size))
        pygame.draw.circle(surf, color, (int(self.x), int(self.y)), radius)

        if self.frenzy and not self.dead:
            pygame.draw.circle(surf, (255, 120, 0), (int(self.x), int(self.y)), radius + 3, 1)

# =========================
# WORLD
# =========================
worms = [Worm() for _ in range(START_WORM_COUNT)]
food = [(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(START_FOOD_COUNT)]

# thread pool for read-only perception pass
max_workers = max(1, min(4, (os.cpu_count() or 1)))
executor = ThreadPoolExecutor(max_workers=max_workers)

running = True
while running:
    screen.fill((0, 0, 0))

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    # remove dead first
    worms = [w for w in worms if not w.dead]

    # build spatial indices from current frame
    worm_grid = SpatialHash(CELL_SIZE, WIDTH, HEIGHT)
    food_grid = SpatialHash(CELL_SIZE, WIDTH, HEIGHT)

    worm_snaps = []
    for idx, w in enumerate(worms):
        worm_grid.insert(idx, w.x, w.y)
        worm_snaps.append(w.snapshot())

    for idx, (fx, fy) in enumerate(food):
        food_grid.insert(idx, fx, fy)

    king_on = any(w.kind == KING and not w.dead for w in worms)

    # threaded sensory pass
    futures = [
        executor.submit(
            worms[i].perceive,
            worm_snaps,
            food,
            worm_grid,
            food_grid,
            king_on
        )
        for i in range(len(worms))
    ]
    perceptions = [f.result() for f in futures]

    newborns = []

    # apply updates sequentially to keep state safe and deterministic
    for w, p in zip(worms, perceptions):
        if not w.dead:
            w.update(p, worms, food, newborns, king_on)

    # append newborns and drop dead
    worms.extend(newborns)
    worms = [w for w in worms if not w.dead]

    # draw food
    for fx, fy in food:
        pygame.draw.circle(screen, (0, 255, 0), (fx, fy), 3)

    # draw worms
    for w in worms:
        w.draw(screen)

    pygame.display.flip()
    clock.tick(60)

executor.shutdown(wait=False, cancel_futures=True)
pygame.quit()
