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

START_WORM_COUNT = 1
START_FOOD_COUNT = 10000

FRENZY_POP_THRESHOLD = 25
FRENZY_CHANCE        = 0.0005
FRENZY_RADIUS        = 120
FRENZY_DURATION      = 150
FRENZY_MIN_GROUP     = 10

KING_MUTATION_CHANCE    = 0.03
KING_BIRTH_NORMAL_CHANCE = 0.98
KING_REPRO_MIN = 1
KING_REPRO_MAX = 10

DEBUG_LOG      = True
LOG_EVERY_FRAMES = 30

CELL_SIZE = 40

# ── Neural-net constants ──────────────────────────────────────────────────────
# Inputs (11):
#   food_dx, food_dy, food_prox          – direction/proximity to nearest food
#   threat_dx, threat_dy, threat_prox    – direction away from / proximity to nearest threat
#   ally_dx, ally_dy                     – direction toward nearest same-kind
#   hunger, fear, boredom                – internal state
NN_IN      = 11
NN_H       = 8        # hidden layer width
NN_OUT     = 3        # turn_strength, speed_mod, flee_strength  (all ∈ [-1,1])
NN_MUT_RATE = 0.06    # Gaussian σ for weight mutation
NN_MUT_PROB = 0.0015    # probability any single weight mutates
# ─────────────────────────────────────────────────────────────────────────────


def activate(x):
    return math.tanh(x)

def log(worm_id, msg):
    if DEBUG_LOG:
        print(f"[{time.strftime('%H:%M:%S')}] Worm {worm_id}: {msg}")

def make_tone(freq=440, duration=0.08, volume=0.3):
    if not SOUND_OK:
        return None
    sr = 44100
    n  = int(sr * duration)
    buf = array.array("h")
    for i in range(n):
        t   = i / sr
        env = 1 - (t / duration)
        val = math.sin(2 * math.pi * freq * t) * env
        buf.append(int(val * 32767 * volume))
    return pygame.mixer.Sound(buffer=buf)

SOUNDS = {
    "eat":    make_tone(700, 0.05, 0.45),
    "fear":   make_tone(200, 0.10, 0.45),
    "pop":    make_tone(120, 0.12, 0.55),
    "frenzy": make_tone(260, 0.09, 0.35),
    "bored":  make_tone(420, 0.07, 0.25),
    "spare":  make_tone(900, 0.05, 0.30),
}

def play(name):
    snd = SOUNDS.get(name)
    if snd is not None:
        snd.play()

def wrap_angle(angle):
    while angle >  math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def kind_name(kind):
    return "N" if kind == NORMAL else "C" if kind == CANNIBAL else "S" if kind == SAFE else "K"


# =========================
# NEURAL BRAIN
# =========================
class NeuralBrain:
    """
    Tiny 2-layer feedforward network: NN_IN → NN_H → NN_OUT, all tanh.

    Each worm owns one NeuralBrain.  Weights are randomly initialised and
    inherited with Gaussian mutation on reproduction – pure neuroevolution,
    no gradient descent.  This replaces the old turn_bias / speed_bias scalars
    with a full personality vector that can encode complex conditional behaviour
    (e.g. "slow down near threats, sprint toward food when starving").
    """
    __slots__ = ("w1", "b1", "w2", "b2")

    def __init__(self, weights=None):
        if weights is None:
            rg = random.gauss
            self.w1 = [[rg(0, 0.6) for _ in range(NN_H)]   for _ in range(NN_IN)]
            self.b1 = [rg(0, 0.1) for _ in range(NN_H)]
            self.w2 = [[rg(0, 0.6) for _ in range(NN_OUT)] for _ in range(NN_H)]
            self.b2 = [rg(0, 0.1) for _ in range(NN_OUT)]
        else:
            self.w1, self.b1, self.w2, self.b2 = weights

    def forward(self, inp):
        """Pure-Python matrix math; keeps the function picklable and GIL-friendly."""
        tanh = math.tanh
        w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2

        # Hidden layer
        hidden = []
        for j in range(NN_H):
            s = b1[j]
            for i in range(NN_IN):
                s += inp[i] * w1[i][j]
            hidden.append(tanh(s))

        # Output layer
        outputs = []
        for k in range(NN_OUT):
            s = b2[k]
            for j in range(NN_H):
                s += hidden[j] * w2[j][k]
            outputs.append(tanh(s))

        return outputs  # [turn_strength, speed_mod, flee_strength]

    def mutate_copy(self, rate=NN_MUT_RATE, prob=NN_MUT_PROB):
        rg = random.gauss
        rr = random.random

        def mut_mat(m):
            return [[v + rg(0, rate) if rr() < prob else v for v in row] for row in m]
        def mut_vec(v):
            return [x + rg(0, rate) if rr() < prob else x for x in v]

        return NeuralBrain((
            mut_mat(self.w1), mut_vec(self.b1),
            mut_mat(self.w2), mut_vec(self.b2),
        ))


# =========================
# SPATIAL HASH
# =========================
class SpatialHash:
    __slots__ = ("cell", "w", "h", "grid")

    def __init__(self, cell_size, width, height):
        self.cell = cell_size
        self.w    = width
        self.h    = height
        self.grid = {}

    def _key(self, x, y):
        return (int(x) // self.cell, int(y) // self.cell)

    def clear(self):
        self.grid.clear()

    def insert(self, idx, x, y):
        k    = self._key(x, y)
        cell = self.grid.get(k)
        if cell is None:
            self.grid[k] = [idx]
        else:
            cell.append(idx)

    def query(self, x, y, radius):
        cx   = int(x) // self.cell
        cy   = int(y) // self.cell
        r    = int(math.ceil(radius / self.cell))
        out  = []
        grid = self.grid
        for gx in range(cx - r, cx + r + 1):
            for gy in range(cy - r, cy + r + 1):
                cell = grid.get((gx, gy))
                if cell:
                    out.extend(cell)
        return out


# =========================
# WORM
# =========================
class Worm:
    id_counter = 0

    def __init__(self, x=None, y=None, kind=None, brain=None):
        self.id    = Worm.id_counter
        Worm.id_counter += 1

        self.x     = x if x is not None else random.randint(0, WIDTH)
        self.y     = y if y is not None else random.randint(0, HEIGHT)
        self.angle = random.random() * 2 * math.pi

        self.kind  = kind if kind is not None else random.choice([NORMAL, CANNIBAL, SAFE])
        self.speed = 2.0 if self.kind != KING else 0.65
        self.size  = 4.0 if self.kind != KING else 6.0

        # Neuroevolved personality – every worm has its own brain weights
        self.brain = brain if brain is not None else NeuralBrain()

        self.hunger  = random.uniform(0.55, 1.0)
        self.fear    = 0.0
        self.boredom = random.uniform(0, 1)
        self.mercy   = random.uniform(0, 1) if self.kind == SAFE else 0.0

        self.repro_timer = (random.randint(500, 1000) if self.kind != KING
                            else random.randint(KING_REPRO_MIN, KING_REPRO_MAX))

        self.starving        = False
        self.starve_countdown = 0

        self.fog       = False
        self.fog_timer = 0

        self.frenzy           = False
        self.frenzy_timer     = 0
        self.frenzy_target_id = None

        self.splitting       = False
        self.split_countdown = 0

        self.dead           = False
        self.frame_counter  = 0
        self.last_state_log = 0

    def strength(self):
        return self.size + (1.0 - self.hunger)

    def snapshot(self):
        """
        Compact tuple for the threaded perceive pass.
        Layout: (id, x, y, kind, size, hunger, dead, starving, splitting,
                 frenzy, frenzy_target_id, strength)
        """
        return (
            self.id, self.x, self.y, self.kind, self.size, self.hunger,
            self.dead, self.starving, self.splitting,
            self.frenzy, self.frenzy_target_id, self.strength(),
        )

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
            self.x = r;            self.angle = math.pi - self.angle
        elif self.x > WIDTH - r:
            self.x = WIDTH - r;    self.angle = math.pi - self.angle
        if self.y < r:
            self.y = r;            self.angle = -self.angle
        elif self.y > HEIGHT - r:
            self.y = HEIGHT - r;   self.angle = -self.angle
        self.angle = wrap_angle(self.angle)

    # ─────────────────────────────────────────────────────────────────────────
    # PERCEIVE  (runs in thread pool – read-only, returns pure data)
    # ─────────────────────────────────────────────────────────────────────────
    def perceive(self, worm_snapshots, food_positions, worm_grid, food_grid, king_on):
        """
        Sense the world, build an 11-dim input vector, run the neural brain,
        and return a compact perception dict for update().

        Food targeting: strictly nearest single food item – no blending of
        multiple attractors, no priority weighting beyond distance.
        """
        x, y  = self.x, self.y
        kind  = self.kind
        hypot = math.hypot

        # ── Nearest food (one and only one) ───────────────────────────────────
        food_dx, food_dy, food_prox = 0.0, 0.0, 0.0
        best_food_d = 1e9

        for idx in food_grid.query(x, y, 240):
            fx, fy = food_positions[idx]
            d = hypot(fx - x, fy - y)
            if d < best_food_d:
                best_food_d = d
                food_dx     = fx - x
                food_dy     = fy - y

        if best_food_d < 240:
            inv_d      = 1.0 / max(1.0, best_food_d)
            food_dx   *= inv_d
            food_dy   *= inv_d
            food_prox  = 1.0 / (1.0 + best_food_d * 0.01)  # 1 = touching, ~0 = far
        else:
            food_dx = food_dy = food_prox = 0.0

        # ── Worm neighbourhood ────────────────────────────────────────────────
        threat_dx, threat_dy, threat_prox = 0.0, 0.0, 0.0
        ally_dx,   ally_dy                = 0.0, 0.0
        sep_x,     sep_y                  = 0.0, 0.0
        fear_acc    = 0.0
        best_threat_d = 1e9
        best_ally_d   = 1e9

        for idx in worm_grid.query(x, y, 120):
            ws = worm_snapshots[idx]
            # (id, x, y, kind, size, hunger, dead, starving, splitting,
            #  frenzy, frenzy_target_id, strength)
            if ws[0] == self.id or ws[6] or ws[7] or ws[8]:  # self / dead / starving / splitting
                continue

            dx = ws[1] - x
            dy = ws[2] - y
            d  = hypot(dx, dy)
            if d < 0.0001:
                continue

            ndx = dx / d
            ndy = dy / d

            # Separation: stay out of each other's personal space
            if d < 18:
                w     = (18 - d) / 18
                sep_x -= ndx * w
                sep_y -= ndy * w

            ws_kind = ws[3]

            # ── Threat classification ─────────────────────────────────────
            is_threat = False
            if kind == NORMAL:
                if ws_kind == CANNIBAL:
                    is_threat = True
                elif ws[9] and ws[10] == self.id:   # frenzy mob targeting me
                    is_threat = True
            elif kind == CANNIBAL:
                if ws_kind == SAFE and ws[11] >= self.strength():
                    is_threat = True
            elif kind == SAFE:
                if ws_kind == CANNIBAL and ws[11] >= self.strength():
                    is_threat = True

            if is_threat:
                fear_acc += 1.0 / (d + 1.0)
                if d < best_threat_d:
                    best_threat_d = d
                    threat_dx     = -ndx          # flee direction (away from threat)
                    threat_dy     = -ndy
                    threat_prox   = 1.0 / (1.0 + d * 0.02)
            elif ws_kind == kind:                  # nearest ally for flocking nudge
                if d < best_ally_d:
                    best_ally_d = d
                    ally_dx = ndx
                    ally_dy = ndy

            # King presence calms fear a bit
            if king_on and kind in (NORMAL, SAFE) and ws_kind == CANNIBAL:
                fear_acc *= 0.8

        fear_final = min(1.0, fear_acc * 0.85)

        # ── Build neural input vector (11 floats, all ∈ [-1, 1] roughly) ─────
        inp = [
            food_dx,            # 0  unit vector toward nearest food, x
            food_dy,            # 1  unit vector toward nearest food, y
            food_prox,          # 2  food proximity  (0 far → 1 close)
            threat_dx,          # 3  flee direction from nearest threat, x
            threat_dy,          # 4  flee direction from nearest threat, y
            threat_prox,        # 5  threat proximity
            ally_dx,            # 6  unit vector toward nearest same-kind, x
            ally_dy,            # 7  unit vector toward nearest same-kind, y
            self.hunger,        # 8  internal hunger  [0, 1]
            fear_final,         # 9  accumulated fear  [0, 1]
            self.boredom,       # 10 boredom           [0, 1]
        ]

        nn_turn, nn_speed, nn_flee = self.brain.forward(inp)

        return {
            "food_dx":    food_dx,    "food_dy":    food_dy,
            "threat_dx":  threat_dx,  "threat_dy":  threat_dy,
            "ally_dx":    ally_dx,    "ally_dy":    ally_dy,
            "sep_x":      sep_x,      "sep_y":      sep_y,
            "fear":       fear_final,
            "nn_turn":    nn_turn,    # [-1,1] food bias (pos) vs threat bias (neg)
            "nn_speed":   nn_speed,   # [-1,1] speed modifier
            "nn_flee":    nn_flee,    # [-1,1] flee weight amplifier
        }

    # ─────────────────────────────────────────────────────────────────────────
    # STATE MACHINES
    # ─────────────────────────────────────────────────────────────────────────
    def maybe_enter_fog(self):
        if (not self.fog) and random.random() < 0.002:
            self.fog       = True
            self.fog_timer = random.randint(20, 80)
            log(self.id, "entered mental fog")

    def maybe_start_frenzy(self, worms):
        alive_count = sum(1 for w in worms if not w.dead)
        if self.kind != NORMAL or self.frenzy or alive_count <= FRENZY_POP_THRESHOLD:
            return
        if random.random() < FRENZY_CHANCE:
            self.frenzy       = True
            self.frenzy_timer = FRENZY_DURATION
            candidates        = [w for w in worms if w is not self and not w.dead
                                  and not w.starving and not w.splitting]
            self.frenzy_target_id = random.choice(candidates).id if candidates else None
            play("frenzy")
            log(self.id, f"FRENZY started target={self.frenzy_target_id}")

    def spread_frenzy(self, worms):
        if not self.frenzy:
            return
        target = self.get_target_by_id(worms, self.frenzy_target_id)
        if target is None:
            self.frenzy = False;  self.frenzy_target_id = None;  return
        for w in worms:
            if w.dead or w.kind != NORMAL or w.frenzy or w.starving or w.splitting:
                continue
            d = math.hypot(w.x - self.x, w.y - self.y)
            if d < FRENZY_RADIUS and random.random() < 0.08:
                w.frenzy           = True
                w.frenzy_timer     = FRENZY_DURATION
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
        self.splitting       = True
        self.split_countdown = 40
        log(self.id, "starting split")

    def spawn_split_children(self):
        if   self.kind == NORMAL:   pool = [SAFE, CANNIBAL]
        elif self.kind == CANNIBAL: pool = [NORMAL, SAFE]
        elif self.kind == SAFE:     pool = [NORMAL, CANNIBAL]
        else:                       pool = [NORMAL]

        if random.random() < KING_MUTATION_CHANCE:
            child_kinds = [KING, random.choice(pool)]
        else:
            child_kinds = [random.choice(pool), random.choice(pool)]

        children = []
        for idx, (ox, oy) in enumerate([(-10, -6), (10, 6)]):
            cx    = clamp(self.x + ox, 0, WIDTH)
            cy    = clamp(self.y + oy, 0, HEIGHT)
            child = Worm(cx, cy, child_kinds[idx], brain=self.brain.mutate_copy())
            child.angle       = self.angle + random.uniform(-0.6, 0.6)
            child.hunger      = random.uniform(0.55, 1.0)
            child.repro_timer = (random.randint(500, 1000) if child.kind != KING
                                 else random.randint(KING_REPRO_MIN, KING_REPRO_MAX))
            child.mercy       = random.uniform(0, 1) if child.kind == SAFE else 0.0
            if child.kind == KING:
                child.speed = 0.65;  child.size = 6.0
            children.append(child)
        return children

    def spawn_king_child(self):
        child_kind = NORMAL if random.random() < KING_BIRTH_NORMAL_CHANCE else CANNIBAL
        child         = Worm(self.x, self.y, child_kind, brain=self.brain.mutate_copy())
        child.x       = clamp(self.x + random.uniform(-6, 6), 0, WIDTH)
        child.y       = clamp(self.y + random.uniform(-6, 6), 0, HEIGHT)
        child.angle   = self.angle + random.uniform(-0.5, 0.5)
        child.hunger  = random.uniform(0.6, 1.0)
        child.repro_timer = random.randint(500, 1000)
        return child

    def tick_internal_state(self):
        self.frame_counter += 1
        self.hunger  = max(0.0, self.hunger - (0.0007 if self.kind == KING else 0.0010))
        self.boredom = min(1.0, self.boredom + 0.002)
        if self.hunger <= 0.0 and not self.starving and not self.splitting:
            self.starving         = True
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
        self.x    += random.uniform(-1.6, 1.6)
        self.y    += random.uniform(-1.6, 1.6)
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
            self.frenzy = False;  self.frenzy_target_id = None;  return

        if self.id == self.frenzy_target_id:
            mobs = [w for w in worms
                    if w.kind == NORMAL and w.frenzy and not w.dead and not w.starving
                    and not w.splitting and w.frenzy_target_id == self.id]
            if mobs:
                vx = self.x - sum(w.x for w in mobs) / len(mobs)
                vy = self.y - sum(w.y for w in mobs) / len(mobs)
                d  = max(1.0, math.hypot(vx, vy))
                self.angle = math.atan2(vy, vx)
                self.x += (vx / d) * 2.5
                self.y += (vy / d) * 2.5
                self.apply_boundary()
        else:
            dx  = target.x - self.x
            dy  = target.y - self.y
            ang = math.atan2(dy, dx)
            self.angle += wrap_angle(ang - self.angle) * 0.25
            self.x += math.cos(self.angle) * 3.0
            self.y += math.sin(self.angle) * 3.0
            self.apply_boundary()

        if target and not target.dead:
            near = [w for w in worms
                    if w.kind == NORMAL and w.frenzy and not w.dead and not w.starving
                    and not w.splitting and w.frenzy_target_id == target.id
                    and math.hypot(w.x - target.x, w.y - target.y) < 15]
            if len(near) >= FRENZY_MIN_GROUP:
                target.dead = True
                play("pop")
                log(self.id, f"group killed {target.id}")
                self.frenzy = False;  self.frenzy_target_id = None;  return

        if self.frenzy_timer <= 0:
            self.frenzy = False;  self.frenzy_target_id = None
            log(self.id, "calmed down from frenzy")

    # ─────────────────────────────────────────────────────────────────────────
    # MOVEMENT  –  driven entirely by neural outputs
    # ─────────────────────────────────────────────────────────────────────────
    def move_from_perception(self, p, king_on):
        """
        nn_turn  ∈ [-1, 1]  – positive → lean toward food; negative → lean away
        nn_speed ∈ [-1, 1]  – additive speed tweak
        nn_flee  ∈ [-1, 1]  – positive → amplify flee; negative → suppress it

        Desire vector = weighted sum of food, threat-flee, separation, ally-nudge.
        The weights come from the network, so each worm develops its own style
        (bold sprinters, cautious snipers, social clusterers, etc.).
        """
        food_dx  = p["food_dx"];   food_dy  = p["food_dy"]
        threat_dx = p["threat_dx"]; threat_dy = p["threat_dy"]
        ally_dx  = p["ally_dx"];   ally_dy  = p["ally_dy"]
        sep_x    = p["sep_x"];     sep_y    = p["sep_y"]
        nn_turn  = p["nn_turn"]    # [-1, 1]
        nn_speed = p["nn_speed"]   # [-1, 1]
        nn_flee  = p["nn_flee"]    # [-1, 1]

        # Neural weights: map [-1,1] to a positive scale around 1.0
        food_w  = 1.0 + nn_turn * 0.8    # [0.2, 1.8]
        flee_w  = 1.0 + nn_flee * 0.8    # [0.2, 1.8]

        desire_x = food_dx * food_w + threat_dx * flee_w + sep_x * 0.8 + ally_dx * 0.2
        desire_y = food_dy * food_w + threat_dy * flee_w + sep_y * 0.8 + ally_dy * 0.2

        # King presence makes normals/safes a bit bolder toward food
        if king_on and self.kind in (NORMAL, SAFE):
            desire_x += food_dx * 0.5
            desire_y += food_dy * 0.5

        if desire_x == 0.0 and desire_y == 0.0:
            desire_x = math.cos(self.angle)
            desire_y = math.sin(self.angle)

        target_angle = math.atan2(desire_y, desire_x)

        # Turn responsiveness is also modulated by neural output magnitude
        turn_rate = 0.12 + abs(nn_turn) * 0.12   # [0.12, 0.24]
        turn      = wrap_angle(target_angle - self.angle)
        self.angle += turn * turn_rate + random.uniform(-0.05, 0.05)

        # Speed: base + hunger/boredom drive + neural modifier
        drive      = activate(self.hunger * 0.7 + self.boredom * 0.2) * 0.6
        move_speed = self.speed + drive + nn_speed * 0.4 + random.uniform(-0.15, 0.25)
        move_speed = max(0.3, move_speed)

        self.x += math.cos(self.angle) * move_speed
        self.y += math.sin(self.angle) * move_speed
        self.apply_boundary()

    def resolve_food_and_bites(self, food, worms):
        for i in range(len(food) - 1, -1, -1):
            fx, fy = food[i]
            if math.hypot(self.x - fx, self.y - fy) < 7:
                food.pop(i)
                self.hunger  = min(1.0, self.hunger + 0.45)
                self.boredom = max(0.0, self.boredom - 0.15)  # eating resets boredom a bit
                play("eat")
                break

        for w in worms:
            if w.dead or w is self or w.starving or w.splitting:
                continue
            d = math.hypot(self.x - w.x, self.y - w.y)
            if d >= 8:
                continue
            if self.kind == CANNIBAL and self.can_eat_target(w):
                w.dead = True;  self.hunger = min(1.0, self.hunger + 0.35)
                play("eat");    log(self.id, f"ate {w.id}");  break
            if self.kind == SAFE and self.can_eat_target(w):
                w.dead = True;  self.hunger = min(1.0, self.hunger + 0.35)
                play("eat");    log(self.id, f"ate cannibal {w.id}");  break
            if self.kind == NORMAL and w.kind == CANNIBAL:
                self.angle = math.atan2(self.y - w.y, self.x - w.x)

    def update(self, perception, worms, food, newborns, king_on):
        if self.dead:
            return

        self.tick_internal_state()

        if self.starving:   self.resolve_starvation(); return
        if self.fog:        self.resolve_fog();         return
        if self.kind == KING:
            self.resolve_king(food, newborns);          return
        if self.splitting:
            self.resolve_splitting(newborns);           return

        if self.kind == NORMAL:
            if not self.frenzy:
                self.maybe_start_frenzy(worms)
                self.spread_frenzy(worms)
            if self.frenzy:
                self.resolve_frenzy(worms);             return

        self.fear = perception["fear"]

        if self.boredom > 0.9 and random.random() < 0.01:
            play("bored")

        self.move_from_perception(perception, king_on)
        self.resolve_food_and_bites(food, worms)

        self.repro_timer -= 1
        if self.repro_timer <= 0 and self.hunger > 0.72:
            self.start_split()

        if self.frame_counter - self.last_state_log >= LOG_EVERY_FRAMES:
            self.last_state_log = self.frame_counter
            log(self.id, f"[{kind_name(self.kind)}] H:{self.hunger:.2f} F:{self.fear:.2f} B:{self.boredom:.2f}")

    def draw(self, surf):
        color  = {NORMAL: (255,255,255), CANNIBAL: (255,0,0), SAFE: (255,255,0), KING: (0,255,255)}.get(self.kind, (255,255,255))
        radius = max(2, int(self.size))
        pygame.draw.circle(surf, color, (int(self.x), int(self.y)), radius)
        if self.frenzy and not self.dead:
            pygame.draw.circle(surf, (255, 120, 0), (int(self.x), int(self.y)), radius + 3, 1)


# =========================
# WORLD LOOP
# =========================
worms = [Worm() for _ in range(START_WORM_COUNT)]
food  = [(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(START_FOOD_COUNT)]

# Scale thread pool to CPU count (perception pass is read-only, safe to parallelize)
_cpu       = os.cpu_count() or 1
max_workers = max(2, min(_cpu, 8))
executor   = ThreadPoolExecutor(max_workers=max_workers)

running = True
while running:
    screen.fill((0, 0, 0))

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    worms = [w for w in worms if not w.dead]

    # Rebuild spatial indices every frame
    worm_grid = SpatialHash(CELL_SIZE, WIDTH, HEIGHT)
    food_grid = SpatialHash(CELL_SIZE, WIDTH, HEIGHT)

    worm_snaps = []
    for idx, w in enumerate(worms):
        worm_grid.insert(idx, w.x, w.y)
        worm_snaps.append(w.snapshot())

    for idx, (fx, fy) in enumerate(food):
        food_grid.insert(idx, fx, fy)

    king_on = any(w.kind == KING and not w.dead for w in worms)

    # ── Threaded sensory + neural forward pass ────────────────────────────────
    futures    = [
        executor.submit(worms[i].perceive, worm_snaps, food, worm_grid, food_grid, king_on)
        for i in range(len(worms))
    ]
    perceptions = [f.result() for f in futures]

    # ── Sequential state update (shared mutable state: food list, newborns) ───
    newborns = []
    for w, p in zip(worms, perceptions):
        if not w.dead:
            w.update(p, worms, food, newborns, king_on)

    worms.extend(newborns)
    worms = [w for w in worms if not w.dead]

    # ── Draw ──────────────────────────────────────────────────────────────────
    for fx, fy in food:
        pygame.draw.circle(screen, (0, 255, 0), (fx, fy), 3)
    for w in worms:
        w.draw(screen)

    pygame.display.flip()
    clock.tick(60)

executor.shutdown(wait=False, cancel_futures=True)
pygame.quit()
