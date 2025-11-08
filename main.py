"""Bot automation module for human-like browser interactions."""
import math
import random
from time import sleep
from typing import Any, Dict, List, Tuple, Union

# --- Third-Party
import numpy as np
import pytweening
import scipy.stats
from selenium.common.exceptions import MoveTargetOutOfBoundsException
from selenium.webdriver import Chrome, Edge, Firefox, Safari
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.actions.action_builder import ActionBuilder

class Player:  # pylint: disable=too-many-instance-attributes
    """Represents a player in the PUBG simulation."""
    def __init__(self, name):
        self.name = name
        self.hp = 100
        self.x = random.randint(0, 100)
        self.y = random.randint(0, 100)
        self.alive = True
        self.weapon_power = random.randint(10, 35)   # base damage
        self.range = random.randint(15, 50)          # attack distance
        self.accuracy = random.uniform(0.4, 0.9)     # hit chance

    def move(self):
        """Move randomly within map boundaries."""
        if not self.alive:
            return
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        self.x = max(0, min(100, self.x + dx))
        self.y = max(0, min(100, self.y + dy))

    def distance(self, other):
        """Calculate distance to another player."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def shoot(self, other):
        """Attempt to shoot another player."""
        if not (self.alive and other.alive):
            return
        d = self.distance(other)
        if d <= self.range and random.random() < self.accuracy:
            damage = random.randint(self.weapon_power // 2, self.weapon_power)
            other.hp -= damage
            print(f"{self.name} hits {other.name} for {damage} dmg! ({other.hp:.1f} HP left)")
            if other.hp <= 0:
                other.alive = False
                print(f"💀 {other.name} has been eliminated by {self.name}!")

# -------------------------------
# Game setup and simulation loop
# -------------------------------

def simulate_pubg(num_players=10):
    """Simulate a PUBG match with the specified number of players."""
    players = [Player(f"Player{i}") for i in range(1, num_players + 1)]
    tick = 0

    print(f"🎮 Starting PUBG Simulation with {num_players} players!\n")

    while sum(p.alive for p in players) > 1:
        tick += 1
        alive_players = [p for p in players if p.alive]

        for p in alive_players:
            p.move()
            enemies = [e for e in alive_players if e != p and e.alive]
            if enemies:
                target = random.choice(enemies)
                p.shoot(target)

        # optional: shrink safe zone (simulated as global damage)
        if tick % 20 == 0:
            for p in players:
                if p.alive and random.random() < 0.3:
                    p.hp -= 10
                    print(f"⚠️ {p.name} took storm damage! ({p.hp:.1f} HP left)")
                    if p.hp <= 0:
                        p.alive = False
                        print(f"💀 {p.name} was lost to the storm!")

        if tick > 500:
            print("⏰ Match timeout!")
            break

    winner = [p for p in players if p.alive]
    if winner:
        print(f"\n🏆 Winner: {winner[0].name} after {tick} ticks!")
    else:
        print("\nNo survivors — the storm claimed everyone!")
class MapZone:
    """Utility for geometry conversions and randomized curve parameters."""

    @staticmethod
    def calculate_drop_zone(target: WebElement, relative_position: List[float]) -> List[int]:
        """Convert relative offsets (0..1) to absolute pixel offsets within an element box."""
        box = target.size
        w, h = box["width"], box["height"]
        return [int(w * relative_position[0]), int(h * relative_position[1])]

    @staticmethod
    def calculate_recoil_pattern(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
        game_client: WebDriver,
        spawn_point: List[float],
        destination: List[float],
        steady_aim: bool,
        map_width: int,
        map_height: int,
        _tweening_method=None,
    ):
        """Choose curve parameters for mouse movement paths."""
        is_in_game = isinstance(game_client, (Chrome, Firefox, Edge, Safari))
        if is_in_game:
            ww, wh = game_client.get_window_size().values()
        else:
            ww, wh = map_width, map_height

        # soft-safe region for start/end sanity
        x_min, x_max = ww * 0.15, ww * 0.85
        y_min, y_max = wh * 0.15, wh * 0.85

        # Enhanced movement strategies with different patterns
        movement_strategies = [
            {"ranges": [range(15, 35), range(35, 60), range(60, 90)],
             "weights": [0.3, 0.5, 0.2]},  # Precise
            {"ranges": [range(25, 50), range(50, 80), range(80, 120)],
             "weights": [0.2, 0.4, 0.4]},  # Natural
            {"ranges": [range(30, 55), range(55, 85), range(85, 110)],
             "weights": [0.1, 0.3, 0.6]},  # Erratic
        ]
        strategy = random.choice(movement_strategies)
        chosen_range = random.choices(strategy["ranges"], strategy["weights"])[0]
        horizontal_recoil = random.choice(chosen_range)
        vertical_recoil = random.choice(chosen_range)

        # if points are near edges, reduce fancy offsets/knots
        waypoints = 2
        sx, sy = spawn_point
        ex, ey = destination
        if not (x_min <= sx <= x_max and y_min <= sy <= y_max):
            horizontal_recoil = vertical_recoil = 1
            waypoints = 1
        if not (x_min <= ex <= x_max and y_min <= ey <= y_max):
            horizontal_recoil = vertical_recoil = 1
            waypoints = 1

        # Enhanced tweening options for more natural movement
        tween_options = [
            pytweening.easeInOutCubic,
            pytweening.easeInOutQuad,
            pytweening.easeInOutQuart,
            pytweening.easeInOutSine,
            pytweening.easeInOutExpo,
        ]
        movement_curve = random.choice(tween_options)

        # make long/straight-ish paths simpler
        is_linear = bool(steady_aim and (abs(sx - ex) < 10 or abs(sy - ey) < 10))
        waypoints = 3 if is_linear else 2

        # Enhanced distortion parameters for more realistic movement
        distortion_profiles = [
            {"mu_range": range(75, 95), "sigma_range": range(80, 100),
             "freq_range": range(20, 50)},  # Subtle
            {"mu_range": range(85, 105), "sigma_range": range(90, 110),
             "freq_range": range(30, 60)},  # Moderate
            {"mu_range": range(95, 115), "sigma_range": range(100, 120),
             "freq_range": range(40, 70)},  # Pronounced
        ]
        profile = random.choice(distortion_profiles)
        recoil_mean = random.choice(profile["mu_range"]) / 100.0
        recoil_spread = random.choice(profile["sigma_range"]) / 100.0
        fire_rate = random.choice(profile["freq_range"]) / 100.0

        # pick target sample density based on distance
        dist = math.hypot(ex - sx, ey - sy)

        def _largest_den(n: int) -> int:
            d = max(1, n // 2)
            while d > 0:
                if n / d > 2:
                    return d
                d -= 1
            return 1

        px = 13
        if dist // px < 2:
            px = _largest_den(int(max(2, dist)))
        bullet_count = int(max(2, dist // px))

        return (int(horizontal_recoil), int(vertical_recoil), waypoints, recoil_mean,
                recoil_spread, fire_rate, movement_curve, bullet_count, is_in_game)

class WeaponRecoil:  # pylint: disable=too-few-public-methods
    """Utility class for generating bezier curve points."""

    @staticmethod
    def _ncr(n: int, k: int) -> float:
        return math.factorial(n) / float(math.factorial(k) * math.factorial(n - k))

    @staticmethod
    def _bern_term(t: float, i: int, n: int) -> float:
        return WeaponRecoil._ncr(n, i) * (t ** i) * ((1 - t) ** (n - i))

    @staticmethod
    def generate_bullet_path(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        n: int, anchors: List[Tuple[float, float]]
    ) -> List[Tuple[int, int]]:
        """Generate bezier curve points with human-like jitter."""
        if not anchors or len(anchors) < 2:
            return []

        p0 = anchors[0]
        p3 = anchors[-1]
        pts: List[Tuple[int, int]] = [p0]

        if n <= 2:
            return [p0, p3]

        # bounding box to clamp jitter
        x_lo = min(p0[0], p3[0]) - 50
        x_hi = max(p0[0], p3[0]) + 50
        y_lo = min(p0[1], p3[1]) - 50
        y_hi = max(p0[1], p3[1]) + 50

        # pick two inner control points
        dist = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
        max_off = min(dist * 0.2, 40)

        r1 = random.uniform(0.1, 0.4)
        c1x = p0[0] + (p3[0] - p0[0]) * r1
        c1y = p0[1] + (p3[1] - p0[1]) * r1
        if random.random() < 0.5:
            c1x += random.gauss(0, max_off / 2)
            c1y += random.gauss(0, max_off / 2)
        else:
            c1x += random.uniform(-max_off, max_off)
            c1y += random.uniform(-max_off, max_off)
        c1x, c1y = max(x_lo, min(x_hi, c1x)), max(y_lo, min(y_hi, c1y))

        r2 = random.uniform(0.6, 0.9)
        c2x = p0[0] + (p3[0] - p0[0]) * r2
        c2y = p0[1] + (p3[1] - p0[1]) * r2
        if random.random() < 0.5:
            c2x += random.gauss(0, max_off / 2)
            c2y += random.gauss(0, max_off / 2)
        else:
            c2x += random.uniform(-max_off, max_off)
            c2y += random.uniform(-max_off, max_off)
        c2x, c2y = max(x_lo, min(x_hi, c2x)), max(y_lo, min(y_hi, c2y))

        p1 = (c1x, c1y)
        p2 = (c2x, c2y)

        samples = random.randint(20, 50)
        for i in range(1, samples - 1):
            base_t = i / (samples - 1)
            t = max(0.0, min(1.0, base_t + random.uniform(-0.02, 0.02)))
            x = ((1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] +
                 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0])
            y = ((1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] +
                 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1])

            # micro jitter & believable imperfections
            jitter = random.uniform(0.5, 1.2)
            x += random.gauss(0, jitter)
            y += random.gauss(0, jitter)

            if random.random() < 0.15:
                x += random.uniform(-1.5, 1.5)
                y += random.uniform(-1.5, 1.5)

            x = max(x_lo, min(x_hi, x))
            y = max(y_lo, min(y_hi, y))

            # little pauses & stutters
            if random.random() < 0.12:
                for _ in range(random.randint(1, 4)):
                    pts.append((
                        int(max(x_lo, min(x_hi, x + random.gauss(0, 0.8)))),
                        int(max(y_lo, min(y_hi, y + random.gauss(0, 0.8))))
                    ))

            if random.random() < 0.10:
                ox = max(x_lo, min(x_hi, x + random.uniform(-3.0, 3.0)))
                oy = max(y_lo, min(y_hi, y + random.uniform(-3.0, 3.0)))
                pts.append((int(ox), int(oy)))
                cx = max(x_lo, min(x_hi, x + random.uniform(-1.5, 1.5)))
                cy = max(y_lo, min(y_hi, y + random.uniform(-1.5, 1.5)))
                pts.append((int(cx), int(cy)))

            if random.random() < 0.05:
                for _ in range(random.randint(2, 5)):
                    pts.append((
                        int(max(x_lo, min(x_hi, x + random.gauss(0, 0.3)))),
                        int(max(y_lo, min(y_hi, y + random.gauss(0, 0.3))))
                    ))

            pts.append((int(x), int(y)))

        pts.append(p3)
        return pts

class MovementPath:  # pylint: disable=too-few-public-methods
    """Builds a 'human' path between two points with boundaries, jitter and tweening."""

    def __init__(self, src_xy: List[int], dst_xy: List[int], **kw):
        self._src = src_xy
        self._dst = dst_xy
        self.curve_kind = kw.get("curve_method", "bezier")
        self.points = self._plan_route(**kw)

    # Internal helpers -------------------------------------------------

    @staticmethod
    def _is_number(x: Any) -> bool:
        return isinstance(x, (float, int, np.int32, np.int64, np.float32, np.float64))

    def _is_point_list(self, pts: Any) -> bool:
        if not isinstance(pts, list):
            return False
        try:
            return all(len(p) == 2 and self._is_number(p[0]) and self._is_number(p[1]) for p in pts)
        except (TypeError, IndexError, KeyError):
            return False

    # Curve steps ------------------------------------------------------

    def _calculate_waypoints(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, left: int, right: int, down: int, up: int, k: int
    ) -> List[Tuple[int, int]]:
        if not all(self._is_number(v) for v in (left, right, down, up)):
            raise ValueError("Boundaries must be numeric")
        if not isinstance(k, int) or k < 0:
            k = 0
        if left > right:
            raise ValueError("left_boundary must be <= right_boundary")
        if down > up:
            raise ValueError("down_boundary must be <= upper_boundary")

        try:
            xs = np.random.choice(range(left, right) or left, size=k)
            ys = np.random.choice(range(down, up) or down, size=k)
        except TypeError:
            xs = np.random.choice(range(int(left), int(right)), size=k)
            ys = np.random.choice(range(int(down), int(up)), size=k)
        return list(zip(xs, ys))

    def _generate_movement_path(
        self, waypoints: List[Tuple[int, int]], _target_points: int
    ) -> List[Tuple[int, int]]:
        if not self._is_point_list(waypoints):
            raise ValueError("waypoints must be valid list of points")
        span = max(abs(self._src[0] - self._dst[0]), abs(self._src[1] - self._dst[1]), 2)
        anchors = [self._src] + waypoints + [self._dst]
        return WeaponRecoil.generate_bullet_path(int(span), anchors)

    def _add_recoil_jitter(
        self, pts: List[Tuple[float, float]], recoil_mean: float,
        recoil_spread: float, fire_rate: float
    ) -> List[Tuple[float, float]]:
        if not (self._is_number(recoil_mean) and self._is_number(recoil_spread) and
                self._is_number(fire_rate)):
            raise ValueError("Distortions must be numeric")
        if not self._is_point_list(pts):
            raise ValueError("points must be valid list of points")
        if not 0 <= fire_rate <= 1:
            raise ValueError("distortion_frequency must be in range [0,1]")

        out: List[Tuple[float, float]] = [pts[0]]
        for (x, y) in pts[1:-1]:
            if random.random() < random.uniform(0.2, 0.5):
                rng = random.uniform(0.3, recoil_spread * 0.8)
                dx = random.gauss(0, rng / 2)
                dy = random.gauss(0, rng / 2)
            else:
                dx = dy = 0.0

            if random.random() < random.uniform(0.05, 0.15):
                cr = random.uniform(0.5, 1.5)
                dx += random.gauss(0, cr / 3)
                dy += random.gauss(0, cr / 3)

            out.append((x + dx, y + dy))

        out.append(pts[-1])
        return out

        # no explicit long-path case; lists are small here

    def _resample_waypoints(
        self, pts: List[Tuple[float, float]], movement_curve, target_points: int
    ):
        if not self._is_point_list(pts):
            raise ValueError("List of points not valid")
        if not isinstance(target_points, int) or target_points < 2:
            raise ValueError("target_points must be >= 2")

        res: List[Tuple[int, int]] = []
        length = len(pts) - 1
        for i in range(target_points):
            idx = int(movement_curve(i / (target_points - 1)) * length)
            res.append((int(pts[idx][0]), int(pts[idx][1])))
        return res

    # Public builder ---------------------------------------------------

    def _plan_route(self, **kw) -> List[Tuple[int, int]]:  # pylint: disable=too-many-locals
        kind = kw.get("curve_method", "bezier")
        offx = kw.get("offset_boundary_x", 80)
        offy = kw.get("offset_boundary_y", 80)

        left = kw.get("left_boundary", min(self._src[0], self._dst[0])) - offx
        right = kw.get("right_boundary", max(self._src[0], self._dst[0])) + offx
        down = kw.get("down_boundary", min(self._src[1], self._dst[1])) - offy
        up = kw.get("up_boundary", max(self._src[1], self._dst[1])) + offy

        kcount = kw.get("knots_count", 2)
        recoil_mean = kw.get("distortion_mean", 1.0)
        recoil_spread = kw.get("distortion_st_dev", 1.0)
        fire_rate = kw.get("distortion_frequency", 0.5)
        movement_curve = kw.get("tween", pytweening.easeOutQuad)
        bullet_count = kw.get("target_points", 100)

        if kind == "bspline" and kcount <= 2:
            kcount = 3

        waypoints = self._calculate_waypoints(left, right, down, up, kcount)
        pts = self._generate_movement_path(waypoints, bullet_count)
        pts = self._add_recoil_jitter(pts, recoil_mean, recoil_spread, fire_rate)
        pts = self._resample_waypoints(pts, movement_curve, bullet_count)

        return [(int(x), int(y)) for (x, y) in pts]

class WeaponDelay:  # pylint: disable=too-many-instance-attributes
    """Advanced latency model with behavioral correction patterns."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        interkey_mean=180, interkey_std=70, interkey_min=50, interkey_max=500,
        hold_mean=80, hold_std=25, hold_min=30, hold_max=200,
        backspace_probability=0.1, correction_delay=500,
    ):
        self.interkey_min = interkey_min
        self.interkey_max = interkey_max
        self.hold_min = hold_min
        self.hold_max = hold_max
        self.interkey_mean = interkey_mean
        self.interkey_std = interkey_std
        self.hold_mean = hold_mean
        self.hold_std = hold_std
        self.backspace_probability = backspace_probability
        self.correction_delay = correction_delay
        self._reload_weapon()

    def _calculate_weapon_stats(self, lo: float, hi: float, mean: float, std: float):
        scale = hi - lo
        loc = lo
        mu = (mean - lo) / scale
        var = (std / scale) ** 2
        t = mu / (1 - mu)
        beta = (t / var - t * t - 2 * t - 1) / (t ** 3 + 3 * t * t + 3 * t + 1)
        alpha = beta * t
        if alpha <= 0 or beta <= 0:
            raise ValueError("Invalid params for bounded beta")
        return scipy.stats.beta(alpha, beta, scale=scale, loc=loc)

    def _reload_weapon(self):
        self.interkey_dist = self._calculate_weapon_stats(
            self.interkey_min, self.interkey_max, self.interkey_mean, self.interkey_std)
        self.hold_dist = self._calculate_weapon_stats(
            self.hold_min, self.hold_max, self.hold_mean, self.hold_std)

    def adjust_weapon(self, **kw):
        """Tune latency model parameters."""
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self._reload_weapon()

    def get_fire_rate(self, n_chars: int):
        """Sample inter-key and hold times for given number of characters."""
        inter = self.interkey_dist.rvs(size=n_chars - 1)
        hold = self.hold_dist.rvs(size=n_chars)
        return inter, hold


class PlayerProfile:  # pylint: disable=too-few-public-methods
    """High-level typing profile facade around LatencyModel with preset profiles."""

    def __init__(self, name: str = "defined"):
        self.name = name
        self.model = WeaponDelay()

    def _calculate_shot_timings(self, ammo_count: str) -> Dict[str, List[int]]:
        inter, hold = self.model.get_fire_rate(len(ammo_count))
        inter = [int(x) for x in inter]
        hold = [int(x) for x in hold]

        # debug prints preserved (original behavior had prints)
        _ = np.std(inter)
        _ = np.std(hold)

        for i in range(1, len(inter)):
            if inter[i] < 0 and abs(inter[i]) > hold[i]:
                # retain the original "print" style side-effect for parity
                print("Warning: Inter-key delay exceeds previous key hold duration:", inter[i])
                print("Previous key hold duration was:", hold[i])

        return {"interkey_latencies": inter, "hold_times": hold}

    def generate_shooting_events(self, ammo_count: str) -> List[Dict[str, Union[str, float, int]]]:
        """Generate typing events with human-like timing patterns."""
        base = [
            {"timestamp": 0.0, "key_index": 1, "action": "press"},
            {"timestamp": 95.0, "key_index": 2, "action": "press"},
            {"timestamp": 118.0, "key_index": 1, "action": "release"},
            {"timestamp": 245.0, "key_index": 3, "action": "press"},
            {"timestamp": 275.0, "key_index": 2, "action": "release"},
            {"timestamp": 345.0, "key_index": 4, "action": "press"},
            {"timestamp": 398.0, "key_index": 3, "action": "release"},
            {"timestamp": 455.0, "key_index": 5, "action": "press"},
            {"timestamp": 525.0, "key_index": 4, "action": "release"},
            {"timestamp": 615.0, "key_index": 6, "action": "press"},
            {"timestamp": 630.0, "key_index": 5, "action": "release"},
            {"timestamp": 720.0, "key_index": 6, "action": "release"},
            {"timestamp": 735.0, "key_index": 7, "action": "press"},
            {"timestamp": 810.0, "key_index": 8, "action": "press"},
            {"timestamp": 875.0, "key_index": 7, "action": "release"},
            {"timestamp": 960.0, "key_index": 8, "action": "release"},
            {"timestamp": 1580.0, "key_index": 9, "action": "press"},
            {"timestamp": 1705.0, "key_index": 10, "action": "press"},
            {"timestamp": 1810.0, "key_index": 9, "action": "release"},
            {"timestamp": 1835.0, "key_index": 10, "action": "release"},
            {"timestamp": 2105.0, "key_index": 11, "action": "press"},
            {"timestamp": 2215.0, "key_index": 11, "action": "release"},
            {"timestamp": 2365.0, "key_index": 12, "action": "press"},
            {"timestamp": 2410.0, "key_index": 12, "action": "release"},
            {"timestamp": 2550.0, "key_index": 13, "action": "press"},
            {"timestamp": 2580.0, "key_index": 13, "action": "release"},
            {"timestamp": 3815.0, "key_index": 14, "action": "press"},
            {"timestamp": 3960.0, "key_index": 14, "action": "release"},
            {"timestamp": 4545.0, "key_index": 15, "action": "press"},
            {"timestamp": 4665.0, "key_index": 15, "action": "release"},
            {"timestamp": 4725.0, "key_index": 16, "action": "press"},
            {"timestamp": 4800.0, "key_index": 16, "action": "release"},
            {"timestamp": 4825.0, "key_index": 17, "action": "press"},
            {"timestamp": 4915.0, "key_index": 18, "action": "press"},
            {"timestamp": 4990.0, "key_index": 17, "action": "release"},
            {"timestamp": 5020.0, "key_index": 19, "action": "press"},
            {"timestamp": 5045.0, "key_index": 18, "action": "release"},
            {"timestamp": 5155.0, "key_index": 20, "action": "press"},
            {"timestamp": 5215.0, "key_index": 21, "action": "press"},
            {"timestamp": 5260.0, "key_index": 19, "action": "release"},
            {"timestamp": 5305.0, "key_index": 20, "action": "release"},
            {"timestamp": 5320.0, "key_index": 22, "action": "press"},
            {"timestamp": 5340.0, "key_index": 21, "action": "release"},
            {"timestamp": 5440.0, "key_index": 22, "action": "release"},
            {"timestamp": 5475.0, "key_index": 23, "action": "press"},
            {"timestamp": 5560.0, "key_index": 23, "action": "release"},
            {"timestamp": 5675.0, "key_index": 24, "action": "press"},
            {"timestamp": 5735.0, "key_index": 24, "action": "release"},
            {"timestamp": 5765.0, "key_index": 25, "action": "press"},
            {"timestamp": 5830.0, "key_index": 25, "action": "release"},
            {"timestamp": 6325.0, "key_index": 26, "action": "press"},
            {"timestamp": 6435.0, "key_index": 27, "action": "press"},
            {"timestamp": 6550.0, "key_index": 26, "action": "release"},
            {"timestamp": 6575.0, "key_index": 28, "action": "press"},
            {"timestamp": 6600.0, "key_index": 27, "action": "release"},
            {"timestamp": 6700.0, "key_index": 28, "action": "release"},
            {"timestamp": 6840.0, "key_index": 29, "action": "press"},
            {"timestamp": 6950.0, "key_index": 29, "action": "release"},
            {"timestamp": 7075.0, "key_index": 30, "action": "press"},
            {"timestamp": 7150.0, "key_index": 31, "action": "press"},
            {"timestamp": 7230.0, "key_index": 32, "action": "press"},
            {"timestamp": 7240.0, "key_index": 30, "action": "release"},
            {"timestamp": 7245.0, "key_index": 31, "action": "release"},
            {"timestamp": 7325.0, "key_index": 32, "action": "release"},
        ]
        # tile the base pattern across the input string
        groups: Dict[int, List[Dict[str, Union[int, float, str]]]] = {}
        for e in base:
            groups.setdefault(e["key_index"], []).append(e)
        out: List[Dict[str, Union[str, float, int]]] = []
        for i, char in enumerate(ammo_count):
            idx = (i % 32) + 1
            for e in groups.get(idx, []):
                c = dict(e)
                c["char"] = char
                c["type"] = "keydown" if e["action"] == "press" else "keyup"
                c["timing"] = e["timestamp"]
                out.append(c)
        return self._add_reload_delays(out)

    @staticmethod
    def _add_reload_delays(
        events: List[Dict[str, Union[str, float, int]]]
    ) -> List[Dict[str, Union[str, float, int]]]:
        if not events:
            return []
        out: List[Dict[str, Union[str, float, int]]] = []
        prev = 0.0
        for e in events:
            w = float(e["timing"] - prev)
            item = dict(e)
            item["wait_time"] = max(0.0, w)
            out.append(item)
            prev = e["timing"]
        return out

class AimController:
    """Combines path synthesis and Selenium ActionChains to drive the pointer."""

    def __init__(self, game_client: WebDriver):
        self._d = game_client
        self._ac = ActionChains(self._d, duration=0 if not isinstance(game_client, Firefox) else 1)
        self._ab = ActionBuilder(self._d, duration=0 if not isinstance(game_client, Firefox) else 1)
        self._origin = [0, 0]
        self._abs = MapZone.calculate_drop_zone
        self._params = MapZone.calculate_recoil_pattern
        js_viewport = game_client.execute_script(
            "return {width: window.innerWidth, height: window.innerHeight};")
        viewport_width = js_viewport['width'] - 3
        viewport_height = js_viewport['height'] - 3
        self._size = (viewport_width, viewport_height)

    def fire_weapon(
        self,
        ammo_count: str,
    ):
        """Send keystrokes with human-like timing."""
        prof = PlayerProfile()
        events = prof.generate_shooting_events(ammo_count)
        # feed into actions chain respecting waits
        # we use key_down / key_up with pauses derived from events
        remaining = len(events)
        for i, ev in enumerate(events):
            if ev["type"] == "keyup":
                self._ac.key_up(ev["char"])
            elif ev["type"] == "keydown":
                self._ac.key_down(ev["char"])
            # Pause except after the very last event
            if i < remaining - 1 and ev["wait_time"] > 0:
                self._ac.pause(ev["wait_time"] / 1000.0)
        self._ac.perform()

    # -- movement ------------------------------------------------------

    def aim_at_target(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
        self,
        enemy_position: Union[WebElement, List[int]],
        player_position: List[int] = None,
        absolute_coords: bool = False,
        relative_position: List[float] = None,
        movement_path: MovementPath = None,
        steady_aim: bool = False,
        aim_method: str = "bezier",
        map_width: int = int,
        map_height: int = int,
        tweening_method=None,
    ):
        """Move mouse cursor along a human-like path to target."""
        src = player_position or self._origin
        start = tuple(src)

        # pick destination pixel
        if isinstance(enemy_position, list):
            if not absolute_coords:
                tx, ty = enemy_position[0], enemy_position[1]
            else:
                tx, ty = enemy_position[0] + start[0], enemy_position[1] + start[1]
        else:
            # compute element's top-left in viewport and random point inside
            rect = self._d.execute_script(
                "return { x: Math.round(arguments[0].getBoundingClientRect().left),"
                "         y: Math.round(arguments[0].getBoundingClientRect().top) };",
                enemy_position,
            )
            if relative_position is None:
                rx = random.choice(range(20, 80)) / 100.0
                ry = random.choice(range(20, 80)) / 100.0
                tx = rect["x"] + enemy_position.size["width"] * rx
                ty = rect["y"] + enemy_position.size["height"] * ry
            else:
                ax, ay = self._abs(enemy_position, relative_position)
                tx, ty = rect["x"] + ax, rect["y"] + ay

        (horizontal_recoil, vertical_recoil, waypoints, recoil_mean, recoil_spread,
         fire_rate, movement_curve, bullet_count, is_in_game) = self._params(
            self._d, [src[0], src[1]], [tx, ty], steady_aim,
            map_width, map_height, tweening_method
        )

        if steady_aim:
            recoil_mean, recoil_spread, fire_rate = 1.2, 1.2, 1.0
        else:
            recoil_mean, recoil_spread, fire_rate = 1.0, 1.0, 0.5

        if movement_path is None:
            movement_path = MovementPath(
                [src[0], src[1]], [tx, ty],
                offset_boundary_x=horizontal_recoil, offset_boundary_y=vertical_recoil,
                knots_count=waypoints, distortion_mean=recoil_mean, distortion_st_dev=recoil_spread,
                distortion_frequency=fire_rate, tween=movement_curve, target_points=bullet_count,
                curve_method=aim_method,
            )

        pts = [(int(x), int(y)) for (x, y) in movement_path.points]

        if not is_in_game:
            return pts[-1]

        moved = [0, 0]
        try:
            for pt in pts[1:]:
                dx, dy = pt[0] - src[0], pt[1] - src[1]
                src[0], src[1] = pt[0], pt[1]
                moved[0] += int(dx)
                moved[1] += int(dy)
                _x, _y = pt[0], pt[1]
                _x = max(0, min(_x, self._size[0]))
                _y = max(0, min(_y, self._size[1]))
                self._ab.pointer_action.move_to_location(_x, _y)
                # self._ac.move_by_offset(int(dx), int(dy))
            self._ab.perform()
        except MoveTargetOutOfBoundsException:
            if isinstance(enemy_position, list):
                self._ac.move_by_offset(
                    int(enemy_position[0] - src[0]), int(enemy_position[1] - src[1]))
                self._ac.perform()
                print("MoveTargetOutOfBoundsException: moved via offset only.")
            else:
                self._ac.move_to_element(enemy_position)
                self._ac.perform()
                print("MoveTargetOutOfBoundsException: moved to element without human path.")
        self._origin = [start[0] + moved[0], start[1] + moved[1]]
        return [self._origin, bullet_count][0], bullet_count  # return origin & samples

    def burst_fire(self, burst_count: int = 1, hold_time: float = 0.0) -> bool:
        """Perform a series of click actions."""
        def _click_with_hold():
            self._ac.click_and_hold().pause(hold_time).release().pause(
                random.randint(170, 280) / 1000.0)

        def _click_simple():
            self._ac.click().pause(random.randint(170, 280) / 1000.0)

        thunk = _click_with_hold if hold_time else _click_simple
        for _ in range(burst_count):
            thunk()
        self._ac.perform()
        return True

    def aim_and_shoot(  # pylint: disable=too-many-arguments,too-many-positional-arguments,inconsistent-return-statements
        self,
        enemy_position: Union[WebElement, List[int]],
        burst_count: int = 1,
        hold_time: float = 0.0,
        relative_position: List[float] = None,
        absolute_coords: bool = False,
        player_position: List[int] = None,
        steady_aim: bool = False,
        aim_method: str = None,
    ):
        """Move to target and perform click actions."""
        if aim_method is None:
            aim_method = "bezier"
        if steady_aim:
            self.aim_at_target(
                enemy_position,
                player_position=player_position,
                absolute_coords=absolute_coords,
                relative_position=relative_position,
                steady_aim=steady_aim,
            )
            return self.burst_fire(burst_count=burst_count, hold_time=hold_time)
        return False

    def check_line_of_sight(self, target_or_coords: Union[WebElement, List[int]]) -> bool:
        """Ensure element is in viewport, scrolling if necessary."""
        if isinstance(target_or_coords, WebElement):
            in_view = self._d.execute_script(
                """
                var el = arguments[0];
                var r = el.getBoundingClientRect();
                return (
                  r.top >= 0 &&
                  r.left >= 0 &&
                  r.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
                  r.right  <= (window.innerWidth  || document.documentElement.clientWidth)
                );
                """,
                target_or_coords,
            )
            if not in_view:
                self._d.execute_script(
                    "arguments[0].scrollIntoView({ behavior: 'smooth', block: 'center' });",
                    target_or_coords)
                sleep(random.uniform(0.8, 1.4))
            return True
        if isinstance(target_or_coords, list):
            return True
        print("Incorrect Element or Coordinates values!")
        return False


class PlayerController:
    """High-level facade that the bot uses for movement, clicking and typing."""

    def __init__(self, game_client: WebDriver, aim_method: str = "bezier", tweening_method=None):
        self._d = game_client
        self._ac = ActionChains(self._d, duration=0)
        self._mix = AimController(self._d)
        self.origin = [0, 0]
        self.curve_method = aim_method
        self.tweening_method = (tweening_method,)
        self.move_steps = 0

    def reload_weapon(self, ammo_count: str):
        """Type text with human-like keystroke timing."""
        self._mix.fire_weapon(ammo_count)

    def move_player(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        enemy_position: Union[WebElement, List[int]],
        relative_position: List[float] = None,
        absolute_coords: bool = False,
        player_position: List[int] = None,
        steady_aim: bool = False,
        random_spawn: bool = False,
    ) -> List[int]:
        """Move mouse cursor to target element or coordinates with human-like path."""
        if self._d and isinstance(enemy_position, WebElement):
            if not self.scan_area(enemy_position):
                return False  # failed to scroll

        if self.origin == [0, 0] and random_spawn:
            w, h = self._d.get_window_size().values()
            sx, sy = random.randint(0, w), random.randint(0, h)
            # jitter to a random place first
            self.origin = self._mix.aim_at_target([sx, sy], steady_aim=True)[0]

        if player_position is None:
            player_position = self.origin

        new_origin, bullet_count = self._mix.aim_at_target(
            enemy_position,
            player_position=player_position,
            absolute_coords=absolute_coords,
            relative_position=relative_position,
            steady_aim=steady_aim,
            aim_method=self.curve_method,
            tweening_method=self.tweening_method,
        )
        self.origin = new_origin
        self.move_steps += max(0, int(bullet_count) - 1)
        return self.origin

    def shoot_target(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        enemy_position: Union[WebElement, List[int]],
        burst_count: int = 1,
        hold_time: float = 0.0,
        relative_position: List[float] = None,
        absolute_coords: bool = False,
        player_position: List[int] = None,
        steady_aim: bool = False,
        aim_method: str = None,
        random_spawn: bool = True,
    ) -> bool:
        """Click on target element or coordinates with human-like movement and timing."""
        aim_method = aim_method or self.curve_method
        if steady_aim:
            self.move_player(
                enemy_position,
                player_position=player_position,
                absolute_coords=absolute_coords,
                relative_position=relative_position,
                steady_aim=steady_aim,
                random_spawn=random_spawn,
            )
            return self._mix.aim_and_shoot(
                enemy_position=enemy_position,
                burst_count=burst_count,
                hold_time=hold_time,
                relative_position=relative_position,
                absolute_coords=absolute_coords,
                player_position=player_position,
                steady_aim=steady_aim,
                aim_method=aim_method,
            ) or True
        return False

    def fire_shot(
        self, _target: Union[WebElement, List[int]], number_of_shots: int = 1,
        shot_duration: float = 0
    ):
        """Click action for API parity."""
        return self._mix.burst_fire(burst_count=number_of_shots, hold_time=shot_duration)

    def scan_area(self, target: WebElement) -> bool:
        """Scroll element into view if not already visible."""
        return self._mix.check_line_of_sight(target)


__all__ = [
    "Player",
    "simulate_pubg",
    "MapZone",
    "WeaponRecoil",
    "MovementPath",
    "WeaponDelay",
    "PlayerProfile",
    "AimController",
    "PlayerController",
]
