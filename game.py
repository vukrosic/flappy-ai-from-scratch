"""The game. A bird falls, a tap pushes it up, pipes come from the right. Touch anything and the game ends.
Game       one game, one step at a time (this is what the learner plays)
play_many  many games at once with the same rules (this is how a finished player is scored)
python3 game.py -> the numbers of the game, and how a player that taps at random does."""
import random
import numpy as np

H = 400          # height of the world, in pixels
BIRD = 24        # the bird is a square of this size
GRAV = 1.0       # the fall gets faster by this much every frame
FLAP = -8.0      # a tap sets the speed to this (up is negative)
VMAX = 10.0      # the fastest fall
SPEED = 4        # pixels a pipe moves left every frame
APART = 200      # pixels from one pipe to the next
PIPE_W = 52      # width of a pipe
GAP = 110        # height of the hole in a pipe
LOW, HIGH = 90, 310   # the middle of a hole lies between these heights
FIRST = 132      # the first pipe starts this far to the right of the bird (the same distance as every next pipe)
CAP = 100        # a scored game stops after this many pipes


def holes(seed, n):
    """The heights of the holes of one game. The same seed gives the same game."""
    r = random.Random(seed)
    return [r.randint(LOW, HIGH) for _ in range(n)]


class Game:
    def __init__(self, seed, cap=CAP):
        self.h = holes(seed, cap + 2)
        self.cap = cap
        self.y, self.v = H / 2, 0.0
        self.x = float(FIRST)        # left edge of the next pipe, measured from the middle of the bird
        self.score, self.frames, self.over = 0, 0, False

    def see(self):
        """What the player is told: how far the end of the next pipe is, how far the bird is below the middle of the hole, its speed."""
        return (self.x + PIPE_W + BIRD / 2, self.y - self.h[self.score], self.v)

    def step(self, tap):
        """One frame. Returns the reward: +1 for a pipe passed, -1 for a crash, 0 otherwise."""
        self.v = FLAP if tap else min(self.v + GRAV, VMAX)
        self.y += self.v
        self.x -= SPEED
        self.frames += 1
        c = self.h[self.score]
        top, bot = self.y - BIRD / 2, self.y + BIRD / 2
        inside = self.x < BIRD / 2 and self.x + PIPE_W > -BIRD / 2          # the bird is between the pipe's two edges
        if top < 0 or bot > H or (inside and (top < c - GAP / 2 or bot > c + GAP / 2)):
            self.over = True
            return -1.0
        if self.x + PIPE_W < -BIRD / 2:                                      # the pipe is behind the bird
            self.score += 1
            self.x += APART
            if self.score >= self.cap:
                self.over = True
            return 1.0
        return 0.0


def play_many(choose, seeds, cap=CAP, look=None):
    """Play one game per seed, all at once. choose(sight) gets an array [games, 3] and returns one 0 or 1 per game.
    look(x, y, h, score): another way of seeing the games (picture.py); choose then gets what look returns."""
    n = len(seeds)
    h = np.array([holes(s, cap + 2) for s in seeds], dtype=np.float64)
    y, v, x = np.full(n, H / 2), np.zeros(n), np.full(n, float(FIRST))
    score, live, rows = np.zeros(n, dtype=np.int64), np.ones(n, dtype=bool), np.arange(n)
    while live.any():
        c = h[rows, score]
        sight = look(x, y, h, score) if look else np.stack([x + PIPE_W + BIRD / 2, y - c, v], 1)
        tap = np.asarray(choose(sight)).astype(bool)
        v = np.where(tap, FLAP, np.minimum(v + GRAV, VMAX))
        y, x = y + v * live, x - SPEED * live
        top, bot = y - BIRD / 2, y + BIRD / 2
        inside = (x < BIRD / 2) & (x + PIPE_W > -BIRD / 2)
        crash = (top < 0) | (bot > H) | (inside & ((top < c - GAP / 2) | (bot > c + GAP / 2)))
        passed = live & ~crash & (x + PIPE_W < -BIRD / 2)
        score, x = score + passed, x + APART * passed
        live = live & ~crash & (score < cap)
    return score


SCORE_SEEDS = list(range(10_000, 10_100))     # the 100 games every player is scored on (never used for learning)

if __name__ == '__main__':
    print(f'world {H} pixels high, bird {BIRD}, hole {GAP} high, a pipe every {APART} pixels = {APART // SPEED} frames')
    print(f'a tap sets the speed to {FLAP:.0f}, the fall gains {GRAV:.0f} a frame up to {VMAX:.0f}')
    for p in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
        r = np.random.default_rng(0)
        s = play_many(lambda o: r.random(len(o)) < p, SCORE_SEEDS)
        g = [Game(k) for k in SCORE_SEEDS[:20]]
        r2, fr = random.Random(0), []
        for a in g:
            while not a.over:
                a.step(r2.random() < p)
            fr.append(a.frames)
        print(f'tap with chance {p:4.2f}: {s.mean():5.2f} pipes a game, best {s.max()}, a game lasts {sum(fr) / len(fr):5.1f} frames')
