"""A learner with a table. One row for every situation, two numbers in a row: the points it expects
from here on if it waits, and the points it expects if it taps. After every frame it corrects one number:
    expected[now][move]  moves toward  reward + 0.99 * (the better of the two numbers in the next row)
That is Q-learning (Watkins, 1989).
python3 table.py -> learns with rows of 40 pixels and prints the score after every 1,000 games."""
import os, random
import numpy as np
from game import *

KEEP = 0.99        # how much a point one frame later is worth now
RATE = 0.2         # how far a number moves toward its new value
TRY = 0.03         # chance of taking the other move, at the first game
TRY_PART = 0.6     # the chance falls to zero over this part of the games
NV = int(VMAX - FLAP) + 1     # the bird has 19 possible speeds
FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')


class Table:
    def __init__(self, px=40):
        """px: one row covers a square of px by px pixels (and one speed)."""
        self.px, self.nx, self.ny = px, -(-200 // px), 2 * H // px
        self.q = np.zeros((self.nx, self.ny, NV, 2))
        self.frames = 0

    def row(self, sight):
        dx, dy, v = sight
        return (min(int(dx) // self.px, self.nx - 1), min(max(int(dy + H) // self.px, 0), self.ny - 1), int(v - FLAP))

    def rows(self):
        return self.nx * self.ny * NV

    def rows_used(self):
        return int((self.q != 0).any(-1).sum())

    def choose(self, o):
        """The table as a player for play_many."""
        i = np.minimum(o[:, 0].astype(int) // self.px, self.nx - 1)
        j = np.clip((o[:, 1] + H).astype(int) // self.px, 0, self.ny - 1)
        k = (o[:, 2] - FLAP).astype(int)
        return self.q[i, j, k, 1] > self.q[i, j, k, 0]

    def learn(self, games, try_=TRY, seed=0, check=1000, log=None, keep=KEEP, rate=RATE):
        """Play `games` games and correct the table after every frame.
        Returns one line per check: (games, frames, mean pipes in the 100 scored games, worst game, perfect games)."""
        q, rnd, curve = self.q, random.Random(seed), []
        for g in range(games):
            chance = try_ * max(0.0, 1 - g / (games * TRY_PART))
            game = Game(seed * 1_000_000 + g)
            s = self.row(game.see())
            while not game.over:
                a = int(q[s][1] > q[s][0])                 # the move with the higher number; wait when they are equal
                if chance and rnd.random() < chance:
                    a = 1 - a                              # now and then: the other move
                r = game.step(a)
                s2 = self.row(game.see())
                target = r if r < 0 else r + keep * q[s2].max()      # after a crash nothing follows
                q[s][a] += rate * (target - q[s][a])
                s = s2
            self.frames += game.frames
            if (g + 1) % check == 0:
                sc = play_many(self.choose, SCORE_SEEDS)
                curve.append((g + 1, self.frames, float(sc.mean()), int(sc.min()), int((sc == CAP).sum())))
                if log:
                    log(curve[-1])
        return curve


def get(px=40, try_=TRY, games=10_000, seed=0):
    """One learning run, saved in files/: {'q': the table, 'curve': the checks}."""
    p = os.path.join(FILES, f'table-{px}-{try_}-{games}-{seed}.npz')
    if not os.path.exists(p):
        t = Table(px)
        curve = t.learn(games, try_, seed)
        os.makedirs(FILES, exist_ok=True)
        np.savez_compressed(p, q=t.q, curve=np.array(curve))
    d = np.load(p)
    t = Table(px)
    t.q = d['q']
    return t, d['curve']


if __name__ == '__main__':
    import time
    t0, t = time.time(), Table(40)
    print(f'{t.rows()} rows of two numbers')
    t.learn(10_000, log=lambda c: print(f'{c[0]:6d} games, {c[1]:9d} frames: {c[2]:6.2f} pipes, worst {c[3]:3d}, perfect {c[4]:3d}   {time.time() - t0:4.0f} s', flush=True))
