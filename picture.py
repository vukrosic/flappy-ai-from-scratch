"""The game as a picture. The learner no longer gets three numbers; it gets what a person sees:
a grey picture of 40 by 24 squares (one square = 10 by 10 pixels of the world), and the picture one frame earlier,
so that it can tell which way the bird is moving.
python3 picture.py -> prints one picture as text."""
import numpy as np
import torch
import torch.nn as nn
from game import *

CELL = 10                    # one square of the picture covers this many pixels of the world
BACK, AHEAD = 30, 210        # the picture shows this many pixels behind and in front of the bird
ROWS, COLS = H // CELL, (BACK + AHEAD) // CELL
_ye = np.arange(ROWS + 1) * CELL            # edges of the rows
_xe = np.arange(COLS + 1) * CELL - BACK     # edges of the columns, measured from the middle of the bird


def _cover(lo, hi, edges):
    """Which part of every square lies between lo and hi (0 … 1)."""
    return np.clip(np.minimum(hi, edges[1:]) - np.maximum(lo, edges[:-1]), 0, CELL) / CELL


_BIRD_ACROSS = _cover(-BIRD / 2, BIRD / 2, _xe)       # the bird never moves sideways


def draw(game):
    """The picture of this frame: pipes are grey (128), the bird is white (255), the rest is black (0)."""
    img = np.zeros((ROWS, COLS))
    for k in (game.score - 1, game.score, game.score + 1):          # the pipe just passed, the next one, the one after
        if k < 0:
            continue
        left = game.x + (k - game.score) * APART
        across = _cover(left, left + PIPE_W, _xe)
        if across.any():
            c = game.h[k]
            down = _cover(0, c - GAP / 2, _ye) + _cover(c + GAP / 2, H, _ye)
            img = np.maximum(img, 0.5 * np.outer(down, across))
    bird = np.outer(_cover(game.y - BIRD / 2, game.y + BIRD / 2, _ye), _BIRD_ACROSS)
    return np.round(np.maximum(img, bird) * 255).astype(np.uint8)


def draw_many(x, y, h, score):
    """The same picture for many games at once (for play_many)."""
    n, rows = len(x), np.arange(len(x))
    img = np.zeros((n, ROWS, COLS))
    for d in (-1, 0, 1):
        k = score + d
        left = (x + d * APART)[:, None]
        across = _cover(left, left + PIPE_W, _xe) * (k >= 0)[:, None]
        c = h[rows, np.maximum(k, 0)][:, None]
        down = _cover(0, c - GAP / 2, _ye) + _cover(c + GAP / 2, H, _ye)
        img = np.maximum(img, 0.5 * down[:, :, None] * across[:, None, :])
    bird = _cover(y[:, None] - BIRD / 2, y[:, None] + BIRD / 2, _ye)[:, :, None] * _BIRD_ACROSS[None, None, :]
    return np.round(np.maximum(img, bird) * 255).astype(np.uint8)


class Eyes:
    """Turns a game into what the learner sees: the picture before and this picture, as one long row of numbers."""
    size, kind = 2 * ROWS * COLS, np.uint8

    def __init__(self, game):
        self.game = game
        self.last = draw(game).ravel()

    def see(self):
        now = draw(self.game).ravel()
        out = np.concatenate([self.last, now])
        self.last = now
        return out


    @staticmethod
    def score(net):
        """The pipes this network reaches in the 100 scored games."""
        with torch.no_grad():
            return play_many(lambda o: net(torch.from_numpy(o)).argmax(1).numpy(), SCORE_SEEDS, look=ManyEyes())


class Squares(nn.Module):
    """One long row of grey levels 0 … 255 becomes two pictures of numbers 0 … 1."""
    def forward(self, x):
        return (x.float() / 255).view(-1, 2, ROWS, COLS)


def new_net(wide=128):
    """Two pictures in, two expected scores out. The first two layers slide a small window over the picture
    (convolutions, as in the paper), so the same pipe edge is recognised wherever it is."""
    return nn.Sequential(Squares(), nn.Conv2d(2, 16, 4, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2), nn.ReLU(), nn.Flatten(),
                         nn.Linear(32 * 9 * 5, wide), nn.ReLU(), nn.Linear(wide, 2))


def chooser(net):
    """The network as a player for play_many (with look=ManyEyes())."""
    def choose(o):
        with torch.no_grad():
            return net(torch.from_numpy(o)).argmax(1).numpy()
    return choose


class ManyEyes:
    """The same for play_many."""
    def __init__(self):
        self.last = None

    def __call__(self, x, y, h, score):
        now = draw_many(x, y, h, score).reshape(len(x), -1)
        out = np.concatenate([now if self.last is None else self.last, now], 1)
        self.last = now
        return out


if __name__ == '__main__':
    g = Game(0)
    for _ in range(18):
        g.step(g.y > g.h[g.score] + 20)
    print(f'{ROWS} by {COLS} squares, two pictures = {2 * ROWS * COLS} numbers')
    for row in draw(g):
        print(''.join(' .:-=+*#%@'[int(v) * 9 // 255] for v in row))
