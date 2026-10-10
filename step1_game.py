"""Step 1. The game, a player that taps at random, and a rule written by a person.
The rule shows what there is to learn: one line is enough to play without a mistake."""
import random
import numpy as np
from game import *

print(f'the world is {H} pixels high, the bird {BIRD}, the hole in a pipe {GAP}; a pipe comes every {APART // SPEED} frames')
print(f'a tap sets the speed to {-FLAP:.0f} pixels a frame upward; every frame without a tap adds {GRAV:.0f} downward, up to {VMAX:.0f}')
print(f'a game is scored up to {CAP} pipes; every player below plays the same {len(SCORE_SEEDS)} games\n')

print('tapping at random')
for p in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
    r = np.random.default_rng(0)
    s = play_many(lambda o: r.random(len(o)) < p, SCORE_SEEDS)
    print(f'  a tap in {p:4.0%} of the frames: {s.mean():5.2f} pipes a game, best game {s.max()}')

print('\na rule written by a person: tap when the bird is more than N pixels below the middle of the hole')
for n in (-10, 0, 10, 20, 30, 40):
    s = play_many(lambda o: o[:, 1] > n, SCORE_SEEDS)
    print(f'  N = {n:3d}: {s.mean():6.2f} pipes a game, {int((s == CAP).sum()):3d} of 100 games without a mistake')

g = Game(SCORE_SEEDS[0])
while not g.over:
    g.step(g.see()[1] > 20)
print(f'\none game with N = 20 has {g.frames} frames, so {g.frames} decisions; the player is told "+1" {g.score} times')
