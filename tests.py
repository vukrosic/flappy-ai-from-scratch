"""Tests. Each one checks a claim the film makes. Run: python3 tests.py"""
import numpy as np
import torch
from game import *
from table import Table, KEEP, RATE
import table, net, picture, runs

FAILED = []


def check(name, ok, note=''):
    print('ok    ' if ok else 'FAILED', name, note)
    if not ok:
        FAILED.append(name)


def one_by_one(choose, seeds, look=None):
    """The same players as play_many, played with the class Game, one game and one frame at a time."""
    out = []
    for s in seeds:
        g = Game(s)
        eyes = look(g) if look else None
        while not g.over:
            o = eyes.see() if look else np.array(g.see())
            g.step(int(choose(o[None])[0]))
        out.append(g.score)
    return np.array(out)


# 1. the game
g = Game(0)
while not g.over:
    g.step(0)
check('a bird that never taps hits the ground', g.score == 0 and g.y + BIRD / 2 > H, f'(after {g.frames} frames)')
g = Game(0)
while not g.over:
    g.step(1)
check('a bird that always taps hits the ceiling', g.score == 0 and g.y - BIRD / 2 < 0, f'(after {g.frames} frames)')
check('the same seed gives the same game', holes(7, 50) == holes(7, 50) and holes(7, 50) != holes(8, 50))
rule = lambda o: o[:, 1] > 20
s = play_many(rule, SCORE_SEEDS)
check('a one-line rule plays 100 games without a mistake', (s == CAP).all())
g, rewards = Game(SCORE_SEEDS[0]), []
while not g.over:
    rewards.append(g.step(g.see()[1] > 20))
check('the learner is told +1 once per pipe and 0 otherwise', rewards.count(1.0) == 100 and rewards.count(0.0) == len(rewards) - 100,
      f'({len(rewards)} frames, {rewards.count(1.0)} rewards)')
some = SCORE_SEEDS[:12]
for name, p in (('the rule', rule), ('a rule that crashes', lambda o: o[:, 1] > 38), ('a late rule', lambda o: o[:, 1] > -5)):
    a, b = play_many(p, some), one_by_one(p, some)
    check(f'many games at once = one game at a time ({name})', (a == b).all(), f'(mean {a.mean():.1f} pipes)')

# 2. the table
t = Table(40)
t.q[1, 2, 3] = [0.25, 0.5]
check('equal numbers mean wait, a higher number for tap means tap',
      not t.choose(np.array([[10.0, 0.0, 0.0]]))[0] and t.choose(np.array([[50.0, 90.0 - H, 3.0 + FLAP]]))[0]
      and t.row((50.0, 90.0 - H, 3.0 + FLAP)) == (1, 2, 3))
t = Table(40)
t.learn(300, check=300)
check('after a crash only the crashing move is corrected, toward -1', t.q.min() < -0.15 and t.q.min() >= -1.0, f'(lowest number {t.q.min():.2f})')
t0, c0 = table.get(40, 0.0)
t1, c1 = table.get(40, table.TRY)
check('trying the other move now and then beats never trying it', c1[-1][2] > c0[-1][2] + 10, f'({c0[-1][2]:.1f} -> {c1[-1][2]:.1f} pipes)')
check('the saved table scores what its curve says', abs(play_many(t1.choose, SCORE_SEEDS).mean() - c1[-1][2]) < 1e-9)
a, b = play_many(t1.choose, some), one_by_one(t1.choose, some)
check('the table plays the same in both kinds of game', (a == b).all())
fine = [table.get(px, table.TRY)[1][-1][2] for px in (40, 10, 5)]
check('finer rows learn less in the same 10,000 games', fine[0] > fine[1] > fine[2], f'({fine[0]:.1f}, {fine[1]:.1f}, {fine[2]:.1f} pipes)')

# 3. the network
r = runs.get('neither', 0)
n = net.new_net()
n.load_state_dict(r['best'])
best = max(c[1] for c in r['curve'])
check('the saved best network scores what its curve says', abs(net.Numbers.score(n).mean() - best) < 1e-9, f'({best:.1f} pipes)')
a, b = play_many(net.chooser(n), some), one_by_one(net.chooser(n), some)
check('the network plays the same in both kinds of game', (a == b).all())
_, _, again = net.learn(10_000, seed=0, memory=False, copy=False)
check('the same seed gives the same learning run', [tuple(c) for c in again] == [tuple(c) for c in r['curve'][:2]])

# 4. the picture
g, e, m, same = Game(3), None, picture.ManyEyes(), True
e = picture.Eyes(g)
x, y, h, sc = np.array([g.x]), np.array([g.y]), np.array([g.h], dtype=np.float64), np.array([0])
while not g.over and g.score < 3:
    one = e.see()
    many = m(np.array([g.x]), np.array([g.y]), h, np.array([g.score]))[0]
    same &= bool((one == many).all())
    g.step(g.see()[1] > 20)
check('one picture at a time = many pictures at once', same and g.score == 3, f'({g.frames} frames)')
img = picture.draw(Game(0))
check('a picture is 40 by 24 grey levels with a white bird', img.shape == (40, 24) and img.max() == 255 and img.dtype == np.uint8)

best = {w: [max(c[1] for c in runs.get(w, sd, pic=True)['curve']) for sd in runs.PIC_SEEDS] for w in runs.PIC_WAYS}
check('on the picture only the network with both ideas reaches 90 pipes',
      min(best['both']) >= 90 and max(max(best[w]) for w in ('neither', 'memory', 'copy')) < 90,
      f"(both: {', '.join(f'{b:.1f}' for b in best['both'])})")
fall = []
for sd in runs.PIC_SEEDS:
    m = [c[1] for c in runs.get('both', sd, pic=True)['curve']]
    fall.append(max(a - b for a, b in zip(m, m[1:])))
check('in every picture run the score falls by 20 pipes or more between two checks', min(fall) >= 20,
      f"(largest falls: {', '.join(f'{f:.0f}' for f in fall)})")
r = runs.get('both', 1, pic=True)
n = picture.new_net()
n.load_state_dict(r['best'])
top = max(c[1] for c in r['curve'])
check('the kept copy of picture run 1 scores what its curve says', abs(picture.Eyes.score(n).mean() - top) < 1e-9, f'({top:.2f} pipes)')
b, l = runs.new_games(1)
check('in 1,000 new games the kept copy beats the last network', b.mean() > l.mean() + 5, f'({b.mean():.1f} against {l.mean():.1f} pipes)')

# 5. the exercise
from exercise import OnlyCrash
g, told = OnlyCrash(SCORE_SEEDS[0]), []
while not g.over:
    told.append(g.step(g.see()[1] > 20))
check('the exercise game says nothing at a pipe', g.score == CAP and set(told) == {0.0}, f'({g.score} pipes, {len(told)} frames)')
g, told = OnlyCrash(0), []
while not g.over:
    told.append(g.step(0))
check('the exercise game still says -1 at a crash', told[-1] == -1.0 and set(told[:-1]) <= {0.0})

print()
print('ALL TESTS PASS' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}')
