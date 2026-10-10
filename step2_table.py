"""Step 2. A table learns the game from +1 and -1 alone (Q-learning).
First it always takes the move with the higher number. Then it takes the other move now and then."""
from table import *

for name, try_ in (('always the move with the higher number', 0.0), ('the other move in 3 % of the frames, falling to 0 %', TRY)):
    print(name)
    for seed in range(3):
        t, curve = get(40, try_, seed=seed)
        print(f'  run {seed + 1}: pipes after 1,000 … 10,000 games: ' + ' '.join(f'{c[2]:5.1f}' for c in curve)
              + f' | {int(curve[-1][4]):2d} of 100 games without a mistake | {int(curve[-1][1]) / 1e6:4.1f} million frames')
t, curve = get(40, TRY)
print(f'\nthe table has {t.rows()} rows of two numbers; {t.rows_used()} rows were ever corrected')
q = t.q[2, :, int(0 - FLAP)]           # 80 to 120 pixels before the end of the pipe, speed 0
print('one column of it (80 to 120 pixels before the end of the pipe, speed 0): expected points for wait / tap')
for j in range(7, 14):
    lo = j * t.px - H
    print(f'  bird {lo:4d} to {lo + t.px:4d} pixels below the middle of the hole: {q[j][0]:6.2f} / {q[j][1]:6.2f}  -> {"tap" if q[j][1] > q[j][0] else "wait"}')
