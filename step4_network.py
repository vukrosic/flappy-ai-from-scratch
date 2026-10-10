"""Step 4. A network instead of the table: three numbers in, two expected scores out, the same correction.
It needs far fewer frames than the table, and it does not stay good."""
import numpy as np
import torch
import net, runs, table
from game import *

n = sum(p.numel() for p in net.new_net().parameters())
t, tc = table.get(40, table.TRY)
print(f'the table: {t.rows():,d} rows of two numbers = {2 * t.rows():,d} numbers. the network: {n:,d} numbers')

print('\nthe plain network (no memory, no copy), six runs of 200,000 frames, checked every 5,000 frames')
for seed in runs.SEEDS:
    c = runs.get('neither', seed)['curve']
    m = [x[1] for x in c]
    first = next(x[0] for x in c if x[1] >= 90)
    after = m[[x[0] for x in c].index(first) + 1:]
    print(f'  run {seed}: first check with 90 pipes or more at {first:6,d} frames; of the {len(after)} checks after it, '
          f'{sum(v < 50 for v in after):2d} are below 50 pipes; last check {m[-1]:5.1f}')
print(f'the table needed {int(tc[-1][1]):,d} frames for {tc[-1][2]:.1f} pipes')

c = runs.get('neither', 0)['curve']
print('\nrun 0, check by check (pipes a game):')
print('  ' + ' '.join(f'{x[1]:3.0f}' for x in c[:20]))
print('  ' + ' '.join(f'{x[1]:3.0f}' for x in c[20:]))

# the fall between two checks: the same run, stopped at 50,000 frames
r = runs.get('neither', 0, 50_000)
good, fallen = net.new_net(), net.new_net()
good.load_state_dict(r['best'])          # the network of the check at 45,000 frames
fallen.load_state_dict(r['last'])        # 5,000 frames later
wa = torch.cat([p.detach().flatten() for p in good.parameters()])
wb = torch.cat([p.detach().flatten() for p in fallen.parameters()])
seen, cg, cf = [], net.chooser(good), net.chooser(fallen)


def spy(o):
    seen.append(o.copy())
    return cg(o)


sg = play_many(spy, SCORE_SEEDS)
sf = play_many(cf, SCORE_SEEDS)
o = np.concatenate(seen)
print(f'\nat 45,000 frames: {sg.mean():.1f} pipes. at 50,000 frames: {sf.mean():.1f} pipes')
print(f'between the two, a number of the network moved by {float((wa - wb).abs().mean()):.4f} on average; '
      f'its average size is {float(wa.abs().mean()):.4f}')
print(f'in the {len(o):,d} situations of the good games the two networks choose the same move {100 * (cg(o) == cf(o)).mean():.0f} % of the time')
