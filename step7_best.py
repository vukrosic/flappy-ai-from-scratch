"""Step 7. The score never settles, so we do what people who train these learners do: check now and then, keep the best copy.
The best copy was picked with the 100 scored games. Is it good, or was it lucky? It plays 1,000 games it has never seen."""
import numpy as np
import net, runs
from game import *

NEW_GAMES = list(range(20_000, 21_000))
print('the best network of each plain run, on 1,000 new games (100 pipes is the most)')
for seed in runs.SEEDS:
    r = runs.get('neither', seed)
    best, last = net.new_net(), net.new_net()
    best.load_state_dict(r['best'])
    last.load_state_dict(r['last'])
    b, l = play_many(net.chooser(best), NEW_GAMES), play_many(net.chooser(last), NEW_GAMES)
    print(f'  run {seed}: best copy {b.mean():6.2f} pipes, {int((b == CAP).sum()):4d} of 1,000 games without a mistake'
          f' | last network {l.mean():6.2f} pipes, {int((l == CAP).sum()):4d} of 1,000')

best = net.new_net()
best.load_state_dict(runs.get('neither', 0)['best'])
long = play_many(net.chooser(best), NEW_GAMES[:100], cap=1000)
print(f'\nthe best copy of run 0, 100 new games that only stop after 1,000 pipes: {long.mean():.1f} pipes a game, '
      f'{int((long == 1000).sum())} of 100 games reach 1,000, shortest game {long.min()} pipes')

print('\nthe same for the picture runs with both ideas (step 6): best copy and last network on the same 1,000 new games')
for seed in runs.PIC_SEEDS:
    b, l = runs.new_games(seed)
    print(f'  run {seed}: best copy {b.mean():6.2f} pipes, {int((b == CAP).sum()):4d} of 1,000 games without a mistake'
          f' | last network {l.mean():6.2f} pipes, {int((l == CAP).sum()):4d} of 1,000')
