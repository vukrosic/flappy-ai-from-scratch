"""Step 6. The same learner, but it no longer gets three numbers. It sees the game as a picture (picture.py):
40 by 24 squares, this frame and the frame before. Memory and copy are switched on and off again:
four ways, three runs of 200,000 frames each, one check every 10,000 frames."""
import numpy as np
import picture, runs

ORDER = ('neither', 'copy', 'memory', 'both')
print(f'the picture: {picture.ROWS} by {picture.COLS} squares, two frames = {picture.Eyes.size:,} numbers in')
print(f'the network: {sum(p.numel() for p in picture.new_net().parameters()):,} dials\n')
print('pipes a game in the 100 scored games (100 is the most)')
print(f'  {"":8s} {"best check of each run":>24s} {"mean of all checks":>20s} {"last check of each run":>24s}')
curves = {way: [[c[1] for c in runs.get(way, seed, pic=True)['curve']] for seed in runs.PIC_SEEDS] for way in ORDER}
for way in ORDER:
    m = curves[way]
    print(f'  {way:8s} {"  ".join(f"{max(x):5.1f}" for x in m):>24s} {np.mean(m):20.1f} {"  ".join(f"{x[-1]:5.1f}" for x in m):>24s}')

print('\nthe three runs with both ideas, check by check:')
for seed, m in zip(runs.PIC_SEEDS, curves['both']):
    print(f'  run {seed}: ' + ' '.join(f'{int(v):3d}' for v in m) + '   (rounded down)')
print('\non three numbers (step 5) the plain network was the best of the four. On the picture it learns nothing,')
print('and only the network with both ideas reaches 90 pipes. Its score still falls between checks, so: step 7.')
