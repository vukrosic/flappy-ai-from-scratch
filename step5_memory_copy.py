"""Step 5. DeepMind's two ideas, switched on and off: the memory and the copy.
Six runs of 200,000 frames for every way, next to the table of the Nature paper (2015, Extended Data Table 3)."""
import numpy as np
import runs

PAPER = {      # highest average game score after 10 million frames: memory + copy, memory only, copy only, neither
    'Breakout': (316.8, 240.7, 10.2, 3.2),
    'Enduro': (1006.3, 831.4, 141.9, 29.1),
    'River Raid': (7446.6, 4102.8, 2867.7, 1453.0),
    'Seaquest': (2894.4, 822.6, 1003.0, 275.8),
    'Space Invaders': (1088.9, 826.3, 373.2, 302.0),
}
ORDER = ('both', 'memory', 'copy', 'neither')


def rows(way):
    """Per run: best check, mean of all checks, last check, checks below 50 pipes after the first check of 90 or more."""
    out = []
    for seed in runs.SEEDS:
        m = [c[1] for c in runs.get(way, seed)['curve']]
        first = next((i for i, v in enumerate(m) if v >= 90), None)
        falls = sum(v < 50 for v in m[first + 1:]) if first is not None else None
        out.append((max(m), sum(m) / len(m), m[-1], falls))
    return out


print('our game: pipes a game in the 100 scored games (100 is the most), six runs each')
print(f'  {"":20s} {"best check":>10s} {"all checks":>10s} {"last check":>10s}   runs that reach 90   checks below 50 after reaching 90')
for way in ORDER + ('both, keeps trying', 'both, step falls'):
    r = rows(way)
    reach = [x for x in r if x[3] is not None]
    falls = f'{np.mean([x[3] for x in reach]):4.1f} a run' if reach else '   -'
    print(f'  {way:20s} {np.mean([x[0] for x in r]):10.1f} {np.mean([x[1] for x in r]):10.1f} {np.mean([x[2] for x in r]):10.1f}   '
          f'{len(reach)} of {len(r)}               {falls}')
print('  ("keeps trying": 1 % random moves to the end. "step falls": the step size falls to zero by the last frame.)')

print('\nper run, mean of all 40 checks:')
for way in ORDER:
    print(f'  {way:8s} ' + ' '.join(f'{x[1]:5.1f}' for x in rows(way)))

print('\nthe paper: five Atari games learned from the picture of the screen, 10 million frames')
print(f'  {"":16s} {"both":>8s} {"memory":>8s} {"copy":>8s} {"neither":>8s}')
for game, v in PAPER.items():
    print(f'  {game:16s} ' + ' '.join(f'{x:8.1f}' for x in v))
