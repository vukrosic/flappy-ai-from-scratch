"""Every learning run of the network is saved in files/, so a step file reads a run instead of repeating it.
python3 runs.py -> makes all runs that are missing: 36 runs on the three numbers (about 15 minutes with 6 free cores)
and 12 runs on the picture (about 30 minutes each, 6 at a time). Delete files/net-*.pt and files/pic-*.pt to measure again."""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, 'files')
FRAMES = 200_000
SEEDS = range(6)
WAYS = {                                   # name: the switches of net.learn
    'neither': dict(memory=False, copy=False),
    'memory': dict(memory=True, copy=False),
    'copy': dict(memory=False, copy=True),
    'both': dict(memory=True, copy=True),
    'both, keeps trying': dict(floor=0.01),
    'both, step falls': dict(fall=True),
}


PIC_SEEDS = range(3)                       # the runs on the picture: four ways, three runs each, checked every 10,000 frames
PIC_WAYS = ('neither', 'memory', 'copy', 'both')


def path(way, seed, frames=FRAMES, pic=False):
    short = '' if frames == FRAMES else f'-{frames}'
    return os.path.join(FILES, f'{"pic" if pic else "net"}-{way.replace(",", "").replace(" ", "-")}-{seed}{short}.pt')


def get(way, seed, frames=FRAMES, pic=False, log=None):
    """One run: {'last': weights, 'best': weights, 'curve': [(frames, pipes, worst, perfect, games), ...]}.
    A run with fewer frames is the start of the long run: the same seed gives the same learning.
    pic: the learner sees the picture of the game (picture.py), not the three numbers."""
    p = path(way, seed, frames, pic)
    if not os.path.exists(p):
        from net import learn
        how = dict(WAYS[way])
        if pic:
            import picture
            how.update(eyes=picture.Eyes, make=picture.new_net, wide=128, every=10_000)
        last, best, curve = learn(frames, seed=seed, log=log, **how)
        os.makedirs(FILES, exist_ok=True)
        torch.save(dict(last=last, best=best, curve=curve), p)
    return torch.load(p)


def new_games(seed, games=range(20_000, 21_000)):
    """The picture run with both ideas: pipes of its best copy and of its last network in 1,000 games it has never seen.
    Saved in files/, because a network that reads pictures needs minutes for 1,000 games."""
    import numpy as np
    p = path('both', seed, pic=True)[:-3] + '-new.npz'
    if not os.path.exists(p):
        import picture
        from game import play_many
        r, out = get('both', seed, pic=True), {}
        for which in ('best', 'last'):
            n = picture.new_net()
            n.load_state_dict(r[which])
            out[which] = play_many(picture.chooser(n), list(games), look=picture.ManyEyes())
        np.savez(p, **out)
    z = np.load(p)
    return z['best'], z['last']


def _one(job):
    way, seed, pic = job
    out = open(path(way, seed, pic=pic)[:-3] + '.log', 'w') if pic else None        # a picture run is slow: its checks go to a file
    r = get(way, seed, pic=pic, log=(lambda c: print(*c, file=out, flush=True)) if out else None)
    m = [c[1] for c in r['curve']]
    return f'{"picture " if pic else ""}{way:20s} seed {seed}: best {max(m):5.1f}, mean of all checks {sum(m) / len(m):5.1f}, last {m[-1]:5.1f}'


if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [(w, s, False) for s in SEEDS for w in WAYS if not os.path.exists(path(w, s))]
    jobs += [(w, s, True) for s in PIC_SEEDS for w in PIC_WAYS if not os.path.exists(path(w, s, pic=True))]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        for line in pool.imap(_one, jobs):
            print(line, flush=True)
