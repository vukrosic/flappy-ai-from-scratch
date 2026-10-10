"""A learner with a network instead of a table: three numbers in, two expected scores out.
The correction is the same as for the table. Two switches are the two ideas of DeepMind's DQN:
  memory   learn from 32 moments drawn at random from the last 50,000, not from the last 32 moments
  copy     take "the better of the two numbers in the next situation" from a copy of the network
           that is only refreshed every 500 frames"""
import random, time
import numpy as np
import torch
import torch.nn as nn
from game import *

torch.set_num_threads(1)
KEEP = 0.99          # how much a point one frame later is worth now
BATCH = 32           # moments in one correction
MEMORY = 50_000      # moments the memory holds
REFRESH = 500        # frames between two refreshes of the copy
RATE = 1e-4          # step size of the optimizer (Adam)
TRY0, TRY_FRAMES = 0.05, 50_000    # chance of a random move at the start; it falls to 0 over this many frames
SCALE = np.array([200.0, 200.0, 10.0])     # brings the three numbers to about -1 … 1


def new_net(wide=64):
    return nn.Sequential(nn.Linear(3, wide), nn.ReLU(), nn.Linear(wide, wide), nn.ReLU(), nn.Linear(wide, 2)).double()


def chooser(net):
    """The network as a player for play_many."""
    def choose(o):
        with torch.no_grad():
            return net(torch.from_numpy(o / SCALE)).argmax(1).numpy()
    return choose


class Numbers:
    """What the learner sees of a game: the three numbers, brought to about -1 … 1."""
    size, kind = 3, np.float64

    def __init__(self, game):
        self.game = game

    def see(self):
        return np.array(self.game.see()) / SCALE

    @staticmethod
    def score(net):
        """The pipes this network reaches in the 100 scored games."""
        return play_many(chooser(net), SCORE_SEEDS)


def learn(frames, seed=0, memory=True, copy=True, every=5000, log=None, wide=64, rate=RATE, refresh=REFRESH, size=MEMORY,
          floor=0.0, fall=False, cap=CAP, eyes=Numbers, make=new_net, try0=TRY0, try_frames=TRY_FRAMES):
    """Play for `frames` frames and correct the network after every frame.
    floor: the chance of a random move never falls below this.   fall: the step size falls to zero by the last frame.
    Returns the last network, the best network, and one line per check:
    (frames, mean pipes in the 100 scored games, worst game, perfect games, games played so far)."""
    torch.manual_seed(seed)
    rnd = random.Random(seed)
    net = make(wide)
    old = make(wide)
    old.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=rate)
    S, S2 = np.zeros((size, eyes.size), dtype=eyes.kind), np.zeros((size, eyes.size), dtype=eyes.kind)
    A, R, END = np.zeros(size, dtype=np.int64), np.zeros(size), np.zeros(size)
    game, n_games, curve, best, best_score = Game(seed * 1_000_000), 0, [], None, -1.0
    look = eyes(game)
    s = look.see()
    for f in range(frames):
        if rnd.random() < max(floor, try0 * (1 - f / try_frames)):
            a = rnd.randrange(2)
        else:
            with torch.no_grad():
                a = int(net(torch.from_numpy(s)[None]).argmax())
        r = game.step(a)
        s2 = look.see()
        i = f % size
        S[i], A[i], R[i], S2[i], END[i] = s, a, r, s2, float(r < 0)          # after a crash nothing follows
        s = s2
        if game.over:
            n_games += 1
            game = Game(seed * 1_000_000 + n_games, cap)
            look = eyes(game)
            s = look.see()
        if f + 1 >= BATCH:
            have = min(f + 1, size)
            idx = np.array(rnd.sample(range(have), BATCH)) if memory else (i - np.arange(BATCH)) % size
            with torch.no_grad():
                nxt = (old if copy else net)(torch.from_numpy(S2[idx])).max(1).values
                target = (torch.from_numpy(R[idx]) + KEEP * nxt * torch.from_numpy(1 - END[idx])).to(nxt.dtype)
            now = net(torch.from_numpy(S[idx])).gather(1, torch.from_numpy(A[idx])[:, None])[:, 0]
            loss = nn.functional.smooth_l1_loss(now, target)
            if fall:
                opt.param_groups[0]['lr'] = rate * (1 - f / frames)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if copy and (f + 1) % refresh == 0:
            old.load_state_dict(net.state_dict())
        if (f + 1) % every == 0:
            sc = eyes.score(net)
            curve.append((f + 1, float(sc.mean()), int(sc.min()), int((sc == CAP).sum()), n_games))
            if sc.mean() > best_score:
                best_score, best = float(sc.mean()), {k: v.clone() for k, v in net.state_dict().items()}
            if log:
                log(curve[-1])
    return net.state_dict(), best, curve


if __name__ == '__main__':
    t = time.time()
    learn(150_000, log=lambda c: print(f'{c[0]:7d} frames, {c[4]:5d} games: {c[1]:6.2f} pipes, worst {c[2]:3d}, perfect {c[3]:3d}   {time.time() - t:5.0f} s', flush=True))
