"""Exercise. Take the +1 away: the program is told only -1 at a crash, and 0 in every other frame.
Does it still learn to pass pipes? Guess first, then run:  python3 exercise.py"""
import net
from game import *


class OnlyCrash(Game):
    def step(self, tap):
        return min(super().step(tap), 0)          # a passed pipe gives 0 now


if __name__ == '__main__':
    net.Game = OnlyCrash                          # the learner plays this game; the 100 scored games still count pipes
    for seed in range(3):
        last, best, curve = net.learn(200_000, seed=seed, memory=False, copy=False)
        m = [c[1] for c in curve]
        first = next((c[0] for c in curve if c[1] >= 90), None)
        print(f'run {seed}: best check {max(m):5.1f} pipes, mean of all checks {sum(m) / len(m):5.1f}, '
              f'first check with 90 or more: {"never" if first is None else f"{first:,d} frames"}')
    print('with +1 and -1 (step4_network.py) the six runs reached 90 pipes after 5,000 to 45,000 frames')
    print('Why it can work: the only way to avoid the -1 is to stay in the air, and a bird that stays in the air passes pipes.')
    print('Your turn: give -1 for every tap as well, so that the bird should tap as rarely as it can. What happens to the score?')
