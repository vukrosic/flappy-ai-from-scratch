# Flappy AI from scratch

Code for the video **Build an AI That Learns to Play Flappy Bird**.
A program plays a small Flappy Bird clone. Nobody tells it the rules. It is told +1 when it passes a pipe, −1 when it crashes and 0 in every other frame.
Seven small programs build the method of DeepMind's Atari papers (Deep Q-Networks, DQN) from an empty file: the game, a table of numbers, the move that is tried on purpose, a neural network, the memory, the frozen copy, the game as a picture, and the kept best copy. About 830 lines of Python 3 with numpy and torch.

```bash
pip install numpy torch
sh run_all.sh         # seven steps, the tests and the exercise; expected output is in demo-output.txt
```

Every learning run is saved in `files/` (21 MB), so the steps read the runs and print in about one minute in total. The exercise trains three networks and takes the longest.
To measure everything again, delete `files/` and run `python3 runs.py`: 36 runs on three numbers (about 15 minutes with 6 free processor cores) and 12 runs on the picture (about 30 minutes each, 6 at a time). A learning run with the same seed gives the same numbers again (`tests.py` checks this on one run).

Scores are **pipes a game in 100 scored games**. The learner never trains on these 100 games. A scored game stops after 100 pipes, so 100 is a perfect score.

| File | Idea | What the run shows |
| --- | --- | --- |
| `game.py` | The game | 97 lines. The world is 400 pixels high, the hole in a pipe is 110. A tap sets the speed to 8 pixels a frame upward, gravity adds 1 downward every frame. The learner gets three numbers: distance to the next pipe, height from the middle of the hole, speed. |
| `step1_game.py` | Random taps and a hand-written rule | Random taps in 10 % of the frames: 0.08 pipes a game (8 pipes in 100 games). The rule "tap when the bird is more than 20 pixels below the middle of the hole": 100 pipes in all 100 games. |
| `table.py` | Q-learning with a table | 1,900 rows of two numbers (wait, tap). After every frame one number moves a fifth of the way toward "reward + 0.99 × the better number of the next row". |
| `step2_table.py` | The other move | 10,000 games of learning, three runs each. Always the higher number: 19.0, 59.4 and 23.8 pipes. The other move in 3 % of the frames, falling to 0 %: 83.6, 74.9 and 85.3 pipes. Run 1 with the other move played 18.8 million frames. |
| `step3_rows.py` | Finer rows | Squares of 40, 20, 10 and 5 pixels: 83.6, 80.9, 58.8 and 7.4 pipes after the same 10,000 games. The finest table has 121,600 rows and 21,727 were ever corrected. |
| `net.py` | A network in place of the table | Three numbers in, two out, 4,546 numbers inside. The same correction, done with the last 32 frames, or with 32 frames drawn from a memory of 50,000. The targets come from the network itself or from a copy that is refreshed every 500 frames. |
| `step4_network.py` | The network learns fast and does not stay good | The plain network, six runs of 200,000 frames. Every run reaches 90 pipes, after 5,000 to 45,000 frames. In every run, 4 to 9 later checks are below 50 pipes. Run 0: 100.0 pipes at 45,000 frames, 6.3 pipes at 50,000 frames. |
| `step5_memory_copy.py` | Memory and copy, on and off | Mean of all 40 checks, six runs each: neither 71.8, memory 63.4, both 60.2, copy 5.8. On three numbers the two ideas of the papers do not help. The step also prints the same table from the 2015 paper for five Atari games. |
| `picture.py` | The game as a picture | 40 by 24 grey squares, this frame and the frame before = 1,920 numbers. The network has two convolution layers and 189,874 numbers. |
| `step6_picture.py` | The same test on the picture | Best check of three runs of 200,000 frames. Neither: 0.0, 0.1, 0.3. Copy: 0.0, 0.0, 0.0. Memory: 79.7, 13.4, 33.4. Both: 90.8, 99.5, 95.9. Run 1 with both falls from 99.5 pipes to 44 within 50,000 frames. |
| `step7_best.py` | Keep the best copy | The best copy of each plain run on 1,000 new games: four of six clear every pipe in all 1,000, the other two in 996 and 998. The last network is worse in five runs of six. With the limit raised to 1,000 pipes, the best copy of run 0 reaches 1,000 in 100 of 100 new games. Picture runs: best copies 90.7, 98.0, 95.6 pipes; last networks 46.2, 83.0, 95.6. |
| `runs.py` | The saved runs | Makes every run that is missing in `files/` and reads the ones that exist. |
| `tests.py` | Checks of the claims | 25 tests. |
| `exercise.py` | Take the +1 away | See below. |

## Reading the numbers

- **Three or six runs are few.** The six plain runs on three numbers have means between 64.7 and 79.6 pipes. The four ways differ by more than that only for "copy alone". The picture versions have three runs each.
- **"Both ideas do not help" holds for three numbers only.** On the picture the order turns around: the plain network learns nothing and only memory + copy passes 90 pipes. This points the same way as Extended Data Table 3 of the 2015 paper and does not reproduce it.
- **The best copy is picked with the 100 scored games.** That is why step 7 tests it on 1,000 other games (seeds 20,000 to 20,999).
- **Frames, not seconds.** 18.8 million frames of the table are about 174 hours of play at 30 frames a second. The network reaches 100 pipes at 45,000 frames, which is 25 minutes of play. On a laptop the table run takes about 80 seconds and a network run on three numbers about two and a half minutes.

## What this is not

- **Not Atari.** The papers learn seven and 49 Atari games from the screen. This is one small game written for the video.
- **A small picture.** 40 × 24 squares and 2 frames against 84 × 84 pixels and 4 frames in the papers.
- **Smaller settings.** Memory of 50,000 frames (the papers: 1,000,000), copy refreshed every 500 frames (2015 paper: every 10,000 corrections), 200,000 frames of play per run (papers: 10 and 50 million), the Adam optimizer with step size 0.0001.
- **A simpler game than the phone game.** The bird is a square, a pipe comes every 50 frames, and the holes are drawn from a fixed range.

Sources: Mnih et al., "Playing Atari with Deep Reinforcement Learning", arXiv 1312.5602 (2013). Mnih et al., "Human-level control through deep reinforcement learning", Nature 518 (2015). Watkins, "Learning from Delayed Rewards", PhD thesis, Cambridge (1989). Lin, experience replay (1992).

## Try it

`exercise.py` changes one line of the game: a passed pipe gives 0, so the learner is told only −1 at a crash.

```python
class OnlyCrash(Game):
    def step(self, tap):
        return min(super().step(tap), 0)          # a passed pipe gives 0 now
```

Guess first whether it still learns to pass pipes. Then run `python3 exercise.py` (three learning runs, about 6 minutes on a laptop). It prints the answer and a second exercise: give −1 for every tap as well.

MIT licence.
