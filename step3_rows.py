"""Step 3. The limit of a table: every row has to be visited before it holds anything.
Finer rows describe the game better and take longer to fill. A picture has no table at all."""
from table import *

print('the same 10,000 games of learning, with rows of a different size')
for px in (40, 20, 10, 5):
    t, curve = get(px, TRY)
    print(f'  squares of {px:2d} pixels: {t.rows():7,d} rows, {t.rows_used():6,d} ever corrected, '
          f'{curve[-1][2]:5.1f} pipes a game, {int(curve[-1][4]):2d} of 100 games without a mistake')

from picture import ROWS, COLS
n = ROWS * COLS
print(f'\na picture of {ROWS} by {COLS} squares with only black or white squares has 2^{n} possible pictures')
print(f'that is a number with {len(str(2 ** n))} digits; the atoms of the visible universe: a number with about 80 digits')
