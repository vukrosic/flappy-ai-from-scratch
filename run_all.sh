#!/bin/sh
# Run every step in order, then the tests, then the exercise.
# The learning runs are saved in files/, so the steps only read them (about one minute in total).
# The exercise trains three networks and is the slow part. To measure every run again: delete files/ and run python3 runs.py first.
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
for f in step1_game.py step2_table.py step3_rows.py step4_network.py step5_memory_copy.py step6_picture.py step7_best.py tests.py exercise.py; do
  echo
  echo "=== $f ==="
  $PY $f
done
