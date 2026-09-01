#!/usr/bin/env bash
# Regenerate every published result, then re-sync the documents that quote them.
#
# This is the command behind the "Reproduce" section of the site. It exists
# because the published record is not one experiment but twelve, and a partial
# rebuild is how documentation drifts: regenerate the ablation but not the
# significance study and the two stop describing the same agent.
#
# Stages continue on failure and each records its exit code, so one flaky step
# does not silently truncate the rebuild -- read the summary before believing
# anything it produced.
#
# Expect several hours on a laptop CPU. Every stage is seeded; the only input
# that can move underneath it is the market data, which is why the run refuses
# to start if that has drifted from the pin.
#
#   bash tools/rebuild_all.sh            # everything
#   bash tools/rebuild_all.sh --quick    # smaller budgets, for checking wiring
#
# Logs land in runs/rebuild/ alongside a summary table.

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

PY="${PY:-python}"
LOG_DIR="$REPO/runs/rebuild"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.txt"
: > "$SUMMARY"

# --quick trades statistical power for a fast wiring check. The numbers it
# produces are NOT publishable -- too few seeds to say anything -- so it is only
# for confirming that every stage still runs end to end.
if [ "${1:-}" = "--quick" ]; then
  BIG=20000; MID=10000; SEEDS=2; SURR_SEEDS=2
  echo "QUICK MODE -- results are for wiring checks only, not for publication" \
    | tee -a "$SUMMARY"
else
  BIG=200000; MID=150000; SEEDS=5; SURR_SEEDS=3
fi

run () {
  local name="$1"; shift
  local started; started=$(date +%s)
  echo "=== [$name] starting: $* ===" | tee -a "$SUMMARY"
  "$@" > "$LOG_DIR/stage_$name.log" 2>&1
  local code=$?
  local elapsed=$(( $(date +%s) - started ))
  printf '%-22s exit=%-3s %4dm%02ds\n' "$name" "$code" \
    $((elapsed / 60)) $((elapsed % 60)) >> "$SUMMARY"
  echo "=== [$name] exit=$code after ${elapsed}s ==="
}

# Refuse to rebuild against drifted data. The whole claim of the published
# figures is that their inputs are pinned, so this gate is not optional: a
# re-fetch moves the train/test boundary and every downstream number with it.
if ! $PY tools/fetch_data.py --verify > "$LOG_DIR/stage_verify.log" 2>&1; then
  echo "ABORT: data/raw does not match data/SNAPSHOT.json" | tee -a "$SUMMARY"
  cat "$LOG_DIR/stage_verify.log" >> "$SUMMARY"
  exit 1
fi
echo "dataset verified against the pin" >> "$SUMMARY"

# --- the experiments -------------------------------------------------------
run site         $PY tools/build_site_data.py --real --timesteps $BIG --seed 42
run export       $PY tools/export_policy.py
run ablation_ms  $PY tools/ablation_multiseed.py --seeds 42 43 44 45 46 --timesteps 60000
run ablation1    $PY tools/ablation.py --timesteps 60000
run significance $PY tools/real_significance.py --seeds $SEEDS --timesteps $MID
run sig_synth_s  $PY tools/significance.py --market stock  --seeds $SEEDS --timesteps 40000
run sig_synth_c  $PY tools/significance.py --market crypto --seeds $SEEDS --timesteps 40000
run surr_synth   $PY tools/surrogate_test.py --mode synthetic --seeds $SEEDS --timesteps 60000
run surr_real    $PY tools/surrogate_test.py --mode real --seeds $SURR_SEEDS --timesteps 120000
run learning     $PY tools/learning_dynamics.py --budgets 20000 60000 120000 200000
run portfolio_s  $PY tools/portfolio_experiment.py --market stock  --timesteps $MID
run portfolio_c  $PY tools/portfolio_experiment.py --market crypto --timesteps $MID
run hpsweep      $PY tools/hyperparameter_sweep.py --seeds 3 --timesteps 60000
run figures      $PY tools/make_figures.py
run attribution  $PY tools/attribution_report.py

# --- and the documents that quote them -------------------------------------
# Last, and not optional. Every table in the README, RESULTS.md and the paper is
# generated from the artifacts above; skipping this is precisely how those
# documents came to describe runs that no longer existed.
run sync         $PY tools/sync_docs.py

echo "ALL STAGES COMPLETE" >> "$SUMMARY"
cat "$SUMMARY"

if grep -qv 'exit=0' <(grep 'exit=' "$SUMMARY"); then
  echo
  echo "NOTE: at least one stage failed. The documents may now describe a"
  echo "partially rebuilt record -- check $SUMMARY before publishing."
fi
