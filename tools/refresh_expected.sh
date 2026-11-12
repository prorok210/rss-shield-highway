#!/usr/bin/env bash
# Refreshes expected/ and runs/verification_run/ from a full run of run_all.sh (repository rule:
# every change of rss_shield/ or of the tests comes with the refreshed reference run).
set -eu
cd "$(dirname "$0")/.."
./run_all.sh
for f in detection_1B.md p0_profile.md pytest.txt run_all.txt scenario2_setup.md \
         table2_check.md table3.csv table3.md; do
  cp "results/$f" "expected/$f"
done
# table 1 of the reference keeps the rows of the reference environment; only the hashes follow
for key in "sha256 (rss_shield" "sha256 (tests"; do
  line="$(grep -F "$key" results/env_versions.md)"
  grep -vF "$key" expected/env_versions.md > expected/env_versions.md.tmp
  echo "$line" >> expected/env_versions.md.tmp
  mv expected/env_versions.md.tmp expected/env_versions.md
done
rm -rf runs/verification_run/results runs/verification_run/logs
mkdir -p runs/verification_run
cp -r results runs/verification_run/results
cp -r logs runs/verification_run/logs
echo "expected/ and runs/verification_run/ refreshed: $(tail -1 expected/pytest.txt)"
