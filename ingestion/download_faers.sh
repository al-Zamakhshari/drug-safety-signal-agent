#!/bin/bash
# Download FAERS quarterly ASCII archives 2018–2026 (current era)
# For 2004–2017 historical data use: download_faers_historical.sh
#
# Usage: ./ingestion/download_faers.sh
# Output: ~/faers_data/

BASE="https://fis.fda.gov/content/Exports"
DIR="$HOME/faers_data"
mkdir -p "$DIR"

# Note: FDA mixes uppercase/lowercase Q — filenames match exactly as listed here
FILES=(
  # 2018
  faers_ascii_2018q1.zip  faers_ascii_2018q2.zip
  faers_ascii_2018q3.zip  faers_ascii_2018q4.zip
  # 2019
  faers_ascii_2019Q1.zip  faers_ascii_2019Q2.zip
  faers_ascii_2019Q3.zip  faers_ascii_2019Q4.zip
  # 2020
  faers_ascii_2020Q1.zip  faers_ascii_2020Q2.zip
  faers_ascii_2020Q3.zip  faers_ascii_2020Q4.zip
  # 2021
  faers_ascii_2021Q1.zip  faers_ascii_2021Q2.zip
  faers_ascii_2021Q3.zip  faers_ascii_2021Q4.zip
  # 2022
  faers_ascii_2022q1.zip  faers_ascii_2022q2.zip
  faers_ascii_2022Q3.zip  faers_ascii_2022Q4.zip
  # 2023
  faers_ascii_2023q1.zip  faers_ascii_2023q2.zip
  faers_ascii_2023Q3.zip  faers_ascii_2023Q4.zip
  # 2024
  faers_ascii_2024q1.zip  faers_ascii_2024q2.zip
  faers_ascii_2024q3.zip  faers_ascii_2024Q4.zip
  # 2025
  faers_ascii_2025q1.zip  faers_ascii_2025q2.zip
  faers_ascii_2025q3.zip  faers_ascii_2025Q4.zip
  # 2026
  faers_ascii_2026q1.zip
)

TOTAL=${#FILES[@]}
echo "Downloading $TOTAL quarters (2018 Q1 → 2026 Q1) to $DIR"
echo ""

downloaded=0; skipped=0; failed=0

for f in "${FILES[@]}"; do
  dest="$DIR/$f"
  if [ -f "$dest" ] && [ -s "$dest" ]; then
    echo "  ✓ $f (exists)"
    ((skipped++))
    continue
  fi
  echo -n "  ↓ $f ... "
  if curl -fsSL "$BASE/$f" -o "$dest" --retry 3 --retry-delay 5 2>/dev/null; then
    size=$(du -sh "$dest" 2>/dev/null | cut -f1)
    echo "✅ $size"
    ((downloaded++))
  else
    echo "❌ FAILED (check URL or network)"
    rm -f "$dest"
    ((failed++))
  fi
done

echo ""
echo "Done: $downloaded downloaded, $skipped already existed, $failed failed"
echo "Total size: $(du -sh "$DIR" | cut -f1)"
echo ""
echo "Next step:"
echo "  uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --all-drugs"
