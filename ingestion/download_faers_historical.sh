#!/bin/bash
# Download FAERS/AERS quarterly archives 2004-2017 (historical data)
#
# Format note:
#   2004-2011: AERS format (files named aers_ascii_YYYYqN.zip, ISR primary key)
#   2012-2017: FAERS format (files named faers_ascii_YYYYqN.zip, primaryid)
#
# Usage: ./ingestion/download_faers_historical.sh
# Output: ~/faers_data/

BASE="https://fis.fda.gov/content/Exports"
DIR="$HOME/faers_data"
mkdir -p "$DIR"

# AERS era: 2004-2011 (lowercase 'q')
AERS_FILES=(
  aers_ascii_2004q1.zip aers_ascii_2004q2.zip aers_ascii_2004q3.zip aers_ascii_2004q4.zip
  aers_ascii_2005q1.zip aers_ascii_2005q2.zip aers_ascii_2005q3.zip aers_ascii_2005q4.zip
  aers_ascii_2006q1.zip aers_ascii_2006q2.zip aers_ascii_2006q3.zip aers_ascii_2006q4.zip
  aers_ascii_2007q1.zip aers_ascii_2007q2.zip aers_ascii_2007q3.zip aers_ascii_2007q4.zip
  aers_ascii_2008q1.zip aers_ascii_2008q2.zip aers_ascii_2008q3.zip aers_ascii_2008q4.zip
  aers_ascii_2009q1.zip aers_ascii_2009q2.zip aers_ascii_2009q3.zip aers_ascii_2009q4.zip
  aers_ascii_2010q1.zip aers_ascii_2010q2.zip aers_ascii_2010q3.zip aers_ascii_2010q4.zip
  aers_ascii_2011q1.zip aers_ascii_2011q2.zip aers_ascii_2011q3.zip aers_ascii_2011q4.zip
)

# FAERS era: 2012-2017 (mixed case Q)
FAERS_FILES=(
  faers_ascii_2012Q1.zip faers_ascii_2012Q2.zip faers_ascii_2012Q3.zip faers_ascii_2012Q4.zip
  faers_ascii_2013Q1.zip faers_ascii_2013Q2.zip faers_ascii_2013Q3.zip faers_ascii_2013Q4.zip
  faers_ascii_2014Q1.zip faers_ascii_2014Q2.zip faers_ascii_2014Q3.zip faers_ascii_2014Q4.zip
  faers_ascii_2015q1.zip faers_ascii_2015q2.zip faers_ascii_2015q3.zip faers_ascii_2015q4.zip
  faers_ascii_2016q1.zip faers_ascii_2016q2.zip faers_ascii_2016q3.zip faers_ascii_2016q4.zip
  faers_ascii_2017q1.zip faers_ascii_2017q2.zip faers_ascii_2017q3.zip faers_ascii_2017q4.zip
)

ALL_FILES=("${AERS_FILES[@]}" "${FAERS_FILES[@]}")
TOTAL=${#ALL_FILES[@]}
echo "Downloading $TOTAL quarters (2004 Q1 → 2017 Q4) to $DIR"
echo ""

downloaded=0; skipped=0; failed=0

for f in "${ALL_FILES[@]}"; do
  dest="$DIR/$f"
  if [ -f "$dest" ] && [ -s "$dest" ]; then
    echo "  ✓ $f (exists)"
    ((skipped++))
    continue
  fi
  echo -n "  ↓ $f ... "
  if curl -fsSL "$BASE/$f" -o "$dest" --retry 3 --retry-delay 5; then
    size=$(du -sh "$dest" | cut -f1)
    echo "✅ $size"
    ((downloaded++))
  else
    echo "❌ FAILED"
    rm -f "$dest"
    ((failed++))
  fi
done

echo ""
echo "Done: $downloaded downloaded, $skipped already existed, $failed failed"
echo ""
echo "Next: uv run python -m ingestion.faers_zip_indexer --dir ~/faers_data --all-drugs"
