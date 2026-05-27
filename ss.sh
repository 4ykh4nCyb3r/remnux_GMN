# Recompute everything fresh and show all intermediate counts
for bin in gzip tar grep; do
  sqlite3 compiled_binaries/test2_database.sqlite \
    "SELECT DISTINCT function_name FROM graphs WHERE function_name LIKE '${bin}_%'" | \
    sed "s/^${bin}_//" | sort -u > /tmp/${bin}_stems.txt
done

sqlite3 compiled_binaries/train_database.sqlite \
  "SELECT DISTINCT function_name FROM graphs" | \
  sed 's/^[^_]*_//' | sort -u > /tmp/train_stems.txt

echo "=== Step 1: per-binary stems ==="
for bin in gzip tar grep; do
  echo "  $bin: $(wc -l < /tmp/${bin}_stems.txt) stems"
done
echo "  train: $(wc -l < /tmp/train_stems.txt) stems"
echo ""

echo "=== Step 2: pairwise shared stems ==="
comm -12 /tmp/gzip_stems.txt /tmp/tar_stems.txt  > /tmp/gzip_tar_shared.txt
comm -12 /tmp/gzip_stems.txt /tmp/grep_stems.txt > /tmp/gzip_grep_shared.txt
comm -12 /tmp/tar_stems.txt  /tmp/grep_stems.txt > /tmp/tar_grep_shared.txt
comm -12 /tmp/gzip_tar_shared.txt /tmp/grep_stems.txt > /tmp/all_three_shared.txt
echo "  gzip ∩ tar:        $(wc -l < /tmp/gzip_tar_shared.txt)"
echo "  gzip ∩ grep:       $(wc -l < /tmp/gzip_grep_shared.txt)"
echo "  tar  ∩ grep:       $(wc -l < /tmp/tar_grep_shared.txt)"
echo "  gzip ∩ tar ∩ grep: $(wc -l < /tmp/all_three_shared.txt)"
echo ""

echo "=== Step 3: all-three before vs after filters ==="
echo "  raw (all three):                $(wc -l < /tmp/all_three_shared.txt)"
echo "  after sub_* removal:            $(grep -v '^sub_' /tmp/all_three_shared.txt | wc -l)"
echo "  after sub_* AND train removal:  $(grep -v '^sub_' /tmp/all_three_shared.txt | comm -23 - /tmp/train_stems.txt | wc -l)"
echo ""

echo "=== Step 4: what's actually in all-three shared (before train filter) ==="
head -30 /tmp/all_three_shared.txt
echo ""

echo "=== Step 5: pairwise CLEAN counts (named, train-removed) ==="
for pair in "gzip_tar" "gzip_grep" "tar_grep"; do
  grep -v '^sub_' /tmp/${pair}_shared.txt | \
    comm -23 - /tmp/train_stems.txt > /tmp/${pair}_clean.txt
  echo "  ${pair}: $(wc -l < /tmp/${pair}_clean.txt) clean stems"
done
echo ""

echo "=== Step 6: sample clean stems per pair ==="
for pair in "gzip_tar" "gzip_grep" "tar_grep"; do
  echo "  --- ${pair} ---"
  head -10 /tmp/${pair}_clean.txt
done
