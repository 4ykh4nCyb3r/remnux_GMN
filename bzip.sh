# Get bzip2's stems
sqlite3 compiled_binaries/test2_database.sqlite \
  "SELECT DISTINCT function_name FROM graphs WHERE function_name LIKE 'bzip2_%'" | \
  sed 's/^bzip2_//' | sort -u > /tmp/bzip2_stems.txt

echo "bzip2 stems: $(wc -l < /tmp/bzip2_stems.txt)"
echo ""

# Pairwise overlap
echo "Raw shared stems:"
echo "  bzip2 ∩ gzip: $(comm -12 /tmp/bzip2_stems.txt /tmp/gzip_stems.txt | wc -l)"
echo "  bzip2 ∩ tar:  $(comm -12 /tmp/bzip2_stems.txt /tmp/tar_stems.txt  | wc -l)"
echo "  bzip2 ∩ grep: $(comm -12 /tmp/bzip2_stems.txt /tmp/grep_stems.txt | wc -l)"
echo ""

# Apply name + train filters
echo "Clean (named, not in train):"
for other in gzip tar grep; do
  comm -12 /tmp/bzip2_stems.txt /tmp/${other}_stems.txt | \
    grep -v '^sub_' | \
    comm -23 - /tmp/train_stems.txt > /tmp/bzip2_${other}_clean.txt
  echo "  bzip2 ∩ ${other}: $(wc -l < /tmp/bzip2_${other}_clean.txt)"
done
echo ""

echo "Sample clean stems (bzip2 ∩ tar — likely the largest):"
head -20 /tmp/bzip2_tar_clean.txt
