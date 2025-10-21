#!/usr/bin/env bash
mmseqs makepaddedseqdb uniref30/uniref30_2302_db uniref30/uniref30_2302_db_padded

set -euo pipefail

QUERIES=splits/IDRome_DB-clustered-train.fasta
DB_PREFIX=uniref30/uniref30_2302_db_padded   # padded!
THREADS=64
GPUS=1
E_VAL=1e-3
MAX_SEQ_ID=0.95

mkdir -p msas_out tmp

grep -A1 '^>' "$QUERIES" | sed '/^--$/d' | \
while read -r header; do
  read -r seq
  name=${header#>}
  echo ">>> $name"

  mkdir -p "msas_out/$name/a3m"
  printf "%s\n%s\n" "$header" "$seq" > tmp/"$name".fasta

  mmseqs createdb tmp/"$name".fasta tmp/"$name"_qdb

  # here’s the GPU-enabled search
  mmseqs search \
    tmp/"$name"_qdb \
    "$DB_PREFIX" \
    tmp/"$name"_res \
    tmp \
    --threads $THREADS \
    --gpu $GPUS \
    -e $E_VAL \
    --max-seq-id $MAX_SEQ_ID

  mmseqs result2msa \
    tmp/"$name"_qdb \
    "$DB_PREFIX" \
    tmp/"$name"_res \
    msas_out/"$name"/a3m/"$name".a3m \
    --db-load-mode 2 \
    --threads $THREADS

  mmseqs rmdb tmp/"$name"_qdb
  mmseqs rmdb tmp/"$name"_res
done

rm -rf tmp