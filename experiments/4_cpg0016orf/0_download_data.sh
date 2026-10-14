#!/usr/bin/env bash
set -euo pipefail

BASE_S3="https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/source_all/workspace/profiles_assembled/ORF/v1.0a"
BASE_GITHUB_MORPHMAP="https://github.com/jump-cellpainting/2024_Chandrasekaran_Morphmap/raw"
BASE_GITHUB_DATASETS="https://github.com/jump-cellpainting/datasets/raw"

MORPHMAP_COMMIT="0ad6c5042583ad072c0d4c6797a582daaf5b6da5"
DATASETS_COMMIT="d824c432650409a1f67aa2ef045fbc05e33460ea"

urls=(
    "${BASE_S3}/profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet"
    "${BASE_GITHUB_MORPHMAP}/${MORPHMAP_COMMIT}/00.download-and-process-annotations/output/orf_metadata.tsv.gz"
    "${BASE_GITHUB_DATASETS}/${DATASETS_COMMIT}/metadata/compound.csv.gz"
    "${BASE_GITHUB_MORPHMAP}/${MORPHMAP_COMMIT}/00.download-and-process-annotations/input/experiment-metadata.tsv"
    "${BASE_GITHUB_MORPHMAP}/${MORPHMAP_COMMIT}/00.download-and-process-annotations/output/orf-reagents-low-infection-efficiency-and-outliers.csv.gz"
)

mkdir -p inputs

for url in "${urls[@]}"; do
    filename=$(basename "$url")
    output_file="inputs/${filename}"

    if [[ ! -f "${output_file}" ]]; then
        echo "Downloading ${filename}..."
        wget -q -nc -P inputs "${url}"
    else
        echo "File ${filename} already exists, skipping."
    fi
done

echo "cpg0016orf data download complete."
