# 3. Calculate mAP for bulk and single cell perturbseq data
#
# Also calculate UMAP embeddings for each perturbation at the same time.

import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import umap
import numpy as np
import pandas as pd

from map_utils.map import calculate_map

np.random.seed(42)


def load_bulk_activity(data_dir: Path, gse_id: str) -> pd.DataFrame:
    """
    Load bulk activity screen phenotypes and add a perturbation column.
    """
    activity_file = (
        data_dir / "paper_supplement" / "Table_S16_perturb-seq_screen_phenotypes.txt"
    )
    activity_df = pd.read_csv(activity_file, sep="\t").rename(
        {"Unnamed: 0": "id"}, axis="columns"
    )
    activity_df = activity_df.assign(
        perturbation=activity_df.gene + "_" + activity_df.id
    )
    print("Bulk activity data shape:", activity_df.shape)
    return activity_df


def process_bulk_data(gse_id: str, inputs_dir: Path, results_dir: Path) -> None:
    """
    Process bulk perturbseq data: calculate mAP and merge with activity data.
    """
    # Load bulk final analytical data
    bulk_file = results_dir / f"{gse_id}_bulk_final_analytical.tsv.gz"
    bulk_df = pd.read_csv(bulk_file, sep="\t")
    bulk_df = bulk_df.query("Metadata_gene_identity != '*'").reset_index(drop=True)
    bulk_df["Metadata_reference_index"] = np.where(
        bulk_df["Metadata_gene_identity"] == "neg", bulk_df.index, -1
    )
    print("Bulk data shape after filtering:", bulk_df.shape)

    # Determine genes to retain (variance > 0.001)
    non_meta = bulk_df.filter(regex="^(?!Metadata_)")
    var_df = (
        pd.DataFrame(non_meta.var() > 0.001)
        .reset_index()
        .rename({"index": "gene", 0: "keep"}, axis="columns")
    )
    genes_to_retain = var_df.query("keep").gene.tolist()
    print("Number of genes to retain in bulk:", len(genes_to_retain))

    # Define mapping configurations for bulk data
    pair_config = {
        "pos_sameby": ["Metadata_gene_identity", "Metadata_reference_index"],
        "pos_diffby": [],
        "neg_sameby": [],
        "neg_diffby": ["Metadata_gene_identity", "Metadata_reference_index"],
    }
    map_config = {
        "null_size": 10000,
        "groupby_columns": ["Metadata_guide_identity", "Metadata_gene_identity"],
    }
    metadata_cols = bulk_df.filter(regex="Metadata_").columns.tolist()

    # Calculate mAP on bulk data
    control_config = {
        "condition": "Metadata_gene_identity == 'neg'",
        "reference_col": "Metadata_reference_index",
    }
    map_results = calculate_map(
        bulk_df.loc[:, metadata_cols + genes_to_retain],
        pair_config,
        map_config,
        reference_index_config=control_config,
    )
    print("Bulk map_results shape:", map_results.shape)

    # Merge with activity data
    activity_df = load_bulk_activity(inputs_dir, gse_id)
    result = map_results.merge(
        activity_df, left_on="Metadata_guide_identity", right_on="perturbation"
    )
    output_results_file = results_dir / f"{gse_id}_map.tsv"
    result.to_csv(output_results_file, sep="\t", index=False)
    print("Final bulk map shape:", result.shape)
    print("Bulk mAP results written to:", output_results_file)


def load_and_filter_data(
    gse_id: str, results_dir: Path
) -> tuple[pd.DataFrame, list, pd.DataFrame]:
    """Loads single-cell data, filters invalid cells, and prepares negative controls."""
    gene_exp_file = results_dir / f"{gse_id}_final_analytical.tsv.gz"
    sc_df = pd.read_csv(gene_exp_file, sep="\t", low_memory=False)

    # Extract gene features (all except metadata)
    gene_features = [col for col in sc_df.columns if not col.startswith("Metadata_")]
    metadata_cols = sc_df.filter(regex="^Metadata_").columns.tolist()
    print("Single-cell data shape (raw):", sc_df.shape)

    # Filter invalid cells
    sc_df = (
        sc_df.query("Metadata_gene_identity != '*'")
        .query("Metadata_good_coverage == True")
        .reset_index(drop=True)
    )
    sc_df["Metadata_reference_index"] = np.where(
        sc_df["Metadata_gene_identity"] == "neg", sc_df.index, -1
    )
    print("Single-cell data shape after filtering:", sc_df.shape)

    # Sample 20% of negative control cells
    sc_neg_controls_df = sc_df.query("Metadata_gene_identity == 'neg'").sample(frac=0.2)
    print("Negative control single-cell data shape:", sc_neg_controls_df.shape)

    return sc_df, gene_features, sc_neg_controls_df


def compute_map_results(
    sc_df: pd.DataFrame, neg_controls_df: pd.DataFrame, results_dir: Path, gse_id: str
):
    """Computes mAP results and saves them."""
    pair_config = {
        "pos_sameby": ["Metadata_guide_identity", "Metadata_reference_index"],
        "pos_diffby": [],
        "neg_sameby": [],
        "neg_diffby": ["Metadata_guide_identity", "Metadata_reference_index"],
    }
    map_config = {
        "null_size": 500000,
        "groupby_columns": ["Metadata_guide_identity", "Metadata_cell_identity"],
    }

    # Identify valid genes
    valid_target_genes = sc_df["Metadata_gene_identity"].dropna().unique()
    valid_target_genes = [g for g in valid_target_genes if g not in ["neg", "*", "nan"]]
    print("Valid target genes (single-cell):", valid_target_genes)

    # Prepare subset of single-cell data (targets + sampled negatives)
    subset_df = pd.concat(
        [sc_df.query("Metadata_gene_identity in @valid_target_genes"), neg_controls_df]
    ).reset_index(drop=True)

    # Compute mAP
    all_sc_map_results = calculate_map(
        subset_df.loc[
            :,
            subset_df.filter(regex="^Metadata_").columns.tolist()
            + genes_to_retain(sc_df),
        ],
        pair_config,
        map_config,
    )

    # Merge mAP results with metadata
    merge_cols = [
        "Metadata_cell_identity",
        "Metadata_guide_identity",
        "Metadata_gene_identity",
    ]
    all_sc_map_results = all_sc_map_results.merge(
        subset_df.loc[subset_df.Metadata_gene_identity != "neg", merge_cols],
        on=merge_cols[:-1],
    )

    output_sc_map_file = results_dir / f"{gse_id}_single_cell_map.tsv.gz"
    all_sc_map_results.to_csv(
        output_sc_map_file, sep="\t", compression="gzip", index=False
    )

    print("Single-cell mAP results shape:", all_sc_map_results.shape)
    print("Single-cell mAP results written to:", output_sc_map_file)

    return valid_target_genes


def compute_umap_for_gene(gene, sc_df, neg_controls_df, gene_features):
    """Compute UMAP embeddings for a single gene, merged with negative controls."""
    print(f"Processing UMAP embeddings for gene: {gene}...")
    subset_sc_df = sc_df.query("Metadata_gene_identity == @gene")
    subset_sc_df = pd.concat([subset_sc_df, neg_controls_df]).reset_index(drop=True)

    reducer = umap.UMAP(metric="cosine", n_neighbors=30, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(subset_sc_df.loc[:, gene_features])

    embedding_df = pd.concat(
        [
            subset_sc_df.drop(columns=gene_features).reset_index(drop=True),
            pd.DataFrame(embedding, columns=["umap_0", "umap_1"]),
        ],
        axis=1,
    )
    return embedding_df.assign(Metadata_gene_identity=gene)


def compute_umap_embeddings(
    sc_df: pd.DataFrame,
    neg_controls_df: pd.DataFrame,
    gene_features: list,
    valid_target_genes: list,
    results_dir: Path,
    gse_id: str,
):
    """Computes and saves UMAP embeddings in parallel."""
    print("Computing UMAP embeddings per gene...")
    # spawn, not fork: libgomp aborts a forked child that reuses OpenMP once the parent has
    # (numba-parallel copairs kernels run on GNU OpenMP)
    with ProcessPoolExecutor(
        max_workers=12, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        umap_results = list(
            executor.map(
                compute_umap_for_gene,
                valid_target_genes,
                [sc_df] * len(valid_target_genes),
                [neg_controls_df] * len(valid_target_genes),
                [gene_features] * len(valid_target_genes),
            )
        )

    all_umap_embeddings = pd.concat(umap_results).reset_index(drop=True)
    output_umap_file = results_dir / f"{gse_id}_single_cell_umap_embeddings.tsv.gz"
    all_umap_embeddings.to_csv(
        output_umap_file, sep="\t", compression="gzip", index=False
    )

    print("UMAP embeddings shape:", all_umap_embeddings.shape)
    print("UMAP embeddings written to:", output_umap_file)


def process_single_cell_data(gse_id: str, results_dir: Path) -> None:
    """
    Main function to process single-cell perturbseq data.
    """
    sc_df, gene_features, neg_controls_df = load_and_filter_data(gse_id, results_dir)
    valid_target_genes = compute_map_results(
        sc_df, neg_controls_df, results_dir, gse_id
    )
    compute_umap_embeddings(
        sc_df, neg_controls_df, gene_features, valid_target_genes, results_dir, gse_id
    )


def genes_to_retain(df: pd.DataFrame) -> list:
    """
    Compute a list of genes to retain based on variance threshold.
    Variance is computed on columns that do not start with 'Metadata_'.
    """
    non_meta = df.filter(regex="^(?!Metadata_)")
    var_df = (
        pd.DataFrame(non_meta.var() > 0.001)
        .reset_index()
        .rename({"index": "gene", 0: "keep"}, axis="columns")
    )
    genes = var_df.query("keep").gene.tolist()
    print("Number of genes to retain:", len(genes))
    return genes


def main():
    gse_id = "GSE132080"
    inputs_dir = Path("inputs")
    results_dir = Path("outputs")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Processing bulk perturbseq data...")
    process_bulk_data(gse_id, inputs_dir, results_dir)

    print("\nProcessing single-cell perturbseq data...")
    process_single_cell_data(gse_id, results_dir)


if __name__ == "__main__":
    main()
