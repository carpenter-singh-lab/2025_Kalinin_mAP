#!/usr/bin/env python3
# coding: utf-8

import warnings
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import umap
import numpy as np
import pandas as pd

from copairs import compute, map
from copairs.matching import assign_reference_index

warnings.filterwarnings("ignore", category=UserWarning, module="umap")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def p_values(dframe: pd.DataFrame, null_size: int, seed: int) -> np.ndarray:
    """Compute p-values using copairs.compute.p_values."""
    mask = dframe["n_pos_pairs"] > 0
    pvals = np.full(len(dframe), np.nan, dtype=np.float32)
    scores = dframe.loc[mask, "average_precision"].values
    null_confs = dframe.loc[mask, ["n_pos_pairs", "n_total_pairs"]].values
    pvals[mask] = compute.p_values(scores, null_confs, null_size, seed)
    return pvals


def extract_features(df: pd.DataFrame):
    """
    Extract metadata, CP, and DP feature columns from the dataframe.
    Returns meta_features and a data_config dictionary.
    """
    meta_features = df.filter(regex="^(?!CP__|DP__)").columns
    cp_features = df.filter(regex="^(CP__)").columns
    dp_features = df.filter(regex="^(DP__)").columns
    data_config = {"cp": cp_features, "dp": dp_features}
    return meta_features, data_config


def compute_umap_for_class(mc: str, sc_df: pd.DataFrame, output_dir: Path, reducer):
    """Compute and save UMAP embedding for a single morphological class."""
    mc_umap_path = output_dir / f"umap_{mc}.parquet"
    if mc_umap_path.exists():
        return pd.read_parquet(mc_umap_path)

    df_mc = sc_df[sc_df["Mitocheck_Phenotypic_Class"].isin([mc, "neg_control"])]
    mc_umap = df_mc.filter(regex="^(?!CP_|DP_).*").reset_index(drop=True).copy()

    for feats in ["CP_", "DP_"]:
        embed = reducer.fit_transform(df_mc.filter(regex=f"^{feats}").values)
        mc_umap[f"{feats}umap_0"] = embed[:, 0]
        mc_umap[f"{feats}umap_1"] = embed[:, 1]

    mc_umap.to_parquet(mc_umap_path)
    return mc_umap


def compute_umap_embeddings(sc_df: pd.DataFrame, output_dir: Path) -> tuple[list, list]:
    """
    Compute UMAP embeddings for CellProfiler (CP) and DeepProfiler (DP) features for each
    morphological class (excluding 'neg_control'), in parallel.
    """
    morpho_classes = [
        mc for mc in sc_df["Mitocheck_Phenotypic_Class"].unique() if mc != "neg_control"
    ]

    reducers = [
        umap.UMAP(metric="cosine", n_neighbors=30, min_dist=0.1, random_state=42)
    ] * len(morpho_classes)
    # spawn, not fork: libgomp aborts a forked child that reuses OpenMP once the parent has
    # (numba-parallel copairs kernels run on GNU OpenMP)
    with ProcessPoolExecutor(
        max_workers=12, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        results = list(
            executor.map(
                compute_umap_for_class,
                morpho_classes,
                [sc_df] * len(morpho_classes),
                [output_dir] * len(morpho_classes),
                reducers,
            )
        )

    return results, morpho_classes


def compute_average_precision(
    df: pd.DataFrame,
    meta_features: list,
    feature_cols: list,
    grouping: str,
    null_size: int = 500000,
    seed: int = 0,
    feature_type: str = "",
) -> pd.DataFrame:
    """
    Compute average precision (AP) for a given feature type and grouping.
    """
    ap_result = map.average_precision(
        df[meta_features],
        df[feature_cols].values,
        pos_sameby=[grouping, "Metadata_control_index"],
        pos_diffby=[],
        neg_sameby=[],
        neg_diffby=[grouping, "Metadata_control_index"],
    )
    ap_result = ap_result.query(
        "Mitocheck_Phenotypic_Class != 'neg_control'"
    ).reset_index(drop=True)
    ap_result["p_value"] = p_values(ap_result, null_size=null_size, seed=seed)
    ap_result["p < 0.05"] = ap_result["p_value"] < 0.05
    ap_result["-log10(AP p-value)"] = -np.log10(ap_result["p_value"])
    if "AP" in ap_result.columns:
        ap_result.rename(columns={"AP": "average_precision"}, inplace=True)
    ap_result["Features"] = feature_type
    return ap_result


def run_ap_calculations(
    df: pd.DataFrame,
    groupings: list,
    data_config: dict,
    null_size: int = 500000,
    seed: int = 0,
) -> dict:
    """
    Run AP calculations for each grouping and each feature type.
    Returns a dictionary mapping grouping names to their combined results.
    """
    results = {grouping: [] for grouping in groupings}
    meta_features = df.filter(regex="^(?!CP__|DP__)").columns
    for grouping in groupings:
        print(f"\nCalculating AP for {grouping}")
        for feature_type, feature_cols in data_config.items():
            print(f"Calculating AP for feature type: {feature_type}")
            ap_result = compute_average_precision(
                df, meta_features, feature_cols, grouping, null_size, seed, feature_type
            )
            results[grouping].append(ap_result)
    return {k: pd.concat(v, axis=0) for k, v in results.items()}


def save_results(results: dict, output_dir: Path):
    """
    Save the AP results to CSV files in the output directory.
    """
    for grouping, df in results.items():
        output_path = output_dir / f"{grouping}_ap_results.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved results for {grouping} to {output_path}")


def main():
    output_dir = Path("outputs")
    processed_path = output_dir / "merged_sc_fs.parquet"

    df = pd.read_parquet(processed_path)
    _, data_config = extract_features(df)

    groupings = ["Metadata_Gene", "Mitocheck_Phenotypic_Class"]
    df = assign_reference_index(
        df,
        condition="Mitocheck_Phenotypic_Class == 'neg_control'",
        reference_col="Metadata_control_index",
    )
    results = run_ap_calculations(df, groupings, data_config, null_size=500000, seed=0)
    save_results(results, output_dir)

    compute_umap_embeddings(df, output_dir)


if __name__ == "__main__":
    main()
