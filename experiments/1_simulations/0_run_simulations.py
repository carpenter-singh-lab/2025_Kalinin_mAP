import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from map_utils.eval_metrics.evaluate import calculate_accuracy, evaluate_metrics
from map_utils.eval_metrics.generate_data import generate_features
from copairs.matching import assign_reference_index


def get_perturb_feature_range(
    n_feats: int,
    distribution: str,
    range_min: int,
    range_max: int,
    step_type: int,
    step_size: int = 1,
) -> tuple[str, list[int]]:
    """Generate a list of feature differences based on the provided configuration."""
    differ_range = range(range_min, range_max, step_size)
    if step_type == "bin_exp":
        differ_range = [2**i for i in differ_range]
    elif step_type == "linear":
        # always start at 1 differential feature
        differ_range = [1] + list(differ_range)
    return distribution, [int(i * (n_feats / 100)) for i in differ_range]


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """Run simulations."""
    save_file = cfg.save_file
    print(f"Save file: {save_file}")

    print(
        f"\n Control distributions: {cfg.control_distribs}, "
        f"perturb distributions: {cfg.perturb_distribs}, "
        f"distrib_proportions: {cfg.distrib_proportions}"
    )

    acc_results = []
    for n_feats in cfg.n_feats_range:
        for n_plates, controls in cfg.plates_controls.items():
            for n_controls in controls:
                perturb_distr, perturb_features = get_perturb_feature_range(
                    n_feats, **cfg.perturb_params
                )

                for n_perturb_features in perturb_features:
                    print(
                        f"\nProcessing {perturb_distr} differential features: {n_perturb_features} of {n_feats} "
                        f"({n_perturb_features / n_feats}), feats {n_feats}, plates {n_plates}, "
                        f"controls {n_controls}"
                    )

                    dframe, feats = generate_features(
                        n_feats,
                        n_plates,
                        cfg.n_perts + n_controls,
                        cfg.n_perts,
                        n_controls,
                        cfg.distrib_proportions,
                        cfg.control_distribs,
                        seed=cfg.seed,
                        features_differ={perturb_distr: n_perturb_features},
                        differ_params=cfg.perturb_distribs,
                    )
                    # each control gets its own index (its row label), perturbations get -1
                    dframe = assign_reference_index(
                        dframe,
                        dframe.index[dframe["Metadata_Perturbation"] == "negative_control"],
                        reference_col="Metadata_Control_Index",
                    )

                    metrics_results = evaluate_metrics(
                        profiles=pd.concat([dframe, feats], axis=1),
                        features=feats.columns,
                        metadata_features=dframe.columns,
                        metadata_groups=OmegaConf.to_object(cfg.profile_grouping),
                        metrics_config=cfg.metrics_config,
                    )

                    acc = calculate_accuracy(metrics_results)
                    acc.update(
                        {
                            "n_feats": n_feats,
                            "# replicates": n_plates,
                            "n_controls": n_controls * n_plates,
                            "features_differ": n_perturb_features,
                        }
                    )

                    acc_results.append(acc)
                    pd.DataFrame(acc_results).to_csv(save_file, index=False)

                    print(f"\nAccuracy: {acc}")

    print(f"Saving results to {save_file}")
    pd.DataFrame(acc_results).to_csv(save_file, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
