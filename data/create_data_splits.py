"""Run with `HF_TOKEN=xxx python -m data.create_data_splits --push_splits_to_hub` to create data splits for the steering datasets and save in HF_REPO_ID."""
from typing import List, Dict, Tuple, Type, Optional, Any
from datasets import Dataset, ClassLabel
import math
from collections import defaultdict, Counter
from data.steering_data import FormattedDataset, HF_REPO_ID
from utils.enums import SteeringFormat, FormatMode, OutputType, Concept
from data.bias import BBQ, ToxiGen
from data.refusal import SaladBench, Alpaca, JailbreakV
from data.hallucination import FaithEvalCounterfactual, FaithEvalInconsistent, FaithEvalUnanswerable, PreciseWiki
from data.secondary_datasets import (
    GPQA,
    ARC_C,
    CMTEST,
    GSM8K,
    TruthfulQA,
    Twinviews,
    EnronEmail,
    DarkBenchAnthro,
    DarkBenchBrandBias,
    DarkBenchSynchopancy,
    DarkBenchSneaking,
    DarkBenchRetention,
)
from pathlib import Path
import argparse

SPLIT_PCTS = [0.4, 0.1, 0.5]  # Default split percentages for stratification (train, val, test).

# Per-dataset split configuration keyed by class name
PER_DATASET_CFG = {
    "ToxiGen":               {"test_size": 4500, "split_pct": [0.4, 0.1, 0.5]},
    # BBQ: test fixed at 5000, remaining split 4:1 for train/val
    "BBQ":                   {"test_size": 5000, "split_pct": [0.4, 0.1, 0.5]},
    # PreciseWiki: 40/10/50 on first 10000 examples
    "PreciseWiki":           {"split_pct": [0.4, 0.1, 0.5]},
    # FaithEval subsets: full data, 40/10/50
    "FaithEvalCounterfactual": {"split_pct": [0.4, 0.1, 0.5]},
    "FaithEvalInconsistent":   {"split_pct": [0.4, 0.1, 0.5]},
    "FaithEvalUnanswerable":   {"split_pct": [0.4, 0.1, 0.5]},
    # SaladBench: 40/10/50
    # "SaladBench":            {"test_size": 5000, "split_pct": [0.4, 0.1, 0.5]},
    "SaladBench":            {"test_size": 4293, "split_pct": [0.4, 0.1, 0.5]},  # 4293 is half of 8586, the total size of SaladBench.
    # JailbreakV: 40/10/50
    # "JailbreakV":            {"split_pct": [0.4, 0.1, 0.5]},
    # Alpaca: only train (4000), no val/test
    # "Alpaca":                {"train_only": True},
    # "Alpaca":                {"test_size": 5000, "split_pct": [0.4, 0.1, 0.5]},
    "Alpaca":                {"test_size": 4293, "split_pct": [0.4, 0.1, 0.5]},  # needs to be same as SaladBench for consistency
    # Secondary datasets: full dataset as test only, no train/val splitting
    "GPQA": {"test_only": True},
    "ARC_C": {"test_only": True},
    "CMTEST": {"test_only": True},
    "GSM8K": {"test_only": True},
    "TruthfulQA": {"test_only": True},
    "Twinviews": {"test_only": True},
    "EnronEmail": {"test_only": True},
    "DarkBenchAnthro": {"test_only": True},
    "DarkBenchBrandBias": {"test_only": True},
    "DarkBenchSynchopancy": {"test_only": True},
    "DarkBenchSneaking": {"test_only": True},
    "DarkBenchRetention": {"test_only": True},
}

def _create_split_instance(
    original_fds: FormattedDataset,
    original_fds_keys: Dict[str, Any],
    hf_split_dataset: Dataset,
    split_mode: FormatMode,
) -> FormattedDataset:
    """
    Creates a new FormattedDataset instance (of the same subclass as the original)
    containing a split subset of the data, using the modified __init__.

    Args:
        original_fds: The original FormattedDataset instance before splitting.
        original_fds_keys: The keys from the original instance to initialize the new instance.
        hf_split_dataset: The Hugging Face Dataset subset for this split.
        split_mode: The mode of the split (train, validation, test).

    Returns:
        A new FormattedDataset instance configured for the split.
    """
    # Use the type of the original instance to create a new one of the same subclass
    Subclass = type(original_fds)

    # Create the new instance, passing the split data via _existing_dataset
    initialize_keys = {
        **original_fds_keys,
        'format_mode': split_mode,
        '_existing_dataset': hf_split_dataset
    }
    new_instance = Subclass(**initialize_keys)

    return new_instance

def get_dataset_splits_by_concept(
    datasets_list: List[FormattedDataset],
    keys_list: List[Dict[str, Any]] = None,  # for use with subclassing.
    split_pct: List[float] = SPLIT_PCTS,
    per_dataset_pct: Optional[Dict[str, List[float]]] = None,
    seed: int = 42
):
    """
    Split each FormattedDataset into train/val/test, keyed by its `.concept`.
    """
    # split_pct: global [train_frac, val_frac, test_frac]
    # per_dataset_pct: optional mapping from dataset.name to [train, val, test]
    train_d, val_d, test_d = defaultdict(list), defaultdict(list), defaultdict(list)

    def _ensure_classlabel(ds, col):
        # turn a text column into ClassLabel *in place* for stratification
        return ds.class_encode_column(col) if (col and not isinstance(ds.features[col], ClassLabel)) else ds
    
    def _convert_classlabel_to_string(ds, col):
        # convert ClassLabel back to original strings and cast to string type
        if col and isinstance(ds.features.get(col), ClassLabel):
            # Store the ClassLabel feature before mapping
            classlabel_feature = ds.features[col]
            # Convert the entire column at once
            string_values = [classlabel_feature.int2str(val) for val in ds[col]]
            # Create new dataset with converted values
            ds_dict = ds.to_dict()
            ds_dict[col] = string_values
            ds = Dataset.from_dict(ds_dict)
        return ds

    def _two_stage_split(ds: Dataset, strat_col: Optional[str], pct: List[float]):
        # empty
        # pct: [train_frac, val_frac, test_frac] for this dataset
        train_p, val_p, test_p = pct
        # empty dataset
        if len(ds) == 0:
            empty = ds.select([])
            return empty, empty, empty
        # If requested, cast stratification column to ClassLabel
        ds_loc = _ensure_classlabel(ds, strat_col)
        strat_arg = {"stratify_by_column": strat_col} if strat_col else {}
        # 1) extract test set
        try:
            split1 = ds_loc.train_test_split(test_size=test_p, seed=seed, **strat_arg)
        except ValueError:
            split1 = ds_loc.train_test_split(test_size=test_p, seed=seed)
        tv, test_ds = split1["train"], split1["test"]
        # 2) split train vs validation
        if train_p + val_p > 0:
            val_frac = val_p / (train_p + val_p)
            try:
                split2 = tv.train_test_split(test_size=val_frac, seed=seed, **strat_arg)
            except ValueError:
                split2 = tv.train_test_split(test_size=val_frac, seed=seed)
            return split2["train"], split2["test"], test_ds
        # no train/val requested
        empty = tv.select([])
        return empty, empty, test_ds

    for orig, keys in zip(datasets_list, keys_list or [{}]):
        ds = orig.dataset
        if not ds or len(ds)==0:
            continue

        key = orig.stratification_key
        # only keep the key if the column actually exists
        strat = key if key in ds.column_names else None

        # Per-dataset split config
        cls_name = type(orig).__name__
        cfg = {} if per_dataset_pct is None else per_dataset_pct.get(cls_name, {})
        # Prepare for stratification
        ds_loc = _ensure_classlabel(ds, strat)
        # 0) Test-only datasets: full dataset as test only
        if cfg.get("test_only", False):
            tr = ds_loc.select([])
            va = ds_loc.select([])
            te = ds_loc
        # 1) Train-only datasets (e.g., Alpaca)
        elif cfg.get("train_only", False):
            tr = ds_loc
            va = ds_loc.select([])
            te = ds_loc.select([])
        # 2) Absolute test size (integer)
        elif "test_size" in cfg:
            # Split off test set of fixed size
            try:
                split1 = ds_loc.train_test_split(
                    test_size=cfg["test_size"], seed=seed,
                    stratify_by_column=strat if strat else None)
            except Exception:
                split1 = ds_loc.train_test_split(test_size=cfg["test_size"], seed=seed)
            # Assign train+val pool (tv) and fixed-size test split
            tv, te = split1["train"], split1["test"]
            # Constrain tv pool to desired size so train+val totals match default ratios
            # Compute number of tv examples needed: (train_pct+val_pct)/test_pct * test_size
            train_pct, val_pct, test_pct = split_pct
            if test_pct > 0:
                desired_tv = int(round(cfg["test_size"] * (train_pct + val_pct) / test_pct))
                if len(tv) > desired_tv:
                    try:
                        small = tv.train_test_split(
                            train_size=desired_tv, seed=seed,
                            stratify_by_column=strat if strat else None)
                        tv = small["train"]
                    except Exception:
                        # Fallback to simple selection
                        tv = tv.shuffle(seed=seed).select(range(desired_tv))
            # Split remaining into train/validation using default ratios (train: 0.4, val: 0.1)
            # Compute validation fraction relative to remaining data
            train_pct, val_pct, _ = split_pct
            if train_pct + val_pct > 0:
                val_frac = val_pct / (train_pct + val_pct)
                try:
                    split2 = tv.train_test_split(
                        test_size=val_frac, seed=seed,
                        stratify_by_column=strat if strat else None)
                except Exception:
                    split2 = tv.train_test_split(test_size=val_frac, seed=seed)
                tr, va = split2["train"], split2["test"]
            else:
                tr, va = tv, tv.select([])
        # 3) Ratio-based split
        else:
            # Use per-dataset ratios or global default
            pct_ds = cfg.get("split_pct", split_pct)
            tr, va, te = _two_stage_split(ds, strat, pct_ds)
        # Verify stratification: print distribution of stratification key in each split
        if strat:
            def _print_dist(split_ds, split_name):
                # Extract values, mapping ClassLabel ints to strings if necessary
                if isinstance(split_ds.features[strat], ClassLabel):
                    labels = split_ds[strat]
                    vals = [split_ds.features[strat].int2str(l) for l in labels]
                else:
                    vals = split_ds[strat]
                cnts = Counter(vals)
                total = len(vals)
                print(f"Distribution for dataset '{orig.name}' (concept={orig.concept}) split='{split_name}' on key '{strat}':")
                for val, cnt in sorted(cnts.items()):
                    pct = cnt / total * 100 if total > 0 else 0
                    print(f"  {val}: {cnt} ({pct:.1f}%)")
            _print_dist(tr, 'train')
            _print_dist(va, 'validation')
            _print_dist(te, 'test')
        else:
            print(f"No stratification key '{key}' in dataset '{orig.name}', available columns: {ds.column_names}")

        # Convert ClassLabel columns back to original strings before creating split instances
        tr = _convert_classlabel_to_string(tr, strat) if len(tr) > 0 else tr
        va = _convert_classlabel_to_string(va, strat) if len(va) > 0 else va
        te = _convert_classlabel_to_string(te, strat) if len(te) > 0 else te
        
        if len(tr)>0:
            train_d[orig.concept].append(_create_split_instance(orig, keys, tr, FormatMode.TRAIN))
        if len(va)>0:
            val_d[orig.concept].append(_create_split_instance(orig, keys, va, FormatMode.VALIDATION))
        if len(te)>0:
            test_d[orig.concept].append(_create_split_instance(orig, keys, te, FormatMode.TEST))

    return dict(train_d), dict(val_d), dict(test_d)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Create data splits for datasets.")
    parser.add_argument(
        "--push_splits_to_hub",
        action="store_true",
        help="Push the generated splits to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--split_pcts",
        type=str,
        default=None,
        help="Global split fractions as comma-separated floats: train,val,test (must sum to <=1).",
    )
    parser.add_argument(
        "--dataset_pcts",
        action="append",
        default=[],
        help="Per-dataset split fractions in format NAME=tr,va,te; can repeat for multiple datasets.",
    )
    args = parser.parse_args()
    # Define dataset-specific sample sizes (None = use full)
    # BBQ, SaladBench, FaithEval, ToxiGen use full; PreciseWiki uses first 10000; Alpaca uses first 4000
    bbq_keys = dict(n=None, format_mode=FormatMode.TRAIN)
    salad_keys = dict(n=None, format_mode=FormatMode.TRAIN)
    # jailbreakv_keys = dict(n=None, format_mode=FormatMode.TRAIN)
    alpaca_keys = dict(n=None, format_mode=FormatMode.TRAIN)
    faitheval_counterfactual_keys = dict(n=None, format_mode=FormatMode.TRAIN)
    faitheval_inconsistent_keys = dict(n=None, format_mode=FormatMode.TRAIN)
    faitheval_unanswerable_keys = dict(n=None, format_mode=FormatMode.TRAIN)
    precise_wiki_keys = dict(n=None, format_mode=FormatMode.TRAIN)
    toxigen_keys = dict(n=None, format_mode=FormatMode.TRAIN)
    # Secondary datasets: use full dataset for test only
    gpqa_keys = dict(n=None, format_mode=FormatMode.TEST)
    arc_c_keys = dict(n=None, format_mode=FormatMode.TEST)
    cmtest_keys = dict(n=None, format_mode=FormatMode.TEST)
    gsm8k_keys = dict(n=None, format_mode=FormatMode.TEST)
    truthfulqa_keys = dict(n=None, format_mode=FormatMode.TEST)
    twinviews_keys = dict(n=None, format_mode=FormatMode.TEST)
    enronemail_keys = dict(n=None, format_mode=FormatMode.TEST)
    darkbench_anthro_keys = dict(n=None, format_mode=FormatMode.TEST)
    darkbench_brandbias_keys = dict(n=None, format_mode=FormatMode.TEST)
    darkbench_sycophancy_keys = dict(n=None, format_mode=FormatMode.TEST)
    darkbench_sneaking_keys = dict(n=None, format_mode=FormatMode.TEST)
    darkbench_retention_keys = dict(n=None, format_mode=FormatMode.TEST)
    keys_list = [
        bbq_keys,
        salad_keys,
        # jailbreakv_keys,
        alpaca_keys,
        faitheval_counterfactual_keys,
        faitheval_inconsistent_keys,
        faitheval_unanswerable_keys,
        precise_wiki_keys,
        toxigen_keys,
        gpqa_keys,
        arc_c_keys,
        cmtest_keys,
        gsm8k_keys,
        truthfulqa_keys,
        twinviews_keys,
        enronemail_keys,
        darkbench_anthro_keys,
        darkbench_brandbias_keys,
        darkbench_sycophancy_keys,
        darkbench_sneaking_keys,
        darkbench_retention_keys
    ]

    bbq_ambig = BBQ(**bbq_keys)
    salad_inst = SaladBench(**salad_keys)
    # salad_inst = JailbreakV(**jailbreakv_keys)
    alpaca = Alpaca(**alpaca_keys)
    faitheval_counterfactual = FaithEvalCounterfactual(**faitheval_counterfactual_keys)
    faitheval_inconsistent = FaithEvalInconsistent(**faitheval_inconsistent_keys)
    faitheval_unanswerable = FaithEvalUnanswerable(**faitheval_unanswerable_keys)
    precise_wiki = PreciseWiki(**precise_wiki_keys)
    toxigen = ToxiGen(**toxigen_keys)
    # Secondary datasets: no splitting, full dataset as test only
    gpqa = GPQA(**gpqa_keys)
    arc_c = ARC_C(**arc_c_keys)
    cmtest = CMTEST(**cmtest_keys)
    gsm8k = GSM8K(**gsm8k_keys)
    truthfulqa = TruthfulQA(**truthfulqa_keys)
    twinviews = Twinviews(**twinviews_keys)
    enronemail = EnronEmail(**enronemail_keys)
    darkbench_anthro = DarkBenchAnthro(**darkbench_anthro_keys)
    darkbench_brandbias = DarkBenchBrandBias(**darkbench_brandbias_keys)
    darkbench_sycophancy = DarkBenchSynchopancy(**darkbench_sycophancy_keys)
    darkbench_sneaking = DarkBenchSneaking(**darkbench_sneaking_keys)
    darkbench_retention = DarkBenchRetention(**darkbench_retention_keys)

    def is_primary(ds):
        """
        Check if the dataset is a primary dataset (not secondary).
        """
        return ds.__class__.__name__ in [
            "BBQ",
            "SaladBench",
            # "JailbreakV",
            "Alpaca",
            "FaithEvalCounterfactual",
            "FaithEvalInconsistent",
            "FaithEvalUnanswerable",
            "PreciseWiki",
            "ToxiGen"
        ]

    all_datasets = [
        bbq_ambig,
        salad_inst,
        alpaca,
        faitheval_counterfactual,
        faitheval_inconsistent,
        faitheval_unanswerable,
        precise_wiki,
        toxigen,
        # Secondary datasets
        gpqa,
        arc_c,
        cmtest,
        # gsm8k,
        truthfulqa,
        twinviews,
        # enronemail,
        darkbench_anthro,
        darkbench_brandbias,
        darkbench_sycophancy,
        darkbench_sneaking,
        darkbench_retention
    ]

    for ds in all_datasets:
        # shuffle the dataset (skip secondary datasets)
        if is_primary(ds):
            ds.dataset = ds.dataset.shuffle(seed=42)

    # Optionally print dataset sizes
    print("\n--- Dataset Sizes ---")
    for ds in all_datasets:
        print(f"  {ds.name} (split: {ds.split}), Size: {len(ds.dataset)}")
    # Generate per-concept train/val/test splits using fixed per-dataset configuration
    train_split_dict, val_split_dict, test_split_dict = \
        get_dataset_splits_by_concept(
            all_datasets,
            keys_list=keys_list,
            per_dataset_pct=PER_DATASET_CFG,
        )

    ds_size = lambda ds: len(ds.dataset)  # * (2 if ds.paired else 1)  # If paired, we have two examples per input.

    print("\n--- Train Splits ---")
    for concept, ds_list in train_split_dict.items():
        print(f"  Concept: {concept}")
        for ds in ds_list:
            print(f"    - {ds.name} (split: {ds.split}), Size: {ds_size(ds)}")

    print("\n--- Validation Splits ---")
    for concept, ds_list in val_split_dict.items():
        print(f"  Concept: {concept}")
        for ds in ds_list:
            print(f"    - {ds.name} (split: {ds.split}), Size: {ds_size(ds)}")

    print("\n--- Test Splits ---")
    for concept, ds_list in test_split_dict.items():
        print(f"  Concept: {concept}")
        for ds in ds_list:
            print(f"    - {ds.name} (split: {ds.split}), Size: {ds_size(ds)}")
    # Save all splits to disk under one folder per dataset name, with Parquet files
    splits_dir = Path(__file__).parent / "splits"
    splits_dir.mkdir(exist_ok=True)
    from collections import defaultdict
    grouped = defaultdict(dict)
    for split_name, split_dict in [('train', train_split_dict), ('validation', val_split_dict), ('test', test_split_dict)]:
        for concept, ds_list in split_dict.items():
            for ds in ds_list:
                name = type(ds).__name__
                grouped[name][split_name] = ds.dataset
    # Persist each grouped dataset to Parquet
    for name, splits in grouped.items():
        out_dir = splits_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for split_name, hf_ds in splits.items():
            df = hf_ds.to_pandas()
            pq_path = out_dir / f"{split_name}.parquet"
            df.to_parquet(pq_path, index=False)
    print(f"Saved precomputed splits as Parquet under {splits_dir}")
    # Generate README.md for splits directory
    readme_lines = ["---", "configs:"]
    for name in sorted(grouped):
        readme_lines.append(f"  - config_name: {name}")
        readme_lines.append("    data_files:")
        if "train" in grouped[name]:
            readme_lines.append("      - split: train")
            readme_lines.append(f"        path: {name}/train.parquet")
        if "validation" in grouped[name]:
            readme_lines.append("      - split: validation")
            readme_lines.append(f"        path: {name}/validation.parquet")
        if "test" in grouped[name]:
            readme_lines.append("      - split: test")
            readme_lines.append(f"        path: {name}/test.parquet")
    readme_lines.append("---")
    readme_path = splits_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write("\n".join(readme_lines) + "\n")
    print(f"Generated README.md in splits directory at {readme_path}")
    # Optionally push grouped splits directory to Hugging Face Hub
    if args.push_splits_to_hub:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError("Install 'huggingface_hub' to push splits to the Hub.")
        api = HfApi()
        print(f"Pushing grouped splits to Hugging Face under namespace...")
        api.upload_folder(
            repo_id=HF_REPO_ID,
            folder_path=str(splits_dir),
            path_in_repo="",
            repo_type="dataset"
        )
        print("Done pushing grouped splits to Hugging Face Hub.")