import os
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import itertools

def abide_stratified_kfold(data_path: str, kfold: int, output_dir: str, signal_dir: str):
    """
    Stratified split data into kfold according to testing sites and label

    Args:
        data_path (str): csv that contains all data
        kfold (int): fold number
        output_dir (str): directory to save splited data
        signal_dir (str): downloaded signal directory, for fitlering out no signal data
    """
    # signal_ids = [int(f.split('_')[-3]) for f in os.listdir(signal_dir)]
    signal_ids = [int(f.split('.')[0]) for f in os.listdir(signal_dir)]
    print(f"Got {len(signal_ids)} signal")

    pheno = pd.read_csv(data_path)
    pheno["split_tag"] = pheno[["SITE_ID", "DX_GROUP", "SEX"]].apply(lambda x: "_".join([str(s) for s in x]), axis=1)
    print(f"Got {len(pheno)} phenotypes before filtering")
    
    pheno = pheno[pheno["SUB_ID"].isin(signal_ids)].reset_index(drop=True)
    print(f"Got {len(pheno)} samples after filtering")

    ids = pheno["SUB_ID"]

    os.makedirs(output_dir, exist_ok=True)
    pheno.to_csv(os.path.join(output_dir, "all.csv"), index=False)
    
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
    chunks = [test_idx for _, (_, test_idx) in enumerate(skf.split(ids, pheno["split_tag"]))]

    for i in range(kfold):
        tst_chunk = [i]
        # val_chunk = [(i+1) % (kfold-1), (i+2) % (kfold-1)]
        val_chunk = [(i+1) % (kfold-1)]
        trn_chunk = [idx for idx in range(kfold) if idx not in tst_chunk + val_chunk]

        train_index = list(itertools.chain.from_iterable([chunks[idx] for idx in trn_chunk]))
        valid_index = list(itertools.chain.from_iterable([chunks[idx] for idx in val_chunk]))
        test_index = list(itertools.chain.from_iterable([chunks[idx] for idx in tst_chunk]))

        fold_dir = os.path.join(output_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)

        trn = pheno[pheno["SUB_ID"].isin([i for i in ids[train_index]])]
        trn.to_csv(os.path.join(fold_dir, "train.csv"), index=False)

        val = pheno[pheno["SUB_ID"].isin([i for i in ids[valid_index]])]
        val.to_csv(os.path.join(fold_dir, "valid.csv"), index=False)

        tst = pheno[pheno["SUB_ID"].isin([i for i in ids[test_index]])]
        tst.to_csv(os.path.join(fold_dir, "test.csv"), index=False)

        print(f"Fold {i} splited with {len(trn)} train, {len(val)} valid, {len(tst)} test")

    print(f"Split succeed! Splitted data save to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified k-fold splitting for ABIDE dataset")

    parser.add_argument("--data_path", type=str, default="./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv",
                        help="Path to the csv file containing all data")
    
    parser.add_argument("--kfold", type=int, default=5,
                        help="Number of folds for stratified k-fold splitting")
    
    parser.add_argument("--output_dir", type=str, default="./data/split/default",
                        help="Output directory to save split data")

    parser.add_argument("--signal_dir", type=str, default="./feature/rois_cc200/raw",
                        help="Downloaded signal directory, for fitlering out no signal data")
    
    args = parser.parse_args()
    abide_stratified_kfold(
        data_path=args.data_path,
        kfold=args.kfold,
        output_dir=args.output_dir,
        signal_dir=args.signal_dir
    )
