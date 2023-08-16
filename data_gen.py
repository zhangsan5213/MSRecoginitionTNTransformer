import os
import re
import random
import pickle
import numpy as np
import argparse
import itertools

from tqdm import tqdm
from copy import deepcopy

def split_smiles_to_vocab(smiles, vocabs):
    if not smiles:
        return [[]]

    valid_splits = []
    for i in range(len(smiles)):
        prefix = smiles[:i+1]
        if prefix in vocabs:
            remaining_string = smiles[i+1:]
            sub_splits = split_smiles_to_vocab(remaining_string, vocabs)
            for sub_split in sub_splits:
                valid_splits.append([prefix] + sub_split)

    return valid_splits

def fab_dataset(
    raw_data_file='./data/raw_ms_smiles_data_trial.pkl',
    vocab_file='./data/vocab.pkl',
    out_file='./data/ms_smiles_dataset.pkl',
    energy_index = 2,
    mz_resolution = 0.5,
    ms_resolution = 1,
    n_max_mz = 700,
    valid_ratio = 0.2
):
    print("## Start loading data.")
    ms_smiles_data = pickle.load(open(raw_data_file, 'rb'))

    vocabs = pickle.load(open(vocab_file, 'rb'))
    vocabs_dict = {v:i for i,v in enumerate(vocabs)}

    print("## Start fabricating dataset.")
    src_mz_list, src_ms_list, tgt_sm_list, n_smiles_characs_max = [], [], [], 0
    for i in tqdm(ms_smiles_data, ncols=80):
        smiles, inchikey, peaks, frags_indices, frags_details, frags_smiles = i
        this_mz, this_ms = np.array(peaks[energy_index]).T

        ## if complicated vocabs
        # valid_splits = split_smiles_to_vocab(smiles, vocabs)
        # for j, valid_split in enumerate(valid_splits):
        #     src_mz_list.append(np.asarray(np.round(this_mz/mz_resolution).flatten(), int).tolist())
        #     src_ms_list.append(np.asarray(np.round(this_ms/ms_resolution).flatten(), int).tolist())
        #     tgt_sm_list.append([vocabs_dict[s] for s in valid_split])

        ## try one character at a time first
        src_mz_list.append(np.asarray(np.round(this_mz/mz_resolution).flatten(), int).tolist())
        src_ms_list.append(np.asarray(np.round(this_ms/ms_resolution).flatten(), int).tolist())
        tgt_sm_list.append([vocabs_dict[s] for s in smiles])

        n_smiles_characs_max = max(n_smiles_characs_max, len(smiles))

    print('## Formulating the dict.')
    data = dict()
    data["dict"], data["train"], data["valid"] = dict(), dict(), dict()

    data["dict"]["src_mz"] = {i/mz_resolution:i for i in range(int(n_max_mz/mz_resolution + 1))}
    data["dict"]["src_ms"] = {i:i for i in range(100+1)}
    data["dict"]["tgt_mz"] = vocabs_dict

    data["settings"] = argparse.Namespace()
    data["settings"].max_token_seq_len = n_smiles_characs_max

    valid_index = int((1-valid_ratio) * len(ms_smiles_data))

    data["train"]["src_mz"] = src_mz_list[:valid_index]
    data["train"]["src_ms"] = src_ms_list[:valid_index]
    data["train"]["tgt_mz"] = tgt_sm_list[:valid_index]

    data["valid"]["src_mz"] = src_mz_list[valid_index:]
    data["valid"]["src_ms"] = src_ms_list[valid_index:]
    data["valid"]["tgt_mz"] = tgt_sm_list[valid_index:]

    pickle.dump(data, open(out_file, "wb"))

if __name__ == '__main__':
    fab_dataset()