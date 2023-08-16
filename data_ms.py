import os
import re
import pickle
import numpy as np

from tqdm import tqdm

def read_cfmid_ms_file(path):
    lines = open(path, "r").readlines()
    # for line in lines: print(line)

    peaks, frags_indices, frags_details, frags_smiles = [[], [], []], [[], [], []], [[], [], []], []
    flag_energy, flag_smiles = 0, 0

    smiles, inchikey = lines[2].strip('#SMILES=').strip(), lines[3].strip('#InChiKey=').strip()
    
    for line in lines:
        # print(line)
        if line == "\n":
            flag_smiles = 1
            continue
        if line.startswith("energy"):
            energy_index = int(line[-2])
            flag_energy = 1
            continue
        if (flag_energy == 1) and (flag_smiles == 0):
            temp = line.split()
            in_bracket = re.findall(re.compile(r'[(](.*?)[)]', re.S), line)[0].split()
            peaks[energy_index].append([float(i) for i in temp[0:2]])
            frags_indices[energy_index].append([int(i) for i in temp[2:2+len(in_bracket)]])
            frags_details[energy_index].append([float(i) for i in in_bracket])
        if (flag_energy == 1) and (flag_smiles == 1):
            temp = line.split()
            frags_smiles.append([int(temp[0]), float(temp[1]), temp[2]])
        
    return lines, smiles, inchikey, peaks, frags_indices, frags_details, frags_smiles

def fab_raw_data_file(file_name="./data/raw_ms_smiles_data.pkl", num=None):
    cfmid_ms_path = "/home/lrl/dataset/zinc_mirror_ms/"
    cfmid_ms_files = [cfmid_ms_path + p for p in os.listdir(cfmid_ms_path)]

    peaks_cache = []
    smiles_file = open('./data/smiles.txt', 'w+')

    if num:
        count = 0
        for cfmid_ms_file in tqdm(cfmid_ms_files, ncols=80):
            _, smiles, inchikey, peaks, frags_indices, frags_details, frags_smiles = read_cfmid_ms_file(cfmid_ms_file)
            peaks_cache.append([smiles, inchikey, peaks, frags_indices, frags_details, frags_smiles])
            smiles_file.write(smiles + '\n')

            count += 1
            if count == num:
                break
    else:
        for cfmid_ms_file in tqdm(cfmid_ms_files, ncols=80):
            _, smiles, inchikey, peaks, frags_indices, frags_details, frags_smiles = read_cfmid_ms_file(cfmid_ms_file)
            peaks_cache.append([smiles, inchikey, peaks, frags_indices, frags_details, frags_smiles])
            smiles_file.write(smiles + '\n')

    pickle.dump(peaks_cache, open(file_name, 'wb'))
    smiles_file.close()

if __name__ == '__main__':
    fab_raw_data_file(file_name='./data/raw_ms_smiles_data_trial.pkl', num=10000)