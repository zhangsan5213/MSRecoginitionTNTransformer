import sys
import pickle
import argparse
from tqdm import tqdm
# from hgraph import *
# from rdkit import Chem
from multiprocessing import Pool

from transformer.Constants import BOS_WORD, EOS_WORD

def dump_vocab(smiles_file='./data/smiles.txt', vocab_file='./data/vocab.pkl'):
    smiles = open(smiles_file, 'r').read().splitlines()
    unique_chars = set()
    for string in smiles:
        unique_chars.update(set(string))
    unique_chars_list = list(unique_chars)
    unique_chars_list += [BOS_WORD, EOS_WORD]
    unique_chars_list.sort()
    pickle.dump(unique_chars_list, open(vocab_file, 'wb'))

if __name__ == '__main__':
    dump_vocab()

# def process(data):
#     vocab = set()
#     for line in data:
#         s = line.strip("\r\n ")
#         hmol = MolGraph(s)
#         for node,attr in hmol.mol_tree.nodes(data=True):
#             smiles = attr['smiles']
#             vocab.add( attr['label'] )
#             for i,s in attr['inter_label']:
#                 vocab.add( (smiles, s) )
#     return vocab

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ncpu', type=int, default=1)
#     args = parser.parse_args()

#     data = [mol for line in sys.stdin for mol in line.split()[:2]]
#     data = list(set(data))

#     batch_size = len(data) // args.ncpu + 1
#     batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
#     pool = Pool(args.ncpu)
#     # vocab_list = pool.map(process, batches)
#     vocab_list = list(tqdm(pool.imap(process, batches), total=len(batches), ncols=50))

#     vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
#     vocab = list(set(vocab))

#     for x,y in tqdm(sorted(vocab), ncols=50):
#         print(x, y)