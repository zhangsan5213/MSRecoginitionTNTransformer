import numpy as np
import torch
import torch.utils.data

from transformer import Constants

def paired_collate_fn(insts):
    src_insts_mz, src_insts_ms, tgt_insts = list(zip(*insts))
    src_insts_mz = collate_fn(src_insts_mz)
    src_insts_ms = collate_fn(src_insts_ms)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts_mz, *src_insts_ms, *tgt_insts)

def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    batch_pos = np.array([
        [pos_i+1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])

    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos

class MZMSDataset(torch.utils.data.Dataset):
    def __init__(
        self, src_word2idx_mz, src_word2idx_ms, tgt_word2idx,
        src_insts_mz=None, src_insts_ms=None, tgt_insts=None):
        ## src: word float -> idx integer as float rounded to 0.01
        ## tgt: word int -> idx int, mapped to itself

        assert src_insts_mz
        assert src_insts_ms
        assert not tgt_insts or (len(src_insts_mz) == len(tgt_insts)) or (len(src_insts_ms) == len(tgt_insts))

        src_idx2word_mz = {idx:word for word, idx in src_word2idx_mz.items()}
        self._src_word2idx_mz = src_word2idx_mz
        self._src_idx2word_mz = src_idx2word_mz
        self._src_insts_mz = src_insts_mz

        src_idx2word_ms = {idx:word for word, idx in src_word2idx_ms.items()}
        self._src_word2idx_ms = src_word2idx_ms
        self._src_idx2word_ms = src_idx2word_ms
        self._src_insts_ms = src_insts_ms

        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word
        self._tgt_insts = tgt_insts

    @property
    def src_mz_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx_mz)
    
    @property
    def src_ms_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx_ms)

    @property
    def tgt_mz_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx_mz(self):
        ''' Property for word dictionary '''
        return self._src_word2idx_mz
    @property
    def src_word2idx_ms(self):
        ''' Property for word dictionary '''
        return self._src_word2idx_ms

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word_mz(self):
        ''' Property for index dictionary '''
        return self._src_idx2word_mz
    @property
    def src_idx2word_ms(self):
        ''' Property for index dictionary '''
        return self._src_idx2word_ms

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        ''' the same as src_insts_ms, so does not matter '''
        return len(self._src_insts_mz)

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._tgt_insts:
            return self._src_insts_mz[idx], self._src_insts_ms[idx], self._tgt_insts[idx]
        return self._src_insts_mz[idx], self._src_insts_ms[idx]
