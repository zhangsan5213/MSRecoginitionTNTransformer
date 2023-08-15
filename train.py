import argparse
import math
import time
import numpy as np
from tqdm import tqdm
import pickle
import torch 
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from dataset import MZMSDataset, paired_collate_fn
from transformer.ModelsDoubleChannelFuse import Tensorized_T
from transformer.Optim import ScheduledOptim

from data_ms import fab_raw_data_file
from data_get_vocab import dump_vocab
from data_gen import fab_dataset

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    print("- (Training)")
    for batch in tqdm(training_data, ncols=50):

        # prepare data
        src_seq_mz, src_pos_mz, src_seq_ms, src_pos_ms, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src_seq_mz, src_pos_mz, src_seq_ms, src_pos_ms, tgt_seq, tgt_pos)

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        print("- (Validation)")
        for batch in tqdm(validation_data, ncols=50):

            # prepare data
            src_seq_mz, src_pos_mz, src_seq_ms, src_pos_ms, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq_mz, src_pos_mz, src_seq_ms, src_pos_ms, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.save_model:
        log_train_file = opt.save_model + '.train.log'
        log_valid_file = opt.save_model + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer, device, smoothing=opt.label_smoothing)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                ppl=math.exp(min(train_loss, 100)),
                accu=100*train_accu,
                elapse=(time.time()-start)/60)
              )

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                ppl=math.exp(min(valid_loss, 100)),
                accu=100*valid_accu,
                elapse=(time.time()-start)/60)
              )

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{train_accu:3.3f}_{valid_accu:3.3f}.chkpt'.format(train_accu=100*train_accu, valid_accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        MZMSDataset(
            src_word2idx_mz=data['dict']['src_mz'],
            src_word2idx_ms=data['dict']['src_ms'],
            tgt_word2idx=data['dict']['tgt_mz'],
            src_insts_mz=data['train']['src_mz'],
            src_insts_ms=data['train']['src_ms'],
            tgt_insts=data['train']['tgt_mz']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        MZMSDataset(
            src_word2idx_mz=data['dict']['src_mz'],
            src_word2idx_ms=data['dict']['src_ms'],
            tgt_word2idx=data['dict']['tgt_mz'],
            src_insts_mz=data['valid']['src_mz'],
            src_insts_ms=data['valid']['src_ms'],
            tgt_insts=data['valid']['tgt_mz']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    return train_loader, valid_loader


if __name__ == '__main__':
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--fab_dataset', type=int, default=0)
    parser.add_argument('--shuffle_dataset', type=int, default=0)
    parser.add_argument('--data_path_name', default='./data/ms_smiles_dataset.pkl')
    parser.add_argument('--load_from_model', type=str, default='')
    parser.add_argument('--save_model', type=str, default='./model/test')
    parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='all')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_word_vec', type=int, default=512)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_inner_hid', type=int, default=512)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_encoder_layers', type=int, default=8)
    parser.add_argument('--n_decoder_layers', type=int, default=8)
    parser.add_argument('--n_warmup_steps', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embs_share_weight', action='store_true')
    parser.add_argument('--proj_share_weight', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--label_smoothing', action='store_true')
    parser.add_argument('--Tensorized_transformer', action='store_false')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    #========= Fabricating Dataset =========#
    if opt.fab_dataset:
        fab_raw_data_file()
        dump_vocab()
        fab_dataset()

    #========= Loading Dataset =========#
    loaded_data = pickle.load(open("./data/ms_smiles_dataset.pkl", "rb"))

    if opt.shuffle_dataset:
        dataset_size, train_size = len(loaded_data['train']['src_mz'] + loaded_data['valid']['src_mz']), len(loaded_data['train']['src_mz'])
        src_mz = loaded_data['train']['src_mz'] + loaded_data['valid']['src_mz']
        src_ms = loaded_data['train']['src_ms'] + loaded_data['valid']['src_ms']
        tgt_mz = loaded_data['train']['tgt_mz'] + loaded_data['valid']['tgt_mz']

        new_train = np.full((dataset_size,), False)
        new_train[np.random.choice(
            [i for i in range(dataset_size)],
            train_size,
            replace=False)
        ] = True

        loaded_data['train']['src_mz'], loaded_data['train']['src_ms'], loaded_data['train']['tgt_mz'] = [], [], []
        loaded_data['valid']['src_mz'], loaded_data['valid']['src_ms'], loaded_data['valid']['tgt_mz'] = [], [], []

        for i, flag in enumerate(new_train):
            if flag == True:
                loaded_data['train']['src_mz'].append(src_mz[i])
                loaded_data['train']['src_ms'].append(src_ms[i])
                loaded_data['train']['tgt_mz'].append(tgt_mz[i])
            else:
                loaded_data['valid']['src_mz'].append(src_mz[i])
                loaded_data['valid']['src_ms'].append(src_ms[i])
                loaded_data['valid']['tgt_mz'].append(tgt_mz[i])

    opt.max_token_seq_len = loaded_data['settings'].max_token_seq_len
    training_data, validation_data = prepare_dataloaders(loaded_data, opt)
    opt.src_mz_size = training_data.dataset.src_mz_size
    opt.src_ms_size = training_data.dataset.src_ms_size
    opt.tgt_mz_size = training_data.dataset.tgt_mz_size

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    device = torch.device('cuda' if opt.cuda else 'cpu')
    # device = torch.device('cpu')
    torch.manual_seed(42)
    
    transformer = Tensorized_T(
        opt.src_mz_size, ## size of the dictionary, i.e. number of m/z. Max=700, so this gives us 70000 embeddings with round to 0.01.
        opt.src_ms_size, ## size of the dictionary, i.e. number of intensities. 100 -> 100.
        opt.tgt_mz_size, ## size of the dictionary, i.e. number of m/z. Max=700, so this gives us 70000 embeddings with round to 0.01.
        opt.max_token_seq_len, ## max length of the sequence
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_encoder_layers=opt.n_encoder_layers,
        n_decoder_layers=opt.n_decoder_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)
    if opt.load_from_model:
        print("Model weight loaded from", opt.load_from_model)
        transformer.load_state_dict(torch.load(opt.load_from_model)["model"])
    else:
        print("Start training from raw model.")

    opt.n_all_param = sum([p.nelement() for p in transformer.parameters()])
    print('#params = {}'.format(opt.n_all_param))

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            lr=8e-5,
            betas=(0.9, 0.98),
            eps=1e-8),
        opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt)