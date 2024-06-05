# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *
from tqdm import tqdm

import copy
import argparse
import numpy as np
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader


def init_args():
    parser = argparse.ArgumentParser(description="Next token prediction task arguments.")
    parser.add_argument("-mode", type=str, default='eval', help="Model mode: train or eval")
    return parser

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'The device selected: {device}')
    model_path = 'bin/best_model.pt'


    # load data
    parser = init_args()
    mode = parser.parse_args().mode
    print(f'Running script in mode: {mode}. If you desire to change it use the -mode argument, i.e. python main.py -mode eval')

    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    # process vocab
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    # construct dataset
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Dataloader instantiation
    # You can reduce the batch_size if the GPU memory is not enough
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device))


    # hyperparams
    hid_size = 200
    emb_size = 300
    lr = 1e-3
    clip = 5 #clip the gradient

    vocab_len = len(lang.word2id)

    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"])
    model.to(device)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    # Training loop
    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    #If the PPL is too high try to change the learning rate

    if mode == 'train':
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description("PPL: %f" % ppl_dev)
                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    model_info = {'state_dict': model.state_dict(), 'lang':lang}
                    torch.save(model_info, model_path)
                    patience = 3
                else:
                    patience -= 1

                if patience <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean
    else:
        print("*You are in evaluation mode*")
        # Load model
        checkpoint = torch.load(model_path)
        lang = checkpoint['lang']
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)

        print('Test PPL: ', final_ppl)