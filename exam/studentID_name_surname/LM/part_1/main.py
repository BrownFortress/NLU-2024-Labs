# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *
from tqdm import tqdm

import os
import argparse
import numpy as np
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader

torch.manual_seed(0)

def init_args():
    parser = argparse.ArgumentParser(description="Next token prediction task arguments.")
    parser.add_argument("--mode", type=str, default='eval', help="Model mode: train or eval")
    parser.add_argument("--use_wandb", type=str, default='true', help="Use wandb to log training results.")
    return parser

def init_wandb():
    # Set your Wandb token
    wandb_token = os.environ["WANDB_TOKEN"]

    # # Login to wandb
    wandb.login(key=wandb_token)

    # # Initialize wandb
    wandb.init(project="nlu-assignmet1-part1", allow_val_change=True)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'The device selected: {device}')


    # load data
    parser = init_args()
    model_path = 'bin/best_model.pt'

    mode = parser.parse_args().mode
    use_wandb = parser.parse_args().use_wandb
    if mode == 'train' and use_wandb == 'true':
        import wandb
        init_wandb()

    print(f'Running script in mode: {mode}. If you desire to change it use the --mode argument, i.e. python main.py --mode train')

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
    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE*2, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device))


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

    config = None
    if mode == 'train' and use_wandb == 'true':
        #wandb: Define your config
        config = wandb.config
        config.epochs = n_epochs
        config.learning_rate = lr
        config.batch_size = BATCH_SIZE
        config.emb_size = emb_size
        config.hidden_size = hid_size
        print(config)


    if mode == 'train':
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                if use_wandb == 'true':
                    wandb.log({"train_loss": np.asarray(loss).mean()})
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                if use_wandb == 'true':
                    # Log validation loss to wandb
                    wandb.log({"val_loss": np.asarray(loss_dev).mean()})
                    wandb.log({"val PPL": np.asarray(ppl_dev).mean()})
                pbar.set_description("PPL: %f" % ppl_dev)
                
                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
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
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, model)

        print('Test PPL: ', final_ppl)