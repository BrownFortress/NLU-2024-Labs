# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from model import *
from utils import *
from tqdm import tqdm

import copy
import numpy as np
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader
import wandb
import os

# Set your Wandb token
wandb_token = os.environ["WANDB_TOKEN"]

# Login to wandb
wandb.login(key=wandb_token)

# Initialize wandb
wandb.init(project="nlu-assignmet1-part2", allow_val_change=True)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
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
    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device))

    # hyperparams
    PATIENCE = 6
    emb_size = 600
    hid_size = 600
    lr = 4.0
    clip = 5 # clip the gradient
    vocab_len = len(lang.word2id)
    drop_p = 0.2
    drop_k = 0.2
    n_layers = 1

    model = LM_LSTM(emb_size=emb_size,
                    hidden_size=hid_size,
                    output_size=vocab_len,
                    n_layers=n_layers,
                    pad_index=lang.word2id["<pad>"],
                    dropout_p=drop_p,
                    dropout_k=drop_k
                    ).to(device)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    # hyperparams
    n_epochs = 200
    patience = PATIENCE
    losses_train = []
    ppls_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))

    ntasgd_interval = 5 # non-monotone interval - # of epochs of non-improving valid loss after which NT-ASGD is triggered
    ntasgd_trigger = False # ntasgd_trigger (bool): Indicates whether NT-ASGD has been triggered.
    asgd_lr = 0.13
    w_decay = 1e-6

    #wandb: Define your config
    config = wandb.config
    config.epochs = n_epochs
    config.learning_rate = lr
    config.asgd_lr = asgd_lr
    config.batch_size = BATCH_SIZE
    config.emb_size = emb_size
    config.hidden_size = hid_size
    config.dropout_emb = drop_p
    config.dropout_lstm = drop_k
    config.weight_decay = w_decay
    config.num_lstm_layers = n_layers
    config.ntasgd_interval = ntasgd_interval
    config.momentum = 0.9
    print(config)

    # params in the paper
    # `t` is a counter for the number of epochs, after each epoch is executed it is incremented
    # `T` is simply the triggering criterion which is updated after every epoch if the perplexity
    # obtained is smaller than what we have stored in the last [:-n] logs (without considering last n)
    # In this code the flag `ntasgd_trigger` is equivalent to `T` in the paper algorithm.

    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            # Log training loss to wandb
            wandb.log({"train_loss": np.asarray(loss).mean()})
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            ppls_dev.append(np.asarray(ppl_dev).mean())
            # Log validation loss to wandb
            wandb.log({"val_loss": np.asarray(loss_dev).mean()})
            wandb.log({"val PPL": np.asarray(ppl_dev).mean()})
            pbar.set_description("PPL: %f" % ppl_dev)

            # Check non-monotone criterion for NT-ASGD
            if not ntasgd_trigger and epoch > ntasgd_interval:
                # as long as the trigger criterion is not met,
                # we compute the gradient and apply SGD to update the weights
                # "a non-monotonic criterion that conservatively triggers the
                # averaging when the validation metric fails to improve for multiple cycles"

                if not ntasgd_trigger and ppl_dev > min(ppls_dev[:-ntasgd_interval]):
                    print("switching to ASGD")
                    ntasgd_trigger = True
                    optimizer = torch.optim.ASGD(
                                    model.parameters(),
                                    lr=asgd_lr,
                                    t0=0,
                                    lambd=0.,
                                    weight_decay=w_decay
                                    )
                    # config.update({"learning_rate": asgd_lr}, allow_val_change=True)
                    # print(config)
            # if non-monotone criterion for NT-ASGD never triggers we use patience
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu') # save to cpu memory
                patience = PATIENCE # reset patience
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                print("Early stopping with patience")
                print(f"ASGD flag={ntasgd_trigger}")
                break

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)

    # To save the model
    path = 'bin/best_model.pt'
    torch.save(model.state_dict(), path)
    # To load the model you need to initialize it
    # model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    # Then you load it
    # model.load_state_dict(torch.load(path))