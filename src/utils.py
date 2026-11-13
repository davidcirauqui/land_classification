import time
import pandas as pd
import torch


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """Function that trains the model for a single epoch, passing over the training data and
    updating its weights after every batch

    inputs:
        - model: model to be trained
        - dataloader: DataLoader of the training data
        - loss_fn: loss function to be minimised
        - optimizer: optimizer updating the model's weights
        - device: device on which the model is trained ("cuda" or "cpu")
    outputs:
        - mean loss over the epoch
        - accuracy over the epoch
    """


    model.train()
    total_loss = 0.0
    Ncorrect = 0
    Ntotal = 0

    for X, y in dataloader:
        # move data to device
        X, y = X.to(device), y.to(device)

        # forward pass
        logits = model(X)

        # compute loss
        loss = loss_fn(logits, y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

        # compute figures of merit
        total_loss += loss.item() * X.shape[0]
        Ncorrect += (logits.argmax(dim=1) == y).sum().item()
        Ntotal += X.shape[0]

    return total_loss / Ntotal, Ncorrect / Ntotal


def evaluate(model, dataloader, loss_fn, device):
    """Function that evaluates the model on a given dataset (validation or test)

    inputs:
        - model: model to be evaluated
        - dataloader: DataLoader of the data on which the model is evaluated (validation or test)
        - loss_fn: loss function
        - device: device on which the model is evaluated ("cuda" or "cpu")
    outputs:
        - mean loss over the dataset
        - accuracy over the dataset
    """


    model.eval()
    total_loss = 0.0
    Ncorrect = 0
    Ntotal = 0

    with torch.no_grad():
        for X, y in dataloader:
            # move data to device
            X, y = X.to(device), y.to(device)

            # forward pass
            logits = model(X)

            # compute loss
            loss = loss_fn(logits, y)

            # compute figures of merit
            total_loss += loss.item() * X.shape[0]
            Ncorrect += (logits.argmax(dim=1) == y).sum().item()
            Ntotal += X.shape[0]

    return total_loss / Ntotal, Ncorrect / Ntotal


def fit(model, train_dataloader, val_dataloader, loss_fn, n_epochs, lr, device,
        cosine_schedule = True, ckpt_path = None, verbose = False):
    """Function that trains a model with an Adam optimizer for a given number of epochs, evaluating
    it on the validation set after each one. Only parameters with requires_grad = True are
    trained, so frozen layers stay untouched

    inputs:
        - model: model to be trained
        - train_dataloader: DataLoader of the training data
        - val_dataloader: DataLoader of the validation data
        - loss_fn: loss function to be minimised
        - n_epochs: number of epochs to train
        - lr: (initial) learning rate
        - device: device on which the model is trained ("cuda" or "cpu")
        - cosine_schedule: if True, the learning rate follows a cosine decay from lr towards 0.
        If False, it stays constant
        - ckpt_path: path where the weights of the best validation-accuracy epoch are saved.
        If None, no weights are saved
        - verbose: if True, prints the learning rate, losses and accuracies of each epoch
    outputs:
        - metrics: DataFrame indexed by epoch with columns: lr, train_loss, train_acc, val_loss,
        val_acc, seconds 
    """


    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
                 if cosine_schedule else None)

    metrics = []
    best_acc = 0.0
    for epoch in range(n_epochs):
        t_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_dataloader, loss_fn, device)

        # read the lr from the optimizer: works with or without a scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        metrics.append({"epoch": epoch,
                        "lr": current_lr,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "seconds": time.perf_counter() - t_start})

        # keep the weights of the best validation epoch
        if val_acc > best_acc:
            best_acc = val_acc
            if ckpt_path is not None:
                torch.save(model.state_dict(), ckpt_path)

        if verbose:
            print(f"{epoch:2d}:   lr {current_lr:.1e},   train {train_loss:.3f}/{train_acc:.3f},   validation {val_loss:.3f} / {val_acc:.3f}")

    return pd.DataFrame(metrics).set_index("epoch")
