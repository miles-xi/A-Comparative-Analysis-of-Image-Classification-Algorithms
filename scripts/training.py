'''
2 training - train a specific model, given data (different batch sizes, preprocessing steps), # epochs, and the learning rate
'''

def trainmodels(data_loader_train, data_loader_valid, model, learning_rate=0.001, num_epochs=20):
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from sklearn.metrics import accuracy_score
    from tqdm import tqdm
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    model.train()
    losses_train = []
    accus_train = []
    losses_valid = []
    accus_valid = []
    runtime = []

    for epoch in range(num_epochs):
        runningloss_train = 0
        preds_train = []
        labels_train = []
        runningloss_valid = 0
        preds_valid = []
        labels_valid = []
        time_start = time.perf_counter()

        for images, labels in tqdm(data_loader_train):
            images, labels = images.to(device), labels.to(device)
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            runningloss_train += loss.item()
            _, preds = torch.max(outputs, 1)
            preds_train.extend(preds.cpu().numpy())
            labels_train.extend(labels.cpu().numpy())
        
        # for this epoch, compute and print training loss and accuracy
        loss_epoch = runningloss_train / len(data_loader_train)
        accu_epoch = accuracy_score(labels_train, preds_train)
        print(f"Epoch {epoch} of {num_epochs}")
        print(f"Training loss: {loss_epoch:.4f}, accuracy: {accu_epoch:.4f}")

        # log these metrics, append to lists
        losses_train.append(loss_epoch)
        accus_train.append(accu_epoch)

        # evaluate and log trained model performance on a validation set
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(data_loader_valid):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                runningloss_valid += loss.item()
                _, preds = torch.max(outputs, 1)
                preds_valid.extend(preds.cpu().numpy())
                labels_valid.extend(labels.cpu().numpy())

            loss_epoch = runningloss_valid / len(data_loader_valid)
            accu_epoch = accuracy_score(labels_valid, preds_valid)
            print(f"Validation loss: {loss_epoch:.4f}, accuracy: {accu_epoch:.4f}", end="\n\n")
            losses_valid.append(loss_epoch)
            accus_valid.append(accu_epoch)
        
        runtime.append(time.perf_counter() - time_start)
    
    # save the model
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch':epoch,
    }, model.__class__.__name__ +'.pth')    

    return losses_train, accus_train, losses_valid, accus_valid, runtime

def save(results, model_name):
    import pandas as pd
    names = ['losses_train', 'accus_train', 'losses_valid', 'accus_valid', 'runtime']
    
    for result, name in zip(results, names):
        pd.DataFrame(result).to_csv(f"{name}_{model_name}.csv")