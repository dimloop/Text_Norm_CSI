import logging
from typing import Dict
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

def train_siamese_network(dataloaders: Dict[str, DataLoader], model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int = 10, device: torch.device | None = None):
    """
    Trains the Siamese network using the provided DataLoader, model, loss criterion, and optimizer.
    
    Args:
        dataloader (DataLoader): DataLoader for the training data.
        model (nn.Module): The Siamese network model.
        criterion (nn.Module): Loss function to use for training.
        optimizer (optim.Optimizer): Optimizer for updating model weights.
        num_epochs (int): Number of epochs to train the model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Starting training of Siamese network...")
    epochs = [i+1 for i in range(num_epochs)]
    loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    data_dict = {"epochs": epochs, "loss_history": loss_history, "val_loss_history": val_loss_history, "train_accuracy_history": train_accuracy_history, "val_accuracy_history": val_accuracy_history}
    train_loader = dataloaders["train"]
    val_loader = dataloaders["validation"]
    best_val_accuracy = 0.0
    best_model_path = "./best_model.pth"


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct, train_total = 0, 0
        for i, (feature_a, feature_b, label) in enumerate(tqdm(train_loader)):
            feature_a = feature_a.to(device)
            feature_b = feature_b.to(device)
            label     = label.to(device)
            
            optimizer.zero_grad()
            similarity, distance, emb1, emb2 = model(feature_a, feature_b)
            
            loss = criterion(similarity.view(-1), distance.view(-1), label.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            predicted = (similarity < 0.5).float()
            train_correct += (predicted.view(-1) == label.float()).sum().item()
            train_total += label.size(0)
        #print(distance.view(-1), similarity.view(-1), label)
        avg_loss = running_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        loss_history.append(avg_loss)
        train_accuracy_history.append(train_accuracy)
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        
        # Validation phase
        if val_loader is not None and len(val_loader) > 0:
            val_loss, val_accuracy = evaluate_siamese_network(model, val_loader, criterion, device=device)
            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)
            logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), best_model_path)
                print(f'Saved the best model with validation accuracy: {val_accuracy:.2f}%')
    
    data_dict["loss_history"] = loss_history
    data_dict["val_loss_history"] = val_loss_history
    data_dict["train_accuracy_history"] = train_accuracy_history
    data_dict["val_accuracy_history"] = val_accuracy_history

    return model, data_dict
        

def evaluate_siamese_network(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device | None = None):
    """
    Evaluates the Siamese network on the provided DataLoader.
    
    Args:
        model (nn.Module): The Siamese network model.
        dataloader (DataLoader): DataLoader for the evaluation data.
        criterion (nn.Module): Loss function to use for evaluation.
    
    Returns:
        float: Average loss over the evaluation dataset.
        float: Accuracy of the model on the evaluation dataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0
    
    with torch.no_grad():
        for feature_a, feature_b, label in tqdm(dataloader):
            feature_a = feature_a.to(device)
            feature_b = feature_b.to(device)
            label     = label.to(device)

            similarity, distance, emb1, emb2 = model(feature_a, feature_b)
            loss = criterion(similarity.view(-1), distance.view(-1), label.float())

            total_loss += loss.item()
            
            predicted = (similarity < 0.5).float()
            correct += (predicted.view(-1) == label.float()).sum().item()
            total += label.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    logger.info(f"Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy


def test_siamese_network(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device | None = None):
    """
    Evaluates the Siamese network on the provided DataLoader.

    Args:
        model (nn.Module): The Siamese network model.
        dataloader (DataLoader): DataLoader for the evaluation data.
        criterion (nn.Module): Loss function to use for evaluation.

    Returns:
        float: Average loss over the evaluation dataset.
        float: Accuracy of the model on the evaluation dataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.apply(lambda m: m.train() if isinstance(m, torch.nn.BatchNorm1d) else None)

    total_loss = 0.0
    correct, total = 0, 0
    pred = []
    y_pred =[]
    true_val = []

    with torch.no_grad():
        for feature_a, feature_b, label in tqdm(dataloader):
            feature_a = feature_a.to(device)
            feature_b = feature_b.to(device)
            label     = label.to(device)
            true_val.append(label)
            similarity, distance, _, _ = model(feature_a, feature_b)
            loss = criterion(similarity.view(-1), distance.view(-1), label.float())
            total_loss += loss.item()
            predicted = (similarity < 0.5).float()
            #print(distance, similarity)
            #pred.append(distance)
            pred.append(1-similarity)
            correct += (predicted.view(-1) == label.float()).sum().item()
            total += label.size(0)
    y_pred.append(torch.cat(pred, dim=0).to('cpu').view(-1))
    y_true = torch.cat(true_val, dim=0).to('cpu')
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    logger.info(f"Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return y_true, y_pred[0]