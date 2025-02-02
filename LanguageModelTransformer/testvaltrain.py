import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm  # Progress bar for batch processing

# ================================================================
# 🚀 TRAINING FUNCTION WITH EARLY STOPPING, LOSS VISUALIZATION, & PROGRESS BAR
# ================================================================

def train_model(model, train_loader, val_loader, epochs=10, lr=2e-5, patience=3, device="cuda"):
    """
    Trains the Transformer model and tracks performance metrics with a progress bar.
    
    Parameters:
    - model: Transformer model (25B parameters).
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - epochs: Maximum number of epochs to train.
    - lr: Learning rate for AdamW optimizer.
    - patience: Number of epochs to wait before early stopping.
    - device: "cuda" or "cpu".
    
    Returns:
    - Trained model with the best validation performance.
    """
    
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        # Add progress bar for training batches
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False)

        for batch in train_progress:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), batch.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # Update progress bar with current loss
            train_progress.set_postfix({"Batch Loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = validate_model(model, val_loader, loss_fn, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_transformer_model.pth")
            print("Model saved (Best Validation Loss Improved).")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs. Validation Loss Over Epochs')
    plt.legend()
    plt.show()

    return model

# ================================================================
# 🚀 VALIDATION FUNCTION WITH PROGRESS BAR
# ================================================================

def validate_model(model, val_loader, loss_fn, device="cuda"):
    """
    Evaluates the model on validation data and computes key performance metrics.
    
    Parameters:
    - model: Transformer model.
    - val_loader: DataLoader for validation data.
    - loss_fn: Loss function.
    - device: "cuda" or "cpu".
    
    Returns:
    - Average validation loss.
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    val_progress = tqdm(val_loader, desc="Validation", leave=False)  # Progress bar for validation

    with torch.no_grad():
        for batch in val_progress:
            batch = batch.to(device)
            outputs = model(batch)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), batch.view(-1))
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=-1).view(-1).cpu().numpy()
            labels = batch.view(-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

            # Update progress bar with current batch loss
            val_progress.set_postfix({"Batch Loss": loss.item()})

    avg_val_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix - Validation")
    plt.show()

    return avg_val_loss

# ================================================================
# 🚀 TEST FUNCTION WITH PROGRESS BAR
# ================================================================

def test_model(model, test_loader, device="cuda"):
    """
    Evaluates the trained model on test data and computes final metrics.
    
    Parameters:
    - model: Transformer model.
    - test_loader: DataLoader for test data.
    - device: "cuda" or "cpu".
    
    Returns:
    - Perplexity score (lower is better).
    """
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0
    all_preds, all_labels = []

    test_progress = tqdm(test_loader, desc="Testing", leave=False)  # Progress bar for testing

    with torch.no_grad():
        for batch in test_progress:
            batch = batch.to(device)
            outputs = model(batch)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), batch.view(-1))
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=-1).view(-1).cpu().numpy()
            labels = batch.view(-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

            # Update progress bar with current batch loss
            test_progress.set_postfix({"Batch Loss": loss.item()})

    avg_test_loss = total_loss / len(test_loader)
    perplexity = torch.exp(torch.tensor(avg_test_loss))

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Perplexity: {perplexity:.2f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix - Test")
    plt.show()

    return perplexity
