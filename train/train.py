import time
import torch

def train_model_sheduled(
    model, train_loader, valid_loader, test_loader, criterion, optimizer, scheduler,
    num_epochs=10, patience=7, checkpoint_path='checkpoint.pth', num_classes=1, device="cpu"
):
    model.train()

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    test_losses = []
    test_accuracies = []
    epoch_times = []
    learning_rates = []

    # list to record saved epochs
    saved_epochs = []

    best_valid_loss = float('inf')  # Initialize the best validation loss
    patience_counter = 0  # Counter for early stopping
    best_model_state = None  # To save the best model state
    stopped_at_epoch = None

    for epoch in range(num_epochs):
        # Record the start time of the epoch
        start_time = time.time()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Iterate through training batches
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            if num_classes == 1:
                outputs = model(inputs).squeeze(1)
                labels = labels.float()
                loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass and optimisation
            loss.backward()
            optimizer.step()

            # Record loss
            running_loss += loss.item()

            # Compute training accuracy
            if num_classes == 1:
                predicted_train = torch.sigmoid(outputs) >= 0.5
            else:
                _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        # Compute and store average training loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Record training time
        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)

        # Evaluate on validation set
        model.eval()
        running_valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if num_classes == 1:
                    outputs = model(inputs).squeeze(1)
                    labels = labels.float()
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_valid_loss += loss.item()

                if num_classes == 1:
                    predicted_valid = torch.sigmoid(outputs) >= 0.5
                else:
                    _, predicted_valid = torch.max(outputs.data, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted_valid == labels).sum().item()


        # Compute and store validation loss and accuracy
        valid_loss = running_valid_loss / len(valid_loader)
        valid_losses.append(valid_loss)
        valid_accuracy = 100 * correct_valid / total_valid
        valid_accuracies.append(valid_accuracy)

        # Save the model if validation loss improves
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Save the best model state
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'valid_loss': best_valid_loss,
            }, checkpoint_path)
            saved_epochs.append(epoch + 1)  # Append the epoch number
            print(f"Model saved at epoch {epoch + 1} with validation loss {best_valid_loss:.4f}.")
        else:
            patience_counter += 1

        # Evaluate on test set
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if num_classes == 1:
                    outputs = model(inputs).squeeze(1)
                    labels = labels.float()
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()

                if num_classes == 1:
                    predicted_test = torch.sigmoid(outputs) >= 0.5
                else:
                    _, predicted_test = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted_test == labels).sum().item()

        # Compute and store test loss and accuracy
        test_loss = running_test_loss / len(test_loader)
        test_losses.append(test_loss)
        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)

        # Update learning rate scheduler
        scheduler.step(valid_loss)
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)

        # Log progress
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, "
              f"Train Time: {epoch_duration:.2f}s")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}. No improvement in validation loss for {patience} consecutive epochs.")
            stopped_at_epoch = epoch + 1
            break

    # Load the best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded the best model state based on validation loss.")

    return train_losses, train_accuracies, valid_losses, valid_accuracies, test_losses, test_accuracies, epoch_times, learning_rates, saved_epochs, stopped_at_epoch