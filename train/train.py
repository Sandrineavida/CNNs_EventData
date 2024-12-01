import time
import torch
from train.validate import validate_epoch
from utils.checkpoint import save_checkpoint
from train.test import test_epoch

def train_epoch(model, train_loader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if num_classes == 1:
            outputs = model(inputs).squeeze(1)
            labels = labels.float()
            loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if num_classes == 1:
            predicted_train = torch.sigmoid(outputs) >= 0.5
        else:
            _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_accuracy = 100 * correct_train / total_train
    return epoch_train_loss, epoch_train_accuracy

def train_model_sheduled(
    model, train_loader, valid_loader, test_loader, criterion, optimizer, scheduler,
    num_epochs=10, patience=7, checkpoint_path='checkpoint.pth', num_classes=1, device="cpu", logger=None
):
    logger.info("\n############################## Training started ##############################")
    if model.quantised:
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)

    model.train()

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    test_losses = []
    test_accuracies = []
    train_times = []
    learning_rates = []


    # list to record saved epochs
    saved_epochs = []

    best_valid_loss = float('inf')  # Initialise the best validation loss
    patience_counter = 0  # Counter for early stopping
    best_model_state = None  # To save the best model state
    stopped_at_epoch = None

    for epoch in range(num_epochs):
        # Record the start time of the epoch
        start_time = time.time()

        # Train for one epoch
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        # Record training time
        end_time = time.time()
        epoch_duration = end_time - start_time
        train_times.append(epoch_duration)

        # Validate for one epoch
        valid_loss, valid_accuracy = validate_epoch(model, valid_loader, criterion, device, num_classes)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        # Save the model if validation loss improves
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch + 1, best_valid_loss)
            saved_epochs.append(epoch + 1)
            logger.info(f"Model saved at epoch {epoch + 1} with validation loss {best_valid_loss:.4f}.")
        else:
            patience_counter += 1

        # Test for one epoch
        test_loss, test_accuracy = test_epoch(model, test_loader, criterion, device, num_classes)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Update scheduler and learning rate
        scheduler.step(valid_loss)
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)

        # Log progress
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.2f}%, "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, "
                f"Train Time: {epoch_duration:.2f}s")

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}. No improvement in validation loss for {patience} consecutive epochs.")
            stopped_at_epoch = epoch + 1
            break

    # Load the best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info("Loaded the best model state based on validation loss.")

    total_train_time = sum(train_times)
    average_train_time = total_train_time / num_epochs


    final_epoch = stopped_at_epoch if stopped_at_epoch is not None else num_epochs
    logger.info(
        f"""
        ############### Training completed in {total_train_time:.2f}s. ###############
        Average time per epoch: {average_train_time:.2f}s.
        Trained for {final_epoch}/{num_epochs} epochs.
        ##############################################################
        """
    )

    logger.info("\n############################## Training completed ##############################")

    return (train_losses, train_accuracies,
            valid_losses, valid_accuracies,
            test_losses, test_accuracies,
            train_times, total_train_time, average_train_time,
            learning_rates, saved_epochs, stopped_at_epoch)

