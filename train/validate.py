import torch

def validate_epoch(model, valid_loader, criterion, device, num_classes):
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

    epoch_valid_loss = running_valid_loss / len(valid_loader)
    epoch_valid_accuracy = 100 * correct_valid / total_valid
    return epoch_valid_loss, epoch_valid_accuracy