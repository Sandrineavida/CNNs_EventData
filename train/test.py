import torch

def test_epoch(model, test_loader, criterion, device, num_classes):
    model.eval()
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

    epoch_test_loss = running_test_loss / len(test_loader)
    epoch_test_accuracy = 100 * correct_test / total_test
    return epoch_test_loss, epoch_test_accuracy