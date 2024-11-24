import matplotlib.pyplot as plt

def plot_frames(frames):

    fig, axes = plt.subplots(1, len(frames), figsize=(10, 5))
    for axis, frame in zip(axes, frames):
        axis.imshow(frame[1] - frame[0])
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def plot_learning_curves(
        num_epochs, train_losses, train_accuracies, valid_losses, valid_accuracies,
        test_losses, test_accuracies, epoch_times, learning_rates, saved_epochs, stopped_at_epoch=None
):
    """
    Plot the training, validation, and test metrics along with learning rate and inference times.

    Parameters:
    - num_epochs: Total number of epochs.
    - train_losses: List of training losses for each epoch.
    - train_accuracies: List of training accuracies for each epoch.
    - valid_losses: List of validation losses for each epoch.
    - valid_accuracies: List of validation accuracies for each epoch.
    - test_losses: List of test losses for each epoch.
    - test_accuracies: List of test accuracies for each epoch.
    - epoch_times: List of training times for each epoch.
    - valid_inference_times: List of validation inference times for each epoch.
    - test_inference_times: List of test inference times for each epoch.
    - learning_rates: List of learning rates for each epoch.
    - saved_epochs: List of epochs where the model was saved.
    - stopped_at_epoch: The epoch at which training stopped early (optional).
    """
    # Determine the range of epochs, considering early stopping
    if stopped_at_epoch:
        epochs = range(1, stopped_at_epoch + 1)
    else:
        epochs = range(1, num_epochs + 1)

    # Truncate all data lists to match the length of epochs
    max_len = len(epochs)
    train_losses = train_losses[:max_len]
    train_accuracies = train_accuracies[:max_len]
    valid_losses = valid_losses[:max_len]
    valid_accuracies = valid_accuracies[:max_len]
    test_losses = test_losses[:max_len]
    test_accuracies = test_accuracies[:max_len]
    epoch_times = epoch_times[:max_len]
    learning_rates = learning_rates[:max_len]

    # Plotting
    plt.figure(figsize=(18, 20))

    # Plot training, validation, and test loss
    plt.subplot(4, 2, 1)
    plt.plot(epochs, train_losses, '-o', label='Train Loss')
    plt.plot(epochs, valid_losses, '-o', label='Valid Loss')
    plt.plot(epochs, test_losses, '-o', label='Test Loss')
    # Highlight saved epochs
    for e in saved_epochs:
        plt.scatter(e, valid_losses[e - 1], color='purple', marker='^', s=120,
                    label=f'Saved Epoch' if e == saved_epochs[0] else "")
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training, validation, and test accuracy
    plt.subplot(4, 2, 2)
    plt.plot(epochs, train_accuracies, '-o', label='Train Accuracy')
    plt.plot(epochs, valid_accuracies, '-o', label='Valid Accuracy')
    plt.plot(epochs, test_accuracies, '-o', label='Test Accuracy')
    # Highlight saved epochs
    for e in saved_epochs:
        plt.scatter(e, valid_accuracies[e - 1], color='purple', marker='^', s=120,
                    label=f'Saved Epoch' if e == saved_epochs[0] else "")
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Plot training times
    plt.subplot(4, 2, 3)
    plt.plot(epochs, epoch_times, '-o', label='Training Time')
    plt.title('Training Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()

    # Plot learning rate schedule
    plt.subplot(4, 2, 4)
    plt.plot(epochs, learning_rates, '-o', label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.tight_layout()
    plt.show()
