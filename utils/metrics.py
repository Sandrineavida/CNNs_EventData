import sys
import datetime
import torch
from utils.datasets import get_test_dataloader_batch_size_eq_1
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def get_inference_time(model, test_data_path, num_tests=5, logger=None):
    model.eval()
    logger.info(f"\n##################### [Inference time] - Testing model for {num_tests} iterations #####################")
    test_dataloader = get_test_dataloader_batch_size_eq_1(test_data_path)

    total_inference_time = datetime.timedelta()
    num_classes = model.num_classes

    for test_iteration in range(num_tests):
        logger.info(f"\n### Testing - Iteration {test_iteration + 1}/{num_tests} ###\n")

        start_testing_time = datetime.datetime.now()

        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for i, (inputs, labels) in enumerate(test_dataloader):
            inputs, labels = inputs, labels

            if num_classes == 1:
                outputs = model(inputs).squeeze(1)
                labels = labels.float()
                predicted_test = torch.sigmoid(outputs) >= 0.5
            else:
                outputs = model(inputs)
                _, predicted_test = torch.max(outputs.data, 1)

            all_preds.append(predicted_test.cpu().item())
            all_labels.append(labels.cpu().item())

            correct += (predicted_test == labels).sum().item()
            total += labels.size(0)

            # Real-time progress display in the terminal
            sys.stdout.write(f"\rExample: {total}, test success: {100 * correct / total:.2f}%")
            sys.stdout.flush()

        # Log final results for the iteration
        logger.info(f"\nIteration {test_iteration + 1} completed: "
                    f"Total examples: {total}, Accuracy: {100 * correct / total:.2f}%")

        end_testing_time = datetime.datetime.now()
        inference_time = end_testing_time - start_testing_time
        total_inference_time += inference_time

        minutes, seconds = divmod(inference_time.total_seconds(), 60)
        logger.info(f"Inference time for iteration {test_iteration + 1}: {int(minutes)} min {seconds:.2f} sec")

    average_inference_time = total_inference_time / num_tests
    avg_minutes, avg_seconds = divmod(average_inference_time.total_seconds(), 60)
    logger.info(f"\nAverage Inference time over {num_tests} iterations: {int(avg_minutes)} min {avg_seconds:.2f} sec")

    logger.info("\n##################### [Inference time] - Testing completed #####################")
    return average_inference_time

def get_preds_and_labels(model, test_dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    for inputs, labels in test_dataloader:
        if model.num_classes == 1:
            inputs, labels = inputs.float(), labels.float()
            outputs = model(inputs).squeeze(1)
            labels = labels.float()
            predicted_test = torch.sigmoid(outputs) >= 0.5
        else:
            inputs, labels = inputs, labels
            outputs = model(inputs)
            _, predicted_test = torch.max(outputs.data, 1)

        all_preds.extend(predicted_test.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

def get_classification_report(test_dataloader, model, logger=None):
    all_preds, all_labels = get_preds_and_labels(model, test_dataloader)
    # Convert the set of numerical labels to strings
    target_names = [str(i) for i in set(all_labels)]

    logger.info("######################################### Classification report #########################################")
    logger.info(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    logger.info("##########################################################################################################")


def get_accuracy_score(test_dataloader, model, logger=None):
    all_preds, all_labels = get_preds_and_labels(model, test_dataloader)

    logger.info("############################################# Accuracy score #############################################")
    logger.info(accuracy_score(all_labels, all_preds))
    logger.info("##########################################################################################################")


def get_confusion_matrix(test_dataloader, model, save_path=None):
    all_preds, all_labels = get_preds_and_labels(model, test_dataloader)

    cm = confusion_matrix(all_labels, all_preds)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    labels = np.array([["{0}\n{1:.2f}%".format(value, percentage) for value, percentage in zip(row, percent_row)]
                          for row, percent_row in zip(cm, cm_percentage)])

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=True, xticklabels=True, yticklabels=True)
    plt.title("Confusion Matrix with Counts and Percentages")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved at [{save_path}].")

    plt.show()


