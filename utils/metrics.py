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

    # 确保模型和数据在同一设备上
    device = next(model.parameters()).device  # 获取模型的设备
    logger.info(f"Model is on device: {device}")

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
            # 确保输入数据移动到模型的设备
            inputs, labels = inputs.to(device), labels.to(device)

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

            # 实时显示推断进度
            sys.stdout.write(f"\rExample: {total}, test success: {100 * correct / total:.2f}%")
            sys.stdout.flush()

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

    num_test_samples = len(test_dataloader.dataset)
    ave_inference_time_per_sample = average_inference_time / num_test_samples
    ave_inference_time_per_sample_ms = ave_inference_time_per_sample.total_seconds() * 1000
    logger.info(f"\nAverage Inference time per sample: {ave_inference_time_per_sample_ms:.2f} ms")

    logger.info("\n##################### [Inference time] - Testing completed #####################")
    return average_inference_time


def get_preds_and_labels(model, test_dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    # 获取模型所在的设备
    device = next(model.parameters()).device

    for inputs, labels in test_dataloader:
        # 将数据移动到模型所在设备
        inputs, labels = inputs.to(device), labels.to(device)

        if model.num_classes == 1:
            outputs = model(inputs).squeeze(1)
            labels = labels.float()
            predicted_test = torch.sigmoid(outputs) >= 0.5
        else:
            outputs = model(inputs)
            _, predicted_test = torch.max(outputs.data, 1)

        all_preds.extend(predicted_test.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def get_classification_report(test_dataloader, model, logger=None):
    all_preds, all_labels = get_preds_and_labels(model, test_dataloader)
    # Convert the set of numerical labels to strings
    target_names = [str(i) for i in set(all_labels)]

    logger.info(
        "\n######################################### Classification report #########################################")
    logger.info(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    logger.info(
        "\n##########################################################################################################")


def get_accuracy_score(test_dataloader, model, logger=None):
    all_preds, all_labels = get_preds_and_labels(model, test_dataloader)

    logger.info(
        "\n############################################# Accuracy score #############################################")
    logger.info(accuracy_score(all_labels, all_preds))
    logger.info(
        "\n##########################################################################################################")


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


