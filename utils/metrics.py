import datetime
import torch

# batch_size = 1
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

num_tests = 5
total_inference_time = datetime.timedelta()

num_classes = 1 # n-cars
# num_classes = 10 # n-mnist

for test_iteration in range(num_tests):
    print(f"\n### Testing - Iteration {test_iteration + 1}/{num_tests} ###\n")

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

        print(f"Example: {total}, test success: {100 * correct / total:.2f}%", end='\r')

    end_testing_time = datetime.datetime.now()
    inference_time = end_testing_time - start_testing_time
    total_inference_time += inference_time

    minutes, seconds = divmod(inference_time.total_seconds(), 60)
    print("\nInference time: " + str(int(minutes)) + " min " + str(seconds) + " sec")

average_inference_time = total_inference_time / num_tests
avg_minutes, avg_seconds = divmod(average_inference_time.total_seconds(), 60)
print("\nAverage Inference time over {} iterations: {} min {} sec".format(num_tests, int(avg_minutes), avg_seconds))
