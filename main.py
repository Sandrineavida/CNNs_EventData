import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
from utils.log import Logger
from utils.checkpoint import load_checkpoint
from utils.checkpoint import get_new_experiment_path


checkpoint_path = None  # or specify a path
# checkpoint_path = 'experiments/exp007/checkpoint.pth' # Note: if you are training a quantised model, you need to create a new checkpoint file
if checkpoint_path is None:
    checkpoint_path = get_new_experiment_path()
print(f"Checkpoint path: {checkpoint_path}")

# Check if the file exists
if os.path.exists(checkpoint_path):
    print(f"File {checkpoint_path} exists")
    # Check the file permissions
    if not os.access(checkpoint_path, os.R_OK):
        print(f"Read permission denied for {checkpoint_path}")
    if not os.access(checkpoint_path, os.W_OK):
        print(f"Write permission denied for {checkpoint_path}")
else:
    print(f"File {checkpoint_path} does not exist")

# Example of setting permissions (if needed)
# os.chmod(checkpoint_path, 0o666)  # Read and write permissions for everyone

# Get the experiment name from the folder path of the checkpoint_path (expXXX)
exp = checkpoint_path.split('/')[-2] # -2 means the second last element
print(f"Experiment: {exp}")

# # Initialise the logger
logger = Logger(log_exp=exp, log_dir="experiments", log_file="train_log.txt")
logger.info("========================================================================================================")
logger.info(f"Experiment: {exp}")
logger.info(f"Checkpoint path: {checkpoint_path}")


logger.info("\n######################### Model architecture #########################")
from models.cnn_lenet import CNNLeNet
from models.separable_convolution_lenet import SeparableConv_LeNet
# model = CNNLeNet(num_classes=1, quantised=False)
model = SeparableConv_LeNet(num_classes=1, quantised=True)

logger.info(model)
# Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Total number of paramters in the model: {total_params}")
logger.info(f"Quantised model: {model.quantised}")
logger.info("\n################### Model successfully initialised ###################")


from utils.config_parser import ConfigParser
from utils.setup import setup_training_components
train_config = ConfigParser("config/train/train_config_c2.cfg")
(num_epochs,
 criteria,
 optimizer,
 scheduler,
 device,
 num_classes,
 patience) = setup_training_components(model, train_config, logger)


logger.info("\n############################## Data loading ##############################")

from utils.datasets import get_dataloaders
train_dataset_path = "data/ncars/ave_32x32_DATASETS/plain/train_n_cars_dataset_poolingave_1framepereventset_plain.pth"
valid_dataset_path = "data/ncars/ave_32x32_DATASETS/plain/valid_n_cars_dataset_poolingave_1framepereventset_plain.pth"
test_dataset_path = "data/ncars/ave_32x32_DATASETS/plain/test_n_cars_dataset_poolingave_1framepereventset_plain.pth"
batch_size = 32
(train_dataloader,
 valid_dataloader,
 test_dataloader) = get_dataloaders(train_dataset_path,
                                    valid_dataset_path,
                                    test_dataset_path,
                                    batch_size)

logger.info("Train set path: " + train_dataset_path)
logger.info("Valid set path: " + valid_dataset_path)
logger.info("Test set path: " + test_dataset_path)

logger.info(f"Total number of samples in train_loader: {len(train_dataloader.dataset)}")
logger.info(f"Total number of samples in valid_loader: {len(valid_dataloader.dataset)}")
logger.info(f"Total number of samples in test_loader: {len(test_dataloader.dataset)}")

logger.info("\n######################### Data loaded successfully #########################")

# from utils.visualisation import plot_frames
# for single_frame, target in test_dataloader:
#     print(f"Single frame batch shape: {single_frame.shape}")
#     print(f"Target batch shape: {target.shape}")
#     break
#
# print(f"Total number of samples in train_loader: {len(test_dataloader.dataset)}")
#
# plot_frames(single_frame[22:28].squeeze())
# print(target[22:28])

logger.info("\n################################## checkpoint info ##################################")
start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler, logger=logger)
logger.info("\n#####################################################################################")

from train.train import train_model_sheduled
(train_losses, train_accuracies,
 valid_losses, valid_accuracies,
 test_losses, test_accuracies,
 train_times, total_train_time, average_train_time,
 learning_rates,
 saved_epochs, stopped_at_epoch) = train_model_sheduled(
    model, train_dataloader, valid_dataloader, test_dataloader,
    criteria, optimizer, scheduler,
    num_epochs=num_epochs, patience=patience,
    checkpoint_path=checkpoint_path, device=device, num_classes=num_classes,
    logger=logger
)

from utils.visualisation import plot_learning_curves
exp_path = os.path.join("experiments", exp)
save_lc_path = os.path.join(exp_path, "learning_curves.png")

plot_learning_curves(
    num_epochs, train_losses, train_accuracies,
    valid_losses, valid_accuracies, test_losses,
    test_accuracies, train_times, learning_rates,
    saved_epochs, stopped_at_epoch, save_path=save_lc_path
)

# Save model dict
import torch
import torch.quantization as quantization
model.eval()

if not model.quantised:
    model_path = os.path.join(exp_path, "model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"============================== Model saved to: {model_path} ==============================")
else:
    model_path = os.path.join(exp_path, "quantised_model.pth")
    if model.quantised:
        quantization.convert(model, inplace=True)  # TRUELY convert the model to a quantised model
    torch.save(model.state_dict(), model_path)
    logger.info(f"========================================================================================================")
    logger.info(f"\n!---------------------------- Quantised model saved to: {model_path} ----------------------------!")
    logger.info(f"scale={model.quant.scale.item()}")
    logger.info(f"zero_point={model.quant.zero_point.item()}")
    logger.info("!-------------------------------------------------------------------------------------------------------------------------!")
    logger.info("\nModel quantised:")
    logger.info(model)
    logger.info(f"========================================================================================================")


# Evaluate the model
from utils.metrics import get_classification_report
get_classification_report(test_dataloader, model, logger=logger)
from utils.metrics import get_accuracy_score
get_accuracy_score(test_dataloader, model, logger=logger)

from utils.metrics import get_confusion_matrix
save_cm_path = os.path.join(exp_path, "confusion_matrix.png")
get_confusion_matrix(test_dataloader, model, save_path=save_cm_path)

from utils.metrics import get_inference_time
average_inference_time = get_inference_time(model, test_dataset_path, num_tests=5, logger=logger)

