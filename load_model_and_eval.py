import torch
from models.cnn_lenet_q import CNNLeNet_q
from models.cnn_lenet import CNNLeNet
from models.separable_convolution_q import SeparableConv_LeNet_q, MobileNet_q
from models.separable_convolution import SeparableConv_LeNet, MobileNet
from utils.log import Logger
from utils.model_load_helper import load_clean_state_dict
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning)

path_dir = "experiments/"
exp = "exp032/"
train_log_path = path_dir + exp + "train_log.txt"

logger = Logger(log_exp=exp, log_dir="experiments", log_file="reload_eval_log.txt")

logger.info("\n-------------------- Extract model info from the train_log.txt -------------------")
# Check the log to see if the model is quantised
# and Get the  `scale` and `zero_point` params from the log
# and Get the class of the model
# and get the test set path
device = None
model_class = None
num_classes = None
test_dataset_path = None
quantised = None
scale, zero_point = None, None
with open(train_log_path, "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if "Device: (CPU)" in line:
            device = torch.device("cpu")
            logger.info(f"Device: {device}")
        elif "Device: (GPU)" in line:
            device = torch.device("cuda")
            logger.info(f"Device: {device}")
        if "######################### Model architecture #########################" in line:
            model_line = lines[i + 1].strip()
            model_class = model_line.split("(")[0]
            logger.info(f"Class of the model: {model_class}")
        if line.startswith("Number of classes"):
            num_classes = int(line.split(":")[-1].strip())
            logger.info(f"Number of classes: {num_classes}")
            if num_classes == 2: num_classes = 1
            logger.info(f"(num_classes = {num_classes})")

        if line.startswith("Quantised model"):
            quantised_str = line.split(":")[-1].strip()
            quantised = quantised_str.lower() == "true"
            logger.info(f"Quantised model: {quantised}")

        if quantised:
            if line.startswith("scale"):
                scale = float(line.split("=")[-1])
                logger.info(f"Scale: {scale}")
            elif line.startswith("zero_point"):
                zero_point = int(line.split("=")[-1])
                logger.info(f"Zero point: {zero_point}")

        if line.startswith("Test set path"):
            test_dataset_path = line.split(": ")[-1]
            logger.info(f"Test set path: {test_dataset_path}")

        if device is not None and model_class is not None and num_classes is not None and quantised is not None and test_dataset_path is not None:
            if not quantised:
               break
            elif quantised and scale is not None and zero_point is not None:
               break

logger.info("\n-------------------- Model info extracted successfully -------------------")


logger.info("\n------------------- Model initialisation and loading -------------------")

if quantised:
    logger.info("Model was quantised, model structure: ")

    if model_class == "CNNLeNet":
        model = CNNLeNet_q(num_classes=num_classes, scale=scale, zero_point=zero_point)
    elif model_class == "SeparableConv_LeNet":
        model = SeparableConv_LeNet_q(num_classes=num_classes, scale=scale, zero_point=zero_point)
    elif model_class == "MobileNet":
        model = MobileNet_q(num_classes=num_classes, scale=scale, zero_point=zero_point)

    dict_path = path_dir + exp + "quantised_model.pth"
    model.load_state_dict(torch.load(dict_path))
    model.eval()
    logger.info(model)
else:
    logger.info("Model was not quantised, model structure: ")

    if model_class == "CNNLeNet":
        model = CNNLeNet(num_classes=num_classes, quantised=False)
    elif model_class == "SeparableConv_LeNet":
        model = SeparableConv_LeNet(num_classes=num_classes, quantised=False)
    elif model_class == "MobileNet":
        model = MobileNet(num_classes=num_classes, quantised=False)

    dict_path = path_dir + exp + "model.pth"
    state_dict = torch.load(dict_path)
    clean_state_dict = load_clean_state_dict(model, state_dict)  # clean the extra (futile) keys
    model.load_state_dict(clean_state_dict)
    model = model.to(device)
    model.eval()
    logger.info(model)

logger.info("Load model from: " + dict_path)
logger.info("\n------------------- Model successfully initialised ------------------- ")

logger.info("\n------------------------------- Test Data loading -------------------------------")
from utils.datasets import get_dataloader
# test_dataset_path = "data/ncars/ave_32x32_DATASETS/plain/test_n_cars_dataset_poolingave_1framepereventset_plain.pth"
batch_size = 32
test_dataloader = get_dataloader(test_dataset_path,
                                    batch_size)
logger.info("Test set path: " + test_dataset_path)
logger.info(f"Total number of samples in test_loader: {len(test_dataloader.dataset)}")
logger.info("\n---------------------------- Test Data loaded successfully ----------------------------")

from utils.metrics import get_classification_report
get_classification_report(test_dataloader, model, logger=logger, device=device)
from utils.metrics import get_accuracy_score
get_accuracy_score(test_dataloader, model, logger=logger, device=device)

# from utils.metrics import get_confusion_matrix
# exp_path = os.path.join("experiments", exp)
# save_cm_path = os.path.join(exp_path, "confusion_matrix.png")
# get_confusion_matrix(test_dataloader, model, save_path=save_cm_path)

from utils.metrics import get_inference_time
average_inference_time = get_inference_time(model, test_dataset_path, num_tests=5, logger=logger, device=device)