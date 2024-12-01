import torch
from models.cnn_lenet_q import CNNLeNet_q
from models.cnn_lenet import CNNLeNet
from utils.log import Logger
from utils.model_load_helper import load_clean_state_dict
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

path_dir = "experiments/"
exp = "exp008/"
log_path = path_dir + exp + "train_log.txt"

logger = Logger(log_exp=exp, log_dir="experiments", log_file="test_log.txt")

# Check the log to see if the model is quantised
with open(log_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("Quantised model"):
            quantised_str = line.split(":")[-1].strip()
            quantised = quantised_str.lower() == "true"
            break

if quantised:
    logger.info("Model is quantised, model structure: ")
    # Get the  `scale` and `zero_point` params from the log
    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("scale"):
                scale = float(line.split("=")[-1])
            elif line.startswith("zero_point"):
                zero_point = int(line.split("=")[-1])


    model = CNNLeNet_q(num_classes=1, scale=scale, zero_point=zero_point)
    dict_path = path_dir + exp + "quantised_model.pth"
    model.load_state_dict(torch.load(dict_path))
    model.eval()
    logger.info(model)
else:
    logger.info("Unquantised model: ")

    model = CNNLeNet(num_classes=1, quantised=False)
    dict_path = path_dir + exp + "model.pth"
    state_dict = torch.load(dict_path)
    clean_state_dict = load_clean_state_dict(model, state_dict)  # clean the extra (futile) keys
    model.load_state_dict(clean_state_dict)
    model.eval()
    logger.info(model)


logger.info("\n############################## Data loading ##############################")
from utils.datasets import get_dataloader
test_dataset_path = "data/ncars/ave_32x32_DATASETS/plain/test_n_cars_dataset_poolingave_1framepereventset_plain.pth"
batch_size = 32
test_dataloader = get_dataloader(test_dataset_path,
                                    batch_size)
logger.info("Test set path: " + test_dataset_path)
logger.info(f"Total number of samples in test_loader: {len(test_dataloader.dataset)}")
logger.info("\n######################### Data loaded successfully #########################")

from utils.metrics import get_classification_report
get_classification_report(test_dataloader, model, logger=logger)
from utils.metrics import get_accuracy_score
get_accuracy_score(test_dataloader, model, logger=logger)