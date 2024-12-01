import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.quantization as quantization
from utils.device_info import get_cpu_info, get_gpu_info_nvidia_smi


# [training]
# num_classes=1
# num_epochs=10
# criteria=BCEWithLogitsLoss
# optimizer=Adam
# learning_rate=0.001
# patience=7
# scheduler=ReduceLROnPlateau
# scheduler_factor=0.1
# scheduler_patience=3
# scheduler_mode='min'
# device='cpu'


def setup_training_components(model, train_config, logger=None):
    logger.info("\n################## Setting up training components ##################")
    # 从配置文件中获取训练参数
    training_section = train_config.get_section("training")

    required_keys = ["learning_rate", "num_epochs", "optimizer",
                     "scheduler", "scheduler_factor", "scheduler_patience", "scheduler_mode",
                     "patience"]
    for key in required_keys:
        if key not in training_section:
            raise KeyError(f"Missing required key: {key} in training section")

    # 获取并转换配置参数
    learning_rate = float(training_section["learning_rate"])
    num_epochs = int(training_section["num_epochs"])
    # criteria = getattr(torch.nn, training_section["criteria"])()

    num_classes = model.num_classes

    if num_classes == 1:
        criteria = torch.nn.BCEWithLogitsLoss()
    else: # num_classes=10 for n-mnist
        criteria = torch.nn.CrossEntropyLoss()

    optimizer = getattr(optim, training_section["optimizer"])(model.parameters(), lr=learning_rate)
    scheduler = getattr(lr_scheduler, training_section["scheduler"])(optimizer,
                                                                     factor=float(training_section["scheduler_factor"]),
                                                                     patience=int(training_section["scheduler_patience"]),
                                                                     mode=training_section["scheduler_mode"])

    patience = int(training_section["patience"])


    if not model.quantised:
        device = torch.device("cpu")
        model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
        quantization.prepare_qat(model, inplace=True)
    else:
        device = torch.device(training_section["device"])
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(device.type)

    # print("Successfully set up training components:")
    logger.info(f"Number of classes: {'2' if num_classes == 1 else num_classes}")
    logger.info(f"Model quantised: {model.quantised}")
    # logger.info(f"Device: {device}")
    device_info = get_gpu_info_nvidia_smi() if device.type == 'cuda' else get_cpu_info()
    device_info = "(CPU)\n" + device_info if device.type == 'cpu' else "(GPU)\n" + device_info
    logger.info(f"Device: {device_info}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Criteria: {criteria}")
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {scheduler}")
    logger.info(f"Patience: {patience}")

    logger.info("\n################## Training components set up successfully ##################")

    return num_epochs, criteria, optimizer, scheduler, device, num_classes, patience