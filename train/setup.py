import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


# [training]
# num_classes=1
# quantised=False
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


def setup_training_components(model, train_config):
    # 从配置文件中获取训练参数
    training_section = train_config.get_section("training")

    # 检查所有必需的键是否存在
    required_keys = ["learning_rate", "num_epochs", "criteria", "optimizer",
                     "scheduler", "scheduler_factor", "scheduler_patience", "scheduler_mode",
                     "device", "num_classes", "patience"]
    for key in required_keys:
        if key not in training_section:
            raise KeyError(f"Missing required key: {key} in training section")

    # 获取并转换配置参数
    learning_rate = float(training_section["learning_rate"])
    num_epochs = int(training_section["num_epochs"])
    criteria = getattr(torch.nn, training_section["criteria"])()
    optimizer = getattr(optim, training_section["optimizer"])(model.parameters(), lr=learning_rate)
    scheduler = getattr(lr_scheduler, training_section["scheduler"])(optimizer,
                                                                     factor=float(training_section["scheduler_factor"]),
                                                                     patience=int(training_section["scheduler_patience"]),
                                                                     mode=training_section["scheduler_mode"])
    device = torch.device(training_section["device"])

    num_classes = int(training_section["num_classes"])

    patience = int(training_section["patience"])

    return num_epochs, criteria, optimizer, scheduler, device, num_classes, patience