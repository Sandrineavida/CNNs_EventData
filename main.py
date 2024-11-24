import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


from models.cnn_lenet import CNNLeNet
model = CNNLeNet(num_classes=1)
print(model)
# Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of paramters in the model: {total_params}")


from utils.config_parser import ConfigParser
from train.setup import setup_training_components
train_config = ConfigParser("config/train/train_config_c2.cfg")
(num_epochs,
 criteria,
 optimizer,
 scheduler,
 device,
 num_classes,
 patience ) = setup_training_components(model, train_config)



from data.datasets import get_dataloaders
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


from utils.checkpoint import load_checkpoint
checkpoint_path = 'experiments/checkpoint.pth'
start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

from train.train import train_model_sheduled
(train_losses, train_accuracies,
 valid_losses, valid_accuracies,
 test_losses, test_accuracies,
 epoch_times, learning_rates,
 saved_epochs, stopped_at_epoch) = train_model_sheduled(
    model, train_dataloader, valid_dataloader, test_dataloader,
    criteria, optimizer, scheduler,
    num_epochs=num_epochs, patience=patience,
    checkpoint_path=checkpoint_path, device=device, num_classes=num_classes
)

from utils.visualisation import plot_learning_curves
plot_learning_curves(
    num_epochs, train_losses, train_accuracies,
    valid_losses, valid_accuracies, test_losses,
    test_accuracies, epoch_times, learning_rates,
    saved_epochs, stopped_at_epoch
)