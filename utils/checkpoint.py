import torch
import os

def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, valid_loss):
    state = {
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'valid_loss': valid_loss,
    }
    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, logger=None):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
        return start_epoch
    else:
        logger.info(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")
        return 0


def get_new_experiment_path(base_path='experiments'):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    existing_experiments = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    existing_experiments.sort()

    if existing_experiments:
        last_experiment = existing_experiments[-1]
        last_experiment_num = int(last_experiment.replace('exp', ''))
        new_experiment_num = last_experiment_num + 1
    else:
        new_experiment_num = 1

    new_experiment_dir = os.path.join(base_path, f'exp{new_experiment_num:03d}')
    os.makedirs(new_experiment_dir, exist_ok=True)  # 创建文件夹
    new_experiment_path= os.path.join(new_experiment_dir, 'checkpoint.pth')

    return new_experiment_path.replace('\\', '/')

