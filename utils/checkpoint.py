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

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
        return start_epoch
    else:
        print("No checkpoint found at", checkpoint_path, ". Starting training from scratch.")
        return 0
