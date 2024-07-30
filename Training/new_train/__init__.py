import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
num_workers = 8
num_epochs = 10
best_val_loss = float('inf')
patience = 3
no_improve = 0
output_dir = "saved_model"
max_length = 128
batch_size = 16
lr = 1e-5