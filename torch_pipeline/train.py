import os
import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from dataset import get_mnist_dataloader
from model import MLP, CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)


batch_size = config["training"]["batch_size"]
epochs = config["training"]["epochs"]
lr = config["training"]["learning_rate"]

checkpoint_dir = config["paths"]["checkpoint_dir"]
log_dir = config["paths"]["log_dir"]

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Dataset
train_loader, test_loader = get_mnist_dataloader(batch_size)

# Model
model_type = config["model"]["type"]

if model_type == "mlp":
    model = MLP(hidden_size=config["model"]["hidden_size"])

elif model_type == "cnn":
    model = CNN()

model = model.to(device)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr
)

# Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config["scheduler"]["step_size"],
    gamma=config["scheduler"]["gamma"]
)

# TensorBoard
writer = SummaryWriter(log_dir)

# Early stopping
patience = config["early_stopping"]["patience"]
best_loss = float("inf")
patience_counter = 0

# Training Loop
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch} Loss {running_loss}")
    writer.add_scalar("Loss/train", running_loss, epoch)

    # Early stopping check
    if running_loss < best_loss:

        best_loss = running_loss
        patience_counter = 0

        torch.save(
            model.state_dict(),
            f"{checkpoint_dir}/model.pth"
        )
        print("Model checkpoint saved")

    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

writer.close()
print("Training finished")