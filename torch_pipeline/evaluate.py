import torch
import yaml
from dataset import get_mnist_dataloader
from model import MLP, CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

model_type = config["model"]["type"]

if model_type == "mlp":
    model = MLP(hidden_size=config["model"]["hidden_size"])

elif model_type == "cnn":
    model = CNN()

model.load_state_dict(
    torch.load("checkpoints/model.pth")
)

model = model.to(device)
model.eval()
_, test_loader = get_mnist_dataloader(64)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
accuracy = correct / total
print("Accuracy:", accuracy)