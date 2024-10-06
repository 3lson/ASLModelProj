import torch
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score
import signal
import sys
import os

# Define image transformations
print("Setting up data transformations...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess training data
train_path = '/content/data/asl_alphabet_train/asl_alphabet_train/'
train_data = datasets.ImageFolder(train_path, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Load and preprocess testing data
print("Loading datasets...")
test_path = '/content/data/asl_alphabet_test/'
test_data = datasets.ImageFolder(test_path, transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Load pre-trained ResNet18 model
print("Setting up the model...")
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(train_data.classes))  # Number of classes

# Check for MPS (Apple Silicon GPU support) or CPU fallback
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Using device: {device}")
model.to(device)

# Set up loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Define a function to save the model
def save_model():
    model_save_path = '/content/models/asl_model.pth'
    print(f"Saving the model to {model_save_path}...")
    torch.save(model.state_dict(), model_save_path)
    print("Model saved.")

# Define a handler for interrupt signal (Ctrl+C)
def signal_handler(sig, frame):
    print('Interrupt received, saving model...')
    save_model()
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Training loop
num_epochs = 4  # You can adjust this
print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Loss calculation
        loss.backward()  # Backward pass (gradient calculation)
        optimizer.step()  # Update weights

        running_loss += loss.item() * images.size(0)

        if (i + 1) % 10 == 0:  # Print loss every 10 batches
            print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

# Save the trained model
save_model()

# Evaluation
print("Starting evaluation...")
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy:.2f}')
