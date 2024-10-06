import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# Load the trained model
model_path = '/Users/elsontho/Desktop/ASLModel/models/asl_model.pth'
model = models.resnet18(weights='DEFAULT')  # Use weights parameter instead of pretrained=True
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 29)  # Make sure this matches the number of classes in your dataset
model.load_state_dict(torch.load(model_path))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define a list of ASL alphabets (assuming you have 26 letters A-Z + 3 additional classes for space, delete, etc.)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'SPACE', 'DELETE', 'NOTHING']

# Define a function to make predictions
def predict_image(image):
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]  # Return the actual ASL letter

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Make predictions
    label = predict_image(frame)
    
    # Display the frame with the prediction
    cv2.putText(frame, f'Predicted Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-Time ASL Recognition', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
