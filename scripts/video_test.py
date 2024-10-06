import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import os

# Load the trained model
model_path = '/Users/elsontho/Desktop/ASLModel/models/asl_model.pth'
model = models.resnet18(weights='DEFAULT')  # Use weights parameter instead of pretrained=True
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 29)  # Ensure this matches the number of classes
model.load_state_dict(torch.load(model_path))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define a list of ASL alphabets (A-Z + space, delete, nothing)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'SPACE', 'DELETE', 'NOTHING']

# Define a function to make predictions
def predict_image(image):
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)  # Get probabilities
        top_probs, top_indices = torch.topk(probabilities, 3)  # Get top 3 predictions
    return top_indices[0].tolist(), top_probs[0].tolist()  # Return the indices and probabilities

# Specify video file path
video_path = '/Users/elsontho/Desktop/ASLModel/videotests/ASLAlphabets.mov'  # Replace with your video file path

# Start video capture from file
cap = cv2.VideoCapture(video_path)

# Check if video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Output video settings (Optional: to save output video with predictions)
output_path = '/Users/elsontho/Desktop/ASLModel/labelledoutput/labeled_video.avi'  # Set the path where you want to save the labeled video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Make predictions on the current frame
    top_indices, top_probs = predict_image(frame)
    
    # Create an overlay for predictions
    overlay = frame.copy()
    
    # Prepare the overlay text
    overlay_text = "\n".join([f"{class_names[idx]}: {prob:.2f}" for idx, prob in zip(top_indices, top_probs)])
    
    # Define the position and size of the overlay rectangle
    overlay_height = 100  # Height of the overlay
    overlay_position = (10, 10)
    
    # Create a transparent rectangle
    cv2.rectangle(overlay, (overlay_position[0], overlay_position[1]), 
                  (overlay_position[0] + 300, overlay_position[1] + overlay_height), 
                  (0, 0, 0), -1)  # Black background
    overlay = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)  # Blend the overlay

    # Put the text on the overlay
    cv2.putText(overlay, 'Predictions:', (overlay_position[0] + 5, overlay_position[1] + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, overlay_text, (overlay_position[0] + 5, overlay_position[1] + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the frame with the overlay
    cv2.imshow('Video ASL Recognition', overlay)

    # Write the labeled frame to the output video
    out.write(overlay)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Save the output video
cv2.destroyAllWindows()
