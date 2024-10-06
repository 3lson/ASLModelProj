import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
#transforms - user to apply transformations to the images 
#datasets - provides a dataset class for loading common datasets and creating custom datasets
#DataLoader - helps in loading the data in batches


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess training data
train_path = '/Users/elsontho/Desktop/ASLModel/data/asl_alphabet_train/asl_alphabet_train/'
train_data = datasets.ImageFolder(train_path, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Load and preprocess testing data
test_path = '/Users/elsontho/Desktop/ASLModel/data/asl_alphabet_test/'
test_data = datasets.ImageFolder(test_path, transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# Get a batch of images
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Show some images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

imshow(torchvision.utils.make_grid(images))