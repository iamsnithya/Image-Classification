# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset
The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the Fashion-MNIST dataset. The Fashion-MNIST dataset contains images of various clothing items (T-shirts, trousers, dresses, shoes, etc.), and the model aims to classify them correctly. The challenge is to achieve high accuracy while maintaining efficiency.


## Neural Network Model
![image](https://github.com/user-attachments/assets/2254dcdc-73bc-4bd4-8567-fe7bdc9e91bd)

## DESIGN STEPS


STEP 1: Problem Statement

Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

STEP 2:Dataset Collection

Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.

STEP 3: Data Preprocessing

Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

STEP 4:Model Architecture

Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.

STEP 5:Model Training

Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

STEP 6:Model Evaluation

Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

STEP 7: Model Deployment & Visualization

Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM

### Name: NITHYA S
### Register Number: 212224240106
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

# Load Fashion-MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)


# Get the shape of the first image in the training dataset
image, label = train_dataset[0]
print(image.shape)
print(len(train_dataset))


# Get the shape of the first image in the test dataset
image, label = test_dataset[0]
print(image.shape)
print(len(test_dataset))

# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)   # 10 classes in Fashion-MNIST

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # (32, 14, 14)
        x = self.pool(torch.relu(self.conv2(x)))  # (64, 7, 7)

        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name: NITHYA S')
print('Register Number: 212224240106')
summary(model, input_size=(1, 28, 28))

model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs=3):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: NITHYA S')
        print('Register Number: 212224240106')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Train the model
train_model(model, train_loader)

## Step 4: Test the Model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name: NITHYA S')
    print('Register Number: 212224240106')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name: NITHYA S')
    print('Register Number: 212224240106')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print('Name: NITHYA S')
    print('Register Number: 212224240106')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))


# Evaluate the model
test_model(model, test_loader)


## Step 5: Predict on a Single Image
import matplotlib.pyplot as plt
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
    class_names = dataset.classes

    # Display the image
    print('Name: NITHYA S')
    print('Register Number: 212224240106')
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')


# Example Prediction
predict_image(model, image_index=80, dataset=test_dataset)



```


## OUTPUT
### Training Loss per Epoch

<img width="292" height="201" alt="image" src="https://github.com/user-attachments/assets/a5abec78-848a-4e0e-b717-7032de64d729" />



### Confusion Matrix
<img width="832" height="732" alt="image" src="https://github.com/user-attachments/assets/0a444c4f-1b3b-4c68-8dd3-2d41473208a7" />



### Classification Report
<img width="538" height="417" alt="image" src="https://github.com/user-attachments/assets/e2e0d9bb-a2cb-4c90-8046-2ea06be8b4a7" />



### New Sample Data Prediction

<img width="516" height="568" alt="image" src="https://github.com/user-attachments/assets/67c823dd-35b8-49ea-a0c9-e89811d4d709" />


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
