# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.

## Problem Statement and Dataset
Develop an image classification model using transfer learning with the pre-trained VGG19 model.

## DESIGN STEPS
### STEP 1:
Import required libraries, load the dataset, and define training & testing datasets.

### STEP 2:
Initialize the model, loss function, and optimizer. Use CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.

### STEP 3:
Train the model using the training dataset with forward and backward propagation.

### STEP 4:
Evaluate the model on the testing dataset to measure accuracy and performance.

### STEP 5:
Make predictions on new data using the trained model.
## PROGRAM
# Load Pretrained Model and Modify for Transfer Learning
```
model = models.vgg19(pretrained=True)
```
# Modify the final fully connected layer to match the dataset classes
```
num_classes = len(train_dataset.classes)
model.classifier[6] = nn.Linear(4096, num_classes)
```
# Include the Loss function and optimizer
```
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
# Train the model
```
## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # Convert labels to one-hot encoding
            labels = nn.functional.one_hot(labels, num_classes=num_classes).float().to(device) # Change this line
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # Convert labels to one-hot encoding
                labels = nn.functional.one_hot(labels, num_classes=num_classes).float().to(device) # Change this line
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:POZHILAN V D")
    print("Register Number:212223240118")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
```


## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
Include your plot here
</br>
</br>
</br>

### Confusion Matrix
Include confusion matrix here
</br>
</br>
</br>

### Classification Report
Include Classification Report here
</br>
</br>
</br>

### New Sample Prediction
</br>
</br>
</br>

## RESULT
</br>
</br>
</br>
