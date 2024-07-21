import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50 as ResNet50

# This file contain the models we will use in the project.
# We have our own simple CNN model and a ResNet50 model from torchvision.

class MyCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fc1 = nn.Linear(32 * 32 * 32, 1024)  # Reduced size
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 32 * 32)  # Flatten the tensor
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, 0.5, training=self.training)  # Dropout layer for regularization
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, 0.5, training=self.training)  # Dropout layer for regularization
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fc1 = nn.Linear(128 * 32 * 32, 1024)  # Reduced size
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 128 * 32 * 32)  # Flatten the tensor
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, 0.5, training=self.training)  # Dropout layer for regularization
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, 0.5, training=self.training)  # Dropout layer for regularization
        x = self.fc3(x)
        return x
    
# A toy model to test the training pipeline
# This model will get a 224x224 image and output a 3-class classification
class ToyModel(nn.Module):
    def __init__(self, num_classes=3):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.linear1 = nn.Linear(6 * 220 * 220, 120)
        self.linear2 = nn.Linear(120, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 6 * 220 * 220)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    

class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ModifiedResNet50, self).__init__()
        self.resnet = ResNet50(pretrained=pretrained)
        # Changing the first layer to accept 1 channel for grayscale images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze the last bottleneck block in layer4 and the fully connected layer (fc)
        for param in self.resnet.layer4[2].parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
    
def get_model(model_name, num_classes, pretrained=False):
    """
    Get the specified model by name.
    
    Parameters:
    model_name (str): The name of the model to get.
    num_classes (int): The number of classes in the dataset.
    
    Returns:
    model: The model instance.
    """
    if model_name.lower() == 'simplecnn':
        return SimpleCNN(num_classes)
    elif model_name.lower() == 'resnet50':
        return ModifiedResNet50(num_classes, pretrained)
    elif model_name.lower() == 'toymodel':
        return ToyModel(num_classes)
    elif model_name.lower() == 'mycnn':
        return MyCNN(num_classes)
    else:
        raise ValueError(f"Model {model_name} not found.")