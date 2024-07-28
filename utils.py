import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import DataLoader, TensorDataset
from modules import ChestXRayDataset
import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from models import get_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
#from datasets_utils import train_transforms, test_transforms_256, verify_transforms_256, verify_transforms_512, test_transforms_512
from datasets_utils import get_transform
import numpy as np
from sklearn.manifold import TSNE

finding_to_label = {'NORMAL':0,
 'BACTERIA':1,
 'Pneumonia/Viral/COVID-19':2}

label_to_finding = {v: k for k, v in finding_to_label.items()}

def display_images(category, df, num_images=5):
    """
    Display a specified number of images from a given category in the DataFrame.
    
    Parameters:
    category (str): The category of images to display (e.g., 'NORMAL', 'BACTERIA', 'COVID').
    df (DataFrame): The DataFrame containing image file paths and categories.
    num_images (int): The number of images to display.
    """
    # Filter the DataFrame for the specified category
    category_df = df[df['finding'] == category]
    
    # Randomly select the specified number of images
    sample_images = category_df.sample(num_images)
    
    # Plot the images
    plt.figure(figsize=(20, 10))
    for i, filepath in enumerate(sample_images['filepath']):
        img = mpimg.imread(filepath)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"{category}")
        plt.xlabel(f"{img.shape}")
    plt.show()


# Function to show a batch of images
def show_images(images, labels):
    # Create a grid of images
    fig, axes = plt.subplots(1, len(images), figsize=(15, 15))
    if len(images) == 1:
        axes = [axes]  # Make sure axes is iterable if only one image

    for img, ax, label in zip(images, axes, labels):
        ax.imshow(img.permute(1, 2, 0).numpy().squeeze(), cmap='gray')  # Convert from (C, H, W) to (H, W, C)
        ax.set_title(f'Label: {label}')
        ax.axis('off')

    plt.show()


# -------------------------------------------------- Datasets -------------------------------------------------- #
def get_original_dataset(train_df, config):#train_transforms=None):
    og_train_df, og_val_df = train_test_split(
        train_df,
        test_size=0.2,  # 20% for validation
        stratify=train_df['finding'],  # Stratify based on the class labels
        random_state=42  # For reproducibility
        )
    train_dataset = ChestXRayDataset(dataframe=og_train_df, transform=get_transform(config['transformation_type']))
    val_dataset = ChestXRayDataset(dataframe=og_val_df, transform=get_transform('test_transforms_256'))
    return train_dataset, val_dataset

def get_undersampled_dataset(train_df, config):
    # Perform the stratified split using train_df
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,  # 20% for validation
        stratify=train_df['finding'],  # Stratify based on the class labels
        random_state=42  # For reproducibility
    )
    
    # Count the number of COVID samples in the training set
    covid_count = train_df[train_df['finding'] == finding_to_label['Pneumonia/Viral/COVID-19']].shape[0]
    print(f'Number of COVID samples in training set: {covid_count}')

    # Sample from the NORMAL and BACTERIA classes in the training set
    normal_samples_train = train_df[train_df['finding'] == finding_to_label['NORMAL']].sample(covid_count, random_state=42)
    bacteria_samples_train = train_df[train_df['finding'] == finding_to_label['BACTERIA']].sample(covid_count, random_state=42)
    covid_samples_train = train_df[train_df['finding'] == finding_to_label['Pneumonia/Viral/COVID-19']]

    # Combine the samples into a new DataFrame for training
    undersampled_train_df = pd.concat([normal_samples_train, bacteria_samples_train, covid_samples_train]).reset_index(drop=True)
    print(f'Undersampled training dataset size: {undersampled_train_df.shape[0]}')

    # Create dataset instance for undersampled training data and original validation data
    undersampled_train_dataset = ChestXRayDataset(dataframe=undersampled_train_df, transform=get_transform(config['transformation_type']))
    val_dataset = ChestXRayDataset(dataframe=val_df, transform=get_transform('test_transforms_256'))
    print(f'Validation dataset size: {val_df.shape[0]}')

    return undersampled_train_dataset, val_dataset


def get_oversampled_dataset(train_df, config):
    # Perform the stratified split using train_df
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,  # 20% for validation
        stratify=train_df['finding'],  # Stratify based on the class labels
        random_state=42  # For reproducibility
    )
    
    # Oversample only the training set
    most_represented_count = train_df[train_df['finding'] == 1].shape[0]
    print(f'Number of samples for label 1 ({label_to_finding[1]}): {most_represented_count}')
    
    label_0_samples = train_df[train_df['finding'] == 0].sample(most_represented_count, replace=True, random_state=42)
    label_2_samples = train_df[train_df['finding'] == 2].sample(most_represented_count, replace=True, random_state=42)
    label_1_samples = train_df[train_df['finding'] == 1]

    oversampled_train_df = pd.concat([label_0_samples, label_1_samples, label_2_samples]).reset_index(drop=True)
    print(f'Oversampled training dataset size: {oversampled_train_df.shape[0]}')
    
    # Create dataset instance for oversampled training data and original validation data
    oversampled_trainset = ChestXRayDataset(dataframe=oversampled_train_df, transform=get_transform(config['transformation_type']))
    valset = ChestXRayDataset(dataframe=val_df, transform=get_transform('test_transforms_256'))
 
    return oversampled_trainset, valset


def get_datasets(config, train_df, test_df, test_trans=None):
    #transformation = get_transform(config['transformation_type'])
    if config['dataset_balance'] == 'oversampled':
        train_dataset, val_dataset = get_oversampled_dataset(train_df, config)# transformation)
    elif config['dataset_balance'] == 'undersampled':
        train_dataset, val_dataset = get_undersampled_dataset(train_df, config)#transformation)
    elif config['dataset_balance'] == 'original' or config['dataset_balance'] == 'class_weights':
        train_dataset, val_dataset = get_original_dataset(train_df, config)#transformation)
        # Need to add weighted sampler for imbalanced dataset
    elif config['dataset_balance'] == 'toy':
        train_dataset, val_dataset = get_undersampled_dataset(train_df, config)#transformation)
    else:
        raise ValueError(f"Unknown dataset balance mode: {config['dataset_balance']}")
    
    if test_trans is None:
        test_dataset = ChestXRayDataset(dataframe=test_df, transform=get_transform('test_transforms_256'))
    else:
        test_dataset = ChestXRayDataset(dataframe=test_df, transform=test_trans)
    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

def compute_class_weights(loader):
    class_counts = {}
    for _, labels in loader:
        for label in labels:
            label = label.item()
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
    total_count = sum(class_counts.values())
    class_weights = {cls: total_count / count for cls, count in class_counts.items()}
    return class_weights

# -------------------------------------------------- Training and Evaluation -------------------------------------------------- #

def make(config, device='cpu', train_loader=None):
    # Choose model and intialize it
    model = get_model(config['model'], num_classes=config['num_classes']).to(device)
    
    # Define loss function
    if config['dataset_balance'] == 'class_weights':
        class_weights = compute_class_weights(train_loader)
        weights = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)    
    else:
        criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    return model, criterion, optimizer


@torch.no_grad()
def evaluate_val(model, criterion, val_loader, device='cpu'):
    '''
    Evaluate the loss and precision score on the validation set for the model
    '''
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    avg_precision = precision_score(all_labels, all_preds, average='weighted')
    
    model.train()
    return avg_loss, avg_precision



def train(model, loader, val_loader, criterion, optimizer, config, device='cpu', use_wandb=False):
    if use_wandb: 
        wandb.watch(model, criterion, log="all", log_freq=10)
    
    example_ct = 0
    batch_ct = 0
    running_loss = 0.0
    running_correct = 0
    samples_cnt_for_acc = 0
    best_val_per = 0.0
    best_model_path = f"train_models/best_model_{config['dataset_balance']}_{config['run_name']}.pth"

    for epoch in tqdm(range(config['epochs'])):
        model.train()  # Ensure the model is in training mode
        for _, (images, labels) in enumerate(loader):
            loss, correct = train_batch(images, labels, model, optimizer, criterion, device)
            running_loss += loss.item()
            running_correct += correct
            example_ct += len(images)
            samples_cnt_for_acc += len(images)
            batch_ct += 1

            # Report metrics every 20th batch
            if (batch_ct % 20) == 0:
                avg_loss = running_loss / 20
                avg_acc = running_correct / samples_cnt_for_acc
                val_loss, val_per = evaluate_val(model, criterion, val_loader, device)
                train_log(avg_loss, val_loss, avg_acc, val_per, example_ct, epoch, use_wandb)
                running_loss = 0.0
                running_correct = 0.0
                samples_cnt_for_acc = 0

                # Save the best model based on validation accuracy
                if val_per > best_val_per:
                    best_val_per = val_per
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Model saved with validation percision: {best_val_per:.4f}")  


def train_batch(images, labels, model, optimizer, criterion, device='cpu'):
    images, labels = images.to(device), labels.to(device)
    
    model.train()  # Ensure the model is in training mode
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    preds = outputs.argmax(1)
    correct = (preds == labels).float().sum().item()

    return loss, correct


def train_log(train_loss, val_loss, train_acc, val_per, example_ct, epoch, use_wandb=False):
    """
    Logs training and validation metrics for each epoch.

    This function performs two main tasks:
    1. If Weights & Biases (wandb) is being used, it logs metrics to the wandb dashboard.
    2. It prints the metrics to the console.

    Args:
    train_loss (float): The average training loss for the epoch
    val_loss (float): The average validation loss for the epoch
    train_acc (float): The training accuracy for the epoch
    val_acc (float): The validation accuracy for the epoch
    example_ct (int): The cumulative number of examples seen so far
    epoch (int): The current epoch number
    use_wandb (bool): Flag to determine if Weights & Biases should be used for logging (default: False)

    Returns:
    None
    """

    # Log metrics to Weights & Biases if it's being used
    if use_wandb:
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_per": val_per
        }, step=example_ct)
    
    # Print metrics to console
    print(f"Epoch: {epoch}, Examples: {example_ct}, "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Train Acc: {train_acc:.4f}, Val Percision: {val_per:.4f}")


def test(model, config, test_loader, device='cpu', use_wandb=False):
    model.eval()
    all_preds = []
    all_labels = []

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            #print(images.shape)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} test images: {correct / total:.4%}")
        if use_wandb: 
            wandb.log({"test_accuracy": correct / total})

    # Save the model in the exchangeable ONNX format
    # torch.onnx.export(model, images, f"model_{config['model']}_dataset_{config['dataset_balance']}_epochs_{config['epochs']}.onnx")
    # wandb.save(f"model_{config['model']}_dataset_{config['dataset_balance']}_epochs_{config['epochs']}.onnx")
    
    # Save the model in PyTorch format
    torch.save(model.state_dict(), f"model_{config['model']}_dataset_{config['dataset_balance']}_epochs_{config['epochs']}.pth")
    if use_wandb : wandb.save(f"model_{config['model']}_dataset_{config['dataset_balance']}_epochs_{config['epochs']}.pth")


def model_pipeline(hyperparameters, train_df, test_df, device='cpu', use_wandb=False):
    '''
    Executes the complete machine learning pipeline for COVID-19 chest X-ray classification.

    This function orchestrates the entire process of training and evaluating a model:
    1. Initializes Weights & Biases (wandb) if specified
    2. Prepares data loaders for training, validation, and testing
    3. Creates the model, loss function, and optimizer
    4. Trains the model using the specified hyperparameters
    5. Evaluates the model on the test set
    6. Logs results to wandb if enabled

    Args:
    hyperparameters (dict): A dictionary of hyperparameters for model configuration
    train_df (pd.DataFrame): DataFrame containing training data
    test_df (pd.DataFrame): DataFrame containing test data
    device (str): The device to run the model on ('cpu' or 'cuda')
    use_wandb (bool): Flag to enable Weights & Biases for experiment tracking

    Returns:
    torch.nn.Module: The trained model
    '''
    if use_wandb:
        # Use Weights & Biases for experiment tracking
        with wandb.init(project='COVID-19_classification', config=hyperparameters):
            config = wandb.config

            # Prepare data loaders
            train_loader, val_loader, test_loader = get_datasets(config, train_df, test_df)
            # Create small train_lodaer to overfit

            # Initialize model, loss function, and optimizer
            model, criterion, optimizer = make(config, device, train_loader)

            # Train the model
            train(model, train_loader, val_loader, criterion, optimizer, config, device, use_wandb)
            #train_combined(model, train_loader, criterion, optimizer, config)
            
            # Evaluate the model
            test(model, config, test_loader, device, use_wandb)

            return model
    else:
        # Run without Weights & Biases
        config = hyperparameters
        # Prepare data loaders
        train_loader, val_loader, test_loader = get_datasets(config, train_df, test_df)

        # Commented out code for creating a dataset from a single batch
        # dataiter = iter(train_loader)
        # images, labels = next(dataiter)
        # # Create a new dataset from this batch
        # first_batch_dataset = TensorDataset(images, labels)
        # first_batch_loader = DataLoader(first_batch_dataset, batch_size=len(images))

        # Initialize model, loss function, and optimizer
        model, criterion, optimizer = make(config, device, train_loader)
        print('train loader:', len(train_loader))
        print('val loader:', len(val_loader))
        # Train the model
        train(model, train_loader, val_loader, criterion, optimizer, config, device)
        #train_combined(model, train_loader, criterion, optimizer, config)
        
        # Evaluate the model
        test(model, config, test_loader, device)

        return model


def evaluate_model(model, test_loader, device):
    """
    Evaluates a PyTorch model on a test dataset and computes various performance metrics.

    This function performs the following tasks:
    1. Sets the model to evaluation mode
    2. Runs the model on the test data
    3. Computes overall accuracy, precision, recall, and F1 score
    4. Calculates per-class accuracy
    5. Generates and plots a confusion matrix

    Args:
    model (torch.nn.Module): The PyTorch model to evaluate
    test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset
    device (torch.device): The device (CPU or GPU) to run the evaluation on

    Returns:
    None (prints metrics and displays confusion matrix plot)
    """

    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation for efficiency
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get the index of the max log-probability

            # Extend prediction and label lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print('Evaluation complete')

    # Calculate per-class accuracy (EPR-Class Accuracy)
    epr_class_accuracy = []
    for cls in range(3):  # Assuming 3 classes: NORMAL, BACTERIA, COVID
        cls_indices = np.where(np.array(all_labels) == cls)[0]
        cls_preds = np.array(all_preds)[cls_indices]
        cls_labels = np.array(all_labels)[cls_indices]
        cls_accuracy = accuracy_score(cls_labels, cls_preds)
        epr_class_accuracy.append(cls_accuracy)
    
    # Calculate overall performance metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    # Print performance metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('EPR-Class Accuracy:')
    for cls, acc in enumerate(epr_class_accuracy):
        print(f'Class {cls}: {acc:.4f}')

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NORMAL', 'BACTERIA', 'COVID'], 
                yticklabels=['NORMAL', 'BACTERIA', 'COVID'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def evaluate_model_with_tta(model, test_loader_1, test_loader_2, test_loader_3, device='cpu'):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation
        for batch_1, batch_2, batch_3 in zip(test_loader_1, test_loader_2, test_loader_3):
            inputs_1, labels_1 = batch_1
            inputs_2, labels_2 = batch_2
            inputs_3, labels_3 = batch_3

            # Move inputs and labels to the appropriate device
            inputs_1, labels_1 = inputs_1.to(device), labels_1.to(device)
            inputs_2, labels_2 = inputs_2.to(device), labels_2.to(device)
            inputs_3, labels_3 = inputs_3.to(device), labels_3.to(device)

            # Get model outputs for each transformed batch
            outputs_1 = model(inputs_1)
            outputs_2 = model(inputs_2)
            outputs_3 = model(inputs_3)

            # Average the outputs
            avg_outputs = (outputs_1 + outputs_2 + outputs_3) / 3

            # Get predictions
            _, preds = torch.max(avg_outputs, 1)

            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_1.cpu().numpy())  # Assuming all labels are the same
    
        # Calculate per-class accuracy (EPR-Class Accuracy)
    epr_class_accuracy = []
    for cls in range(3):  # Assuming 3 classes: NORMAL, BACTERIA, COVID
        cls_indices = np.where(np.array(all_labels) == cls)[0]
        cls_preds = np.array(all_preds)[cls_indices]
        cls_labels = np.array(all_labels)[cls_indices]
        cls_accuracy = accuracy_score(cls_labels, cls_preds)
        epr_class_accuracy.append(cls_accuracy)
    
    # Calculate overall performance metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    # Print performance metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('EPR-Class Accuracy:')
    for cls, acc in enumerate(epr_class_accuracy):
        print(f'Class {cls}: {acc:.4f}')

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NORMAL', 'BACTERIA', 'COVID'], 
                yticklabels=['NORMAL', 'BACTERIA', 'COVID'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    
def plot_tsne(model, config, train_df, test_df, device):
    """
    Extracts features from the input of fc3, performs t-SNE, and plots the result.
    
    Args:
    model (nn.Module): The trained model
    config (dict): Configuration dictionary containing dataset parameters
    device (torch.device): The device to run the model on
    """
    # Get the dataloaders
    train_loader, _, test_loader = get_datasets(config, train_df, test_df)

    # Set up the hook
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = input[0].detach().cpu().numpy()
        return hook

    # Register the hook on fc3
    model.fc3.register_forward_hook(get_features('fc3_input'))

    # Set the model to evaluation mode
    model.eval()
    loaders = {
        'Train': train_loader,
        'Test': test_loader
    }
    for loader_name, loader in loaders.items():
        # Extract features and labels
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Extracting features"):
                images = images.to(device)
                _ = model(images)  # Forward pass
                all_features.append(features['fc3_input'])
                all_labels.extend(labels.numpy())

        # Concatenate all features and convert labels to numpy array
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.array(all_labels)

        # Perform t-SNE
        print("Performing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(all_features)

        # Plot the results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f't-SNE visualization of features before the final layer - {loader_name}')
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')
        
        # Add legend
        classes = ['NORMAL', 'BACTERIA', 'COVID']  # Adjust these labels based on your classes
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=classes[i], 
                        markerfacecolor=plt.cm.viridis(i/2), markersize=10) for i in range(3)]
        plt.legend(handles=legend_elements, title='Classes')
        
        plt.tight_layout()
        plt.show()

    # Remove the hook after we're done
    model.fc3._forward_hooks.clear()

def plot_resnet_tsne(model, config, train_df, test_df, device):
    """
    Extracts features from the penultimate layer, performs t-SNE, and plots the result.
    
    Args:
    model (nn.Module): The trained model (ModifiedResNet50)
    config (dict): Configuration dictionary containing dataset parameters
    train_df (DataFrame): Training data
    test_df (DataFrame): Test data
    device (torch.device): The device to run the model on
    """
    # Get the dataloaders
    train_loader, _, test_loader = get_datasets(config, train_df, test_df)

    # Set up the hook
    features = {}
    def get_features(name):
        def hook(model, input, output):
            # CHANGE: Capture output instead of input
            features[name] = output.detach().cpu().numpy()
        return hook

    # CHANGE: Register the hook on the avgpool layer of ResNet
    model.resnet.avgpool.register_forward_hook(get_features('penultimate_features'))

    # Set the model to evaluation mode
    model.eval()
    loaders = {
        'Train': train_loader,
        'Test': test_loader
    }
    for loader_name, loader in loaders.items():
        # Extract features and labels
        all_features = []
        all_labels = []

        with torch.no_grad():
            # CHANGE: Updated description in tqdm
            for images, labels in tqdm(loader, desc=f"Extracting features from {loader_name}"):
                images = images.to(device)
                _ = model(images)  # Forward pass
                # CHANGE: Extract and flatten features
                extracted_features = features['penultimate_features'].squeeze()
                all_features.append(extracted_features)
                all_labels.extend(labels.numpy())

        # Concatenate all features and convert labels to numpy array
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.array(all_labels)

        # Perform t-SNE
        # CHANGE: Updated print statement
        print(f"Performing t-SNE for {loader_name}...")
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(all_features)

        # Plot the results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap='viridis')
        plt.colorbar(scatter)
        # CHANGE: Updated title
        plt.title(f't-SNE visualization of penultimate layer features - {loader_name}')
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')
        
        # Add legend
        classes = ['NORMAL', 'BACTERIA', 'COVID']  # Adjust these labels based on your classes
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=classes[i], 
                        markerfacecolor=plt.cm.viridis(i/2), markersize=10) for i in range(3)]
        plt.legend(handles=legend_elements, title='Classes')
        
        plt.tight_layout()
        plt.show()

    # CHANGE: Remove the hook from the correct layer
    model.resnet.avgpool._forward_hooks.clear()
