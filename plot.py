import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

current_dir = os.getcwd()

def plot_train_test(train_accuracies, test_accuracies, train_losses, test_losses, name):

    plt.figure(figsize=(10, 5))

    # Plot accuracies
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(name)
    plt.legend()

    # Plot losses
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(name)
    plt.legend()

    plots_dir = os.path.join(current_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'{name}.png')
    plt.savefig(plot_path)
    print(f'saved figure {plot_path}')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, name):

    plt.figure(figsize=(6, 5))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{name} - Confusion Matrix')

    plots_dir = os.path.join(current_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'{name}_confusion_matrix.png')
    plt.savefig(plot_path)
    print(f'saved figure {plot_path}')
    plt.show()

def plot_data_distribution(train_labels, test_labels, class_names):
    """
    Plots the distribution of classes in the training and test datasets as pie charts.
    
    Parameters:
    - train_labels: list or numpy array of training labels.
    - test_labels: list or numpy array of test labels.
    - class_names: list of class names corresponding to the label indices.
    """
    plt.figure(figsize=(12, 6))

    train_labels = train_labels.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    train_label_count = np.bincount(train_labels)
    test_label_count = np.bincount(test_labels)

    # Plot pie chart for training set
    plt.subplot(1, 2, 1)
    plt.pie(train_label_count, labels=class_names, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title('Training Set Distribution')

    # Plot pie chart for test set
    plt.subplot(1, 2, 2)
    plt.pie(test_label_count, labels=class_names, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title('Test Set Distribution')

    plt.tight_layout()

    # Save the plot to a directory
    plots_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'train_test_distribution_pie.png')
    plt.savefig(plot_path)
    print(f'Saved figure: {plot_path}')
    
    plt.show()