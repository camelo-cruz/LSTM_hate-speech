import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

current_dir = os.getcwd()

def plot_train_test(train_accuracies, test_accuracies, train_losses, test_losses, test_f1_scores, name):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{name} - Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{name} - Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(test_f1_scores, label='Test F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'{name} - Test F1 Score')
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
    plt.figure(figsize=(12, 6))

    train_labels = train_labels.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    train_label_count = np.bincount(train_labels)
    test_label_count = np.bincount(test_labels)

    plt.subplot(1, 2, 1)
    plt.pie(train_label_count, labels=class_names, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title('Training Set Distribution')

    plt.subplot(1, 2, 2)
    plt.pie(test_label_count, labels=class_names, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title('Test Set Distribution')

    plt.tight_layout()

    plots_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'train_test_distribution_pie.png')
    plt.savefig(plot_path)
    print(f'Saved figure: {plot_path}')
    
    plt.show()