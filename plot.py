import os
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

    # Save the plot
    plots_dir = os.path.join(current_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'{name}.png')
    plt.savefig(plot_path)

    plt.show()
