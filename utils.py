import numpy as np
from matplotlib import pyplot as plt
import os

def save_model_parameters(model, directory):
    """
    Saves the parameters of a model to individual .npy files within a specified directory.

    Parameters:
    - model: ThreeLayerNN, the neural network model
    - directory: str, the directory name where parameters will be saved
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for param_name, param_value in model.params.items():
        np.save(os.path.join(directory, f"{param_name}.npy"), param_value)

def load_model_parameters(model, directory):
    """
    Loads the parameters of a model from .npy files located in a specific directory.

    Parameters:
    - model: ThreeLayerNN, the neural network model
    - directory: str, the directory where parameters are stored
    """
    params_dict = {}
    for param_name in model.params.keys():
        param_path = os.path.join(directory, f"{param_name}.npy")
        if os.path.exists(param_path):
            params_dict[param_name] = np.load(param_path)
        else:
            raise FileNotFoundError(f"No parameter file found for {param_name} in {directory}")
    model.params = params_dict

def split_array_randomly(data, label , ratio):
    """
    Randomly splits a NumPy array into two parts based on a given ratio.
    Parameters:
    - data: numpy.ndarray, the input array to be split
    - ratio: float, the ratio of data to be included in the first part (0 < ratio < 1)
    Returns:
    - two numpy.ndarrays, the first part and the second part of the input data
    """
    indices = np.random.permutation(data.shape[0])
    num_samples_part1 = int(data.shape[0] * ratio)
    indices_part1 = indices[:num_samples_part1]
    indices_part2 = indices[num_samples_part1:]
    return data[indices_part1], data[indices_part2], label[indices_part1], label[indices_part2]

def evaluate_and_plot(train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list, directory, filename):
    """
    Evaluates the model on the test dataset and plots the accuracy and loss for training and testing, 
    then saves the plot to the specified directory.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracy_list, label='Train')
    plt.plot(test_accuracy_list, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss_list, label='Train')
    plt.plot(test_loss_list, label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Instead of plt.show(), save the plot to the specified directory with the given filename
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created")
    plt.savefig(os.path.join(directory, filename))
    plt.close()  # Close the figure to prevent display

# Example usage: evaluate_and_plot(train_acc, train_loss, test_acc, test_loss, 'path/to/directory', 'plot.png')
