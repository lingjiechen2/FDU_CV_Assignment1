import numpy as np
import sys
sys.path.append('fashion-mnist/utils')
import argparse
from mnist_reader import load_mnist
from model import ThreeLayerNN
from optimizer import SGD
from train import train_model
from evaluation import test_model
from utils import save_model_parameters, load_model_parameters, evaluate_and_plot, split_array_randomly
from config import (input_size, hidden_layers_sizes, output_size, activation_function,
                    reg_lambda, epochs, batch_size, validation_epochs, learning_rate, 
                    warmup_steps, decay_style, lr_decay_rate, lr_decay_steps)

def parse_args():
    """
    Parses command line arguments for the neural network training and testing.

    Returns:
    argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train or Test Neural Network Model on Fashion MNIST')
    parser.add_argument('--test-only', action='store_true',
                        help='Load model parameters and test without training')
    parser.add_argument('--model-path', type=str, default='model_parameters',
                        help='Path to the model parameters directory')
    return parser.parse_args()



def main():
    args = parse_args()
    
    # Load data
    X_train, y_train = load_mnist('fashion-mnist/data/fashion', kind='train')
    X_test, y_test = load_mnist('fashion-mnist/data/fashion', kind='t10k')
    Y_train = np.zeros((y_train.shape[0], 10))
    Y_train[np.arange(y_train.shape[0]), y_train] = 1
    Y_test = np.zeros((y_test.shape[0], 10))
    Y_test[np.arange(y_test.shape[0]), y_test] = 1

    # Initialize model and optimizer
    model = ThreeLayerNN(input_size, hidden_layers_sizes, output_size, activation=activation_function, reg_lambda=reg_lambda)
    if args.test_only:
        # Load model parameters from specified directory
        load_model_parameters(model, args.model_path)
        # Evaluate the model on the test set directly
        test_loss, test_accuracy = test_model(model, X_test, Y_test, batch_size)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy:.2f}%")
    else:
        # Train and evaluate the model
        X_eval, X_train, Y_eval, Y_train = split_array_randomly(X_train, Y_train, ratio=0.1)
        optimizer = SGD(learning_rate, warmup_steps, decay_style, lr_decay_rate, lr_decay_steps)
        train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list = train_model(
            model, optimizer, X_train, Y_train, X_eval, Y_eval, epochs, batch_size, validation_epochs
        )

        # Save and load model parameters
        file_directory = f"model_parameters/epoches_{epochs}_lr_{learning_rate}_decay_{decay_style}_derate_{lr_decay_rate}_destep_{lr_decay_steps}"
        save_model_parameters(model, file_directory)
        evaluate_and_plot(train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list, directory='./result_plot', filename=f"epoches_{epochs}_lr_{learning_rate}_decay_{decay_style}_derate_{lr_decay_rate}_destep_{lr_decay_steps}.jpg")
        load_model_parameters(model, file_directory)

        # Evaluate the model on the test set
        test_loss, test_accuracy = test_model(model, X_test, Y_test, batch_size)
        

if __name__ == "__main__":
    main()
