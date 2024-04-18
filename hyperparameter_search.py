import numpy as np
import argparse
from tqdm import tqdm
from model import ThreeLayerNN
from optimizer import SGD
from train import train_model
from evaluation import test_model
from utils import save_model_parameters, evaluate_and_plot

def hyperparameter_search(X_train, Y_train, X_eval, Y_eval, epochs = 10):
    parser = argparse.ArgumentParser(description='Hyperparameter search for 3-layer neural network')
    parser.add_argument('--decay_style', type=str, default='constant',
                        help='Choose decay style between constant and exponential')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train the model for')
    args = parser.parse_args()
    epochs = args.epochs
    decay_style = args.decay_style
    print(decay_style)
    learning_rates = [0.001, 0.01, 0.1]
    # learning_rates = [0.01]  # Tesing whether the whole code works
    batch_sizes = [32, 64, 128]
    hidden_layer_sizes = [(128, 64), (256, 128)]
    # hidden_layer_sizes = [(128, 64)]
    regularization_strengths = [0.001, 0.005, 0.01]
    # regularization_strengths = [0.001]
    lr_decay_rates = [0.9, 0.95, 0.99] if decay_style == 'exponential' else [1.0]

    best_acc = 0
    best_params = {}

    # Create a list of all parameter combinations
    all_combinations = [(lr, bs, hs, reg, lr_decay_rate) for lr in learning_rates for bs in batch_sizes for hs in hidden_layer_sizes for reg in regularization_strengths for lr_decay_rate in lr_decay_rates]

    # Open a file to write the hyperparameter results
    with open(f'hyperparameter_search_result/hyperparameter_results_{epochs}_epoch_{decay_style}_decay.txt', 'w') as f:
        # Loop over all combinations with tqdm for progress tracking
        for lr, bs, hs, reg, lr_decay_rate in tqdm(all_combinations, desc='Grid Search Progress'):
            model = ThreeLayerNN(input_size=784, hidden_sizes=hs, output_size=10, activation='relu', reg_lambda=reg)
            optimizer = SGD(initial_lr=lr, warmup_steps=100, decay_style=decay_style, lr_decay_rate=lr_decay_rate, lr_decay_steps=1000)
            train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list = train_model(model, optimizer, X_train, Y_train, X_eval, Y_eval, batch_size=bs, epochs=epochs, verbose=False)  # Assuming epochs is set here
            loss, acc = test_model(model, X_eval, Y_eval, batch_size=bs)

            # Write each result to the file
            f.write(f"LR: {lr}, Batch: {bs}, Hidden: {hs}, Lambda: {reg}, Lr_decay_rate:{lr_decay_rate}, Acc: {100*acc:.2f}\n")

            # Update best parameters if current accuracy is higher
            if acc > best_acc:
                best_acc = acc
                best_loss = loss
                best_train_accuracy = train_accuracy_list
                best_train_loss = train_loss_list
                best_test_accuracy = test_accuracy_list
                best_test_loss = test_loss_list
                best_model = model
                best_params = {
                    'learning_rate': lr,
                    'batch_size': bs,
                    'hidden_layer_sizes': hs,
                    'regularization_strength': reg,
                    'lr_decay_rate': lr_decay_rate
                }

        # Write the best result at the end of the file
        file_directory = f"best_param/epoches_{epochs}_lr_{best_params['learning_rate']}_decay_{decay_style}_derate_{best_params['lr_decay_rate']}_batchsize_{best_params['batch_size']}_hidden_{best_params['hidden_layer_sizes']}_lambda_{best_params['regularization_strength']}"
        save_model_parameters(best_model, file_directory)

        evaluate_and_plot(best_train_accuracy, best_train_loss, best_test_accuracy, best_test_loss, directory='./result_plot/best_search_results', filename=f"epoches_{epochs}_lr_{best_params['learning_rate']}_decay_{decay_style}_derate_{best_params['lr_decay_rate']}_batchsize_{best_params['batch_size']}_hidden_{best_params['hidden_layer_sizes']}_lambda_{best_params['regularization_strength']}.jpg")

        f.write(f"Best accuracy: {best_acc}\n")
        f.write(f"Best parameters: {best_params}\n")

    print("Best accuracy:", best_acc)
    print("Best parameters:", best_params)

if __name__ == "__main__":
    import numpy as np
    import sys
    from utils import split_array_randomly
    sys.path.append('fashion-mnist/utils')
    from mnist_reader import load_mnist

    X_train, y_train = load_mnist('fashion-mnist/data/fashion', kind='train')
    X_test, y_test = load_mnist('fashion-mnist/data/fashion', kind='t10k')
    Y_train = np.zeros((y_train.shape[0], 10));Y_train[np.arange(y_train.shape[0]), y_train] = 1
    Y_test = np.zeros((y_test.shape[0], 10));Y_test[np.arange(y_test.shape[0]), y_test] = 1
    X_eval, X_train, Y_eval, Y_train = split_array_randomly(X_train, Y_train, ratio=0.1)
    hyperparameter_search(X_train, Y_train, X_eval, Y_eval)
