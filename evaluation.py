import numpy as np
from data_loader import batch_generator


def test_model(model, X_test, Y_test, batch_size = 32):
    """
    Evaluates the model's performance on the test set.

    Parameters:
    - model: ThreeLayerNN, the trained neural network model
    - X_test: numpy.ndarray, the features of the test set
    - Y_test: numpy.ndarray, the true labels of the test set

    Returns:
    - test_loss: float, the average loss on the test set
    - test_accuracy: float, the accuracy of the model on the test set
    """
    num_batches = X_test.shape[0] // batch_size
    test_loss = 0
    correct_predictions = 0

    for X_batch, Y_batch in batch_generator(X_test, Y_test, batch_size=batch_size, shuffle=False):
        # Forward pass
        Y_pred, _ = model.forward(X_batch)

        # Compute loss
        batch_loss = model.compute_loss(Y_pred, Y_batch)
        test_loss += batch_loss

        # Calculate accuracy
        predictions = np.argmax(Y_pred, axis=1)
        true_labels = np.argmax(Y_batch, axis=1)
        correct_predictions += np.sum(predictions == true_labels)

    # Compute average loss and accuracy
    test_loss /= num_batches
    test_accuracy = correct_predictions / X_test.shape[0]

    return test_loss, test_accuracy

if __name__ == "__main__":
    import sys
    sys.path.append('fashion-mnist/utils')
    
    from mnist_reader import load_mnist
    from config import batch_size, input_size, hidden_layers_sizes, output_size
    from model import ThreeLayerNN
    from utils import load_model_parameters

    X_train, y_train = load_mnist('fashion-mnist/data/fashion', kind='train')
    X_test, y_test = load_mnist('fashion-mnist/data/fashion', kind='t10k')
    Y_train = np.zeros((y_train.shape[0], 10));Y_train[np.arange(y_train.shape[0]), y_train] = 1
    Y_test = np.zeros((y_test.shape[0], 10));Y_test[np.arange(y_test.shape[0]), y_test] = 1

    # Initialize the model
    model = ThreeLayerNN(input_size, hidden_layers_sizes, output_size)

    # Load trained model parameters (ensure the path and method are correctly set)
    file_path = ''
    load_model_parameters(model, file_path)

    # Evaluate the model on the test set
    test_loss, test_accuracy = test_model(model, X_test, Y_test)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
