import numpy as np
from data_loader import batch_generator

def evaluate_model(model, X_eval, Y_eval):
    assert X_eval.shape[0] == Y_eval.shape[0], "Number of samples in X_eval and Y_eval must be equal"
    Y_pred, _ = model.forward(X_eval)
    predictions = np.argmax(Y_pred, axis=1)
    loss = model.compute_loss(Y_pred, Y_eval)
    true_labels = np.argmax(Y_eval, axis=1)
    correct_predictions = np.sum(predictions == true_labels)
    accuracy = correct_predictions / X_eval.shape[0]
    return accuracy, loss

def train_model(model, optimizer, X_train, Y_train, X_eval, Y_eval, epochs, batch_size, validation_epochs=1, verbose = True):
    """
    Trains the neural network model.

    Parameters:
    - model: ThreeLayerNN, the neural network model to be trained
    - optimizer: SGD, the optimizer used for updating model parameters
    - X_train: numpy.ndarray, the features of the training set
    - Y_train: numpy.ndarray, the true labels of the training set
    """
    train_accuracy_list = []
    test_accuracy_list = []
    train_loss_list = []
    test_loss_list = []
    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0  # Initialize counter for correct predictions
        total_samples = 0  # Initialize counter for total samples
        
        for X_batch, Y_batch in batch_generator(X_train, Y_train, batch_size=batch_size):
            # Forward pass
            Y_pred, cache = model.forward(X_batch)
            # Compute loss
            loss = model.compute_loss(Y_pred, Y_batch)
            epoch_loss += loss
            # Calculate accuracy
            predictions = np.argmax(Y_pred, axis=1)
            true_labels = np.argmax(Y_batch, axis=1)
            correct_predictions += np.sum(predictions == true_labels)
            total_samples += X_batch.shape[0]

            # Backward pass
            grads = model.backward(X_batch, Y_batch, cache)
            # print(np.mean(grads['W1']), np.mean(grads['W2']), np.mean(grads['W3']))
            # grad1.append(np.mean(grads['W1']))
            # grad2.append(np.mean(grads['W2']))
            # grad3.append(np.mean(grads['W3']))

            # Update parameters
            model.params = optimizer.update_params(model.params, grads)

        avg_epoch_loss = epoch_loss / (X_train.shape[0] / batch_size)
        accuracy = correct_predictions / total_samples  
        # Validation
        if epoch % validation_epochs == 0:
            eval_accuracy, eval_loss = evaluate_model(model, X_eval, Y_eval)
            print(f"Epoch {epoch + 1}, Loss:{eval_loss}, Validation Accuracy: {100*eval_accuracy:.2f}%") if verbose else None
        else:
            print(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss}, Accuracy: {100*accuracy:.2f}%") if verbose else None
            
        train_accuracy_list.append(accuracy)
        train_loss_list.append(avg_epoch_loss)
        test_accuracy_list.append(eval_accuracy)
        test_loss_list.append(eval_loss)
    return train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list
        # Optional: Implement validation here if desired
