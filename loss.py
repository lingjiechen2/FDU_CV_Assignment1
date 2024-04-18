import numpy as np

def softmax(predictions):
    exps = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(predictions, labels):
    """
    Computes the cross-entropy loss between predictions and labels.

    Parameters:
    - predictions: numpy.ndarray, the raw model outputs (logits) before softmax
    - labels: numpy.ndarray, the true labels in one-hot encoded form

    Returns:
    - loss: float, the mean cross-entropy loss
    """
    m = labels.shape[0]
    probs = softmax(predictions)
    log_likelihood = -np.log(probs[range(m), labels.argmax(axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss

def delta_cross_entropy_softmax(predictions, labels):
    """
    Computes the derivative of the cross-entropy loss with respect to the predictions.

    Parameters:
    - predictions: numpy.ndarray, the raw model outputs (logits)
    - labels: numpy.ndarray, the true labels in one-hot encoded form

    Returns:
    - delta: numpy.ndarray, the gradients of the loss with respect to the predictions
    """
    m = labels.shape[0]
    grads = softmax(predictions)
    grads[range(m), labels.argmax(axis=1)] -= 1
    grads = grads/m
    return grads
