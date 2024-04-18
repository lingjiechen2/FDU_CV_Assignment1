import numpy as np

def batch_generator(X, y, batch_size=32, shuffle=False):
    """
    Parameters:
    - X: numpy.ndarray, feature dataset
    - y: numpy.ndarray, labels dataset
    - batch_size: int, size of each batch
    - shuffle: bool, whether to shuffle the data before creating batches
    """
    dataset_size = X.shape[0]
    indices = np.arange(dataset_size)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, dataset_size, batch_size):
        end_idx = min(start_idx + batch_size, dataset_size)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]

# Example usage:
# for X_batch, y_batch in batch_generator(X_train, Y_train):
#     # Training loop here
