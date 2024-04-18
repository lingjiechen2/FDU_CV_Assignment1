# config.py

# Model parameters
input_size = 784  # For example, 28x28 images flattened into a single vector (MNIST)
hidden_layers_sizes = [256, 128]  # Sizes of hidden layers
output_size = 10  # Number of classes, e.g., 10 for MNIST/Fashion-MNIST
activation_function = 'relu'  # Can be 'relu', 'sigmoid', or 'tanh'
reg_lambda = 0.1  # Regularization strength for L2

# Training parameters
epochs = 10
batch_size = 128
learning_rate = 0.01
warmup_steps = 0
decay_style = 'constant'  # 'constant', 'exponential', or 'step'
lr_decay_rate = 0.95
lr_decay_steps = 2000
shuffle_train_data = True
validation_epochs = 1

