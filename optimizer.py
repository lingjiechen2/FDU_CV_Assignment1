class SGD:
    def __init__(self, initial_lr=0.01, warmup_steps=1000, decay_style='constant', lr_decay_rate=0.95, lr_decay_steps=1000):
        assert decay_style in ['constant', 'exponential', 'step'], "Invalid decay style, please choose from 'constant', 'exponential', or 'step'"
        self.base_lr = initial_lr
        self.current_lr = initial_lr
        self.global_step = 0
        self.warmup_steps = warmup_steps
        self.decay_style = decay_style
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        self.lr_trajectory = []  # Initialize the learning rate trajectory list

    def update_learning_rate(self):
        """
        Updates the learning rate according to the warmup and decay schedules.
        """
        if self.global_step < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.global_step / float(self.warmup_steps)
            self.current_lr = self.base_lr * warmup_factor
        else:
            # Learning rate decay
            if self.decay_style == 'constant':
                # No decay applied, constant learning rate after warmup
                self.current_lr = self.base_lr
            elif self.decay_style == 'exponential':
                # Exponential decay
                decay_steps = (self.global_step - self.warmup_steps) // self.lr_decay_steps
                self.current_lr = self.base_lr * (self.lr_decay_rate ** decay_steps)
            elif self.decay_style == 'step':
                # Step decay
                steps_since_warmup = (self.global_step - self.warmup_steps)
                if steps_since_warmup % self.lr_decay_steps == 0:
                    self.current_lr *= self.lr_decay_rate
        
        self.lr_trajectory.append(self.current_lr)  # Append the current learning rate to the trajectory list

    def update_params(self, params, grads):
        """
        Updates parameters using the gradient descent optimization algorithm with dynamic learning rate.

        Parameters:
        - params: dict, a dictionary containing the parameters to be updated
        - grads: dict, a dictionary containing the gradients of the loss w.r.t. the parameters

        Returns:
        - updated_params: dict, the updated parameters
        """
        self.update_learning_rate()  # Update learning rate based on global step
        updated_params = {}
        for key in params.keys():
            updated_params[key] = params[key] - self.current_lr * grads[key]
        self.global_step += 1  # Increment global step after update
        return updated_params

    def get_lr_trajectory(self):
        """
        Returns the learning rate trajectory as a list.
        """
        return self.lr_trajectory
