# This code implements the CLASSP optimizer for continual learning
# In case of using this software package or parts of it, cite:
# Oswaldo Ludwig, "CLASSP: a Biologically-Inspired Approach to Continual Learning through Adjustment Suppression and Sparsity Promotion", ArXiv, 2024.

import torch
import os


class CLASSP_optimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.2, threshold=0.5, epsilon=1e-5, power=1, apply_scaling=True):
        defaults = dict(lr=lr, threshold=threshold, epsilon=epsilon, power=power, apply_scaling=apply_scaling)
        super(CLASSP_optimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, lr=None, threshold=None, epsilon=None, power=None, apply_scaling=None, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Use the arguments if provided, otherwise use the values from group
            lr = lr if lr is not None else group['lr']
            threshold = threshold if threshold is not None else group['threshold']
            epsilon = epsilon if epsilon is not None else group['epsilon']
            power = power if power is not None else group['power']
            apply_scaling = apply_scaling if apply_scaling is not None else group['apply_scaling']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if 'grad_sum' not in state:
                    state['grad_sum'] = torch.zeros_like(p.data)

                # Update square_sum for this parameter
                if (grad ** 2).any() > threshold:

                    state['grad_sum'].add_(grad.abs()) ** power
                    if apply_scaling == True:
                       # Calculate the scaling factor for this parameter
                       scaling_factor = lr / (state['grad_sum'] + epsilon).pow(1/power)
                    else:
                       scaling_factor = lr

                    # Apply the update
                    p.data.add_(- scaling_factor * grad)

        return loss


    def average_grad_sum(self):
        total_grad_sum = 0
        total_params = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'grad_sum' in state:
                    total_grad_sum += state['grad_sum'].sum().item()
                    total_params += p.numel()
        return total_grad_sum / total_params if total_params > 0 else 0


    def save_checkpoint(model, optimizer, path):
        torch.save({
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
        }, path)

    def load_checkpoint(model, optimizer, path):
        if os.path.isfile(path):
           checkpoint = torch.load(path)
           model.load_state_dict(checkpoint['model_state_dict'])
           optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
           print("Checkpoint loaded successfully from '{}'".format(path))
        else:
           print("No checkpoint found at '{}'".format(path))


    def save_optimizer(self, path):
        torch.save(self.state_dict(), path)

    def load_optimizer(self, path):
        if os.path.isfile(path):
            self.load_state_dict(torch.load(path))
        else:
            print("No optimizer found at '{}'".format(path))
