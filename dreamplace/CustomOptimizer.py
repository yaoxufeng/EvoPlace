import math
import torch
from torch.optim.optimizer import Optimizer, required

class CusOptimizer(Optimizer):
    """
    @brief Dreamplace Custom Optimizer
    This optimizer applies Nesterov's accelerated gradient with the Barzilai-Borwein method for step size estimation.
    """

    def __init__(self, params, lr=required, obj_and_grad_fn=required, constraint_fn=None, use_bb=True):
        """
        @brief Initialization of the optimizer.
        @param params: The variables (placement positions) to be optimized.
        @param lr: Learning rate for the optimization.
        @param obj_and_grad_fn: A callable function that returns the objective function value (e.g., HPWL) and its gradient.
        @param constraint_fn: A callable function to enforce constraints on the placement variables (e.g., legal positions).
        @param use_bb: A flag to enable the Barzilai-Borwein (BB) method for adaptive step size calculation.
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        # Initialization of key variables used in the optimization process
        defaults = dict(
            lr=lr,              # Learning rate
            u_k=[],             # Major solution (current placement)
            v_k=[],             # Reference solution (used in Nesterov acceleration)
            g_k=[],             # Gradient at v_k
            obj_k=[],           # Objective value at v_k
            a_k=[],             # Optimization parameter a_k for acceleration
            alpha_k=[],         # Step size alpha_k
            v_k_1=[],           # Previous reference solution v_k^(k-1)
            g_k_1=[],           # Gradient at previous reference solution g_k^(k-1)
            obj_k_1=[],         # Objective value at previous reference solution
            v_kp1=[None],       # Storage for the next reference solution v_kp1
            obj_eval_count=0    # Counter to track the number of objective evaluations
        )
        super(CusOptimizer, self).__init__(params, defaults)

        self.obj_and_grad_fn = obj_and_grad_fn  # Function to compute objective and gradient
        self.constraint_fn = constraint_fn      # Function to enforce constraints
        self.use_bb = use_bb                    # Enable use of Barzilai-Borwein step size calculation

        # Ensure only a single tensor is passed in the parameter group
        if len(self.param_groups) != 1:
            raise ValueError("Only parameters with a single tensor are supported")

    def step(self, closure=None):
        """
        @brief Performs a single optimization step using the Nesterov accelerated gradient method.
        @param closure: A closure function to recompute the objective function (used for line search).
        """
        loss = None
        if closure is not None:
            loss = closure()  # Recompute the objective function (if needed)

        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            constraint_fn = self.constraint_fn
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue  # Skip parameters with no gradient

                # Initialize variables on the first iteration
                if not group['u_k']:
                    group['u_k'].append(p.data.clone())  # Store the initial major solution u_k
                    group['v_k'].append(p)               # Store the initial reference solution v_k

                u_k = group['u_k'][i]  # Current major solution (placement)
                v_k = group['v_k'][i]  # Current reference solution

                # Compute the objective and gradient at the current reference solution v_k
                obj_k, g_k = obj_and_grad_fn(v_k)
                if not group['obj_k']:
                    group['obj_k'].append(None)
                group['obj_k'][i] = obj_k.data.clone()  # Store the objective value at v_k

                # Initialize previous reference solution v_k_1 and gradient g_k_1
                if not group['a_k']:
                    group['a_k'].append(torch.ones(1, dtype=g_k.dtype, device=g_k.device))  # Initialize a_k
                    group['v_k_1'].append(torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True))
                    group['v_k_1'][i].data.copy_(group['v_k'][i] - group['lr'] * g_k)      # Initialize v_k_1

                a_k = group['a_k'][i]  # Optimization parameter a_k for acceleration
                v_k_1 = group['v_k_1'][i]  # Previous reference solution v_k^(k-1)

                # Compute the objective and gradient at the previous reference solution v_k_1
                obj_k_1, g_k_1 = obj_and_grad_fn(v_k_1)
                if not group['obj_k_1']:
                    group['obj_k_1'].append(None)
                group['obj_k_1'][i] = obj_k_1.data.clone()  # Store the objective value at v_k_1

                # Initialize the next reference solution v_kp1
                if group['v_kp1'][i] is None:
                    group['v_kp1'][i] = torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True)

                v_kp1 = group['v_kp1'][i]  # Next reference solution v_kp1

                # Compute the step size alpha_k using the Barzilai-Borwein method
                if not group['alpha_k']:
                    group['alpha_k'].append((v_k - v_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2))

                alpha_k = group['alpha_k'][i]  # Step size alpha_k

                # Line search to refine the step size, using alpha_k as an initial guess
                a_kp1 = (1 + (4 * a_k.pow(2) + 1).sqrt()) / 2  # Update acceleration parameter a_kp1
                coef = (a_k - 1) / a_kp1  # Coefficient for Nesterov acceleration

                with torch.no_grad():
                    # Compute step sizes based on BB method
                    s_k = (v_k - v_k_1)  # Difference in reference solutions
                    y_k = (g_k - g_k_1)  # Difference in gradients

                    # BB long and short step sizes
                    bb_long_step_size = (s_k.dot(s_k) / torch.sum(s_k * y_k)).data
                    bb_short_step_size = (s_k.dot(y_k) / y_k.dot(y_k)).data

                    # Lipschitz step size estimation
                    lip_step_size = (s_k.norm(p=2) / y_k.norm(p=2)).data

                    # Choose the step size: prefer BB short step size, fallback to other estimates
                    step_size = bb_short_step_size if bb_short_step_size > 0 else min(lip_step_size, alpha_k)

                # Perform one optimization step (Nesterov update)
                u_kp1 = v_k - step_size * g_k  # Gradient descent update for u_k
                v_kp1.data.copy_(u_kp1 + coef * (u_kp1 - u_k))  # Nesterov acceleration update for v_kp1

                # Apply constraints to the updated placement (e.g., legalize positions)
                constraint_fn(v_kp1)

                # Update internal states for the next iteration
                group['obj_eval_count'] += 1
                v_k_1.data.copy_(v_k.data)  # Update previous reference solution
                alpha_k.data.copy_(step_size.data)  # Update step size
                u_k.data.copy_(u_kp1.data)  # Update major solution
                v_k.data.copy_(v_kp1.data)  # Update reference solution
                a_k.data.copy_(a_kp1.data)  # Update acceleration parameter

        return loss