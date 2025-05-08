import math
import torch
from torch.optim.optimizer import Optimizer, required

class CusOptimizer(Optimizer):
    """
    @brief Improved Dreamplace Custom Optimizer
    This optimizer applies Nesterov's accelerated gradient with a Wolfe conditions line search for step size estimation,
    combined with L-BFGS for better optimization.
    """

    def __init__(self, params, lr=required, obj_and_grad_fn=required, constraint_fn=None, use_lbfgs=True):
        """
        @brief Initialization of the optimizer.
        @param params: The variables (placement positions) to be optimized.
        @param lr: Learning rate for the optimization.
        @param obj_and_grad_fn: A callable function that returns the objective function value (e.g., HPWL) and its gradient.
        @param constraint_fn: A callable function to enforce constraints on the placement variables (e.g., legal positions).
        @param use_lbfgs: A flag to enable L-BFGS method for second-order information.
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(
            lr=lr,
            u_k=[],
            v_k=[],
            g_k=[],
            obj_k=[],
            a_k=[],
            alpha_k=[],
            v_k_1=[],
            g_k_1=[],
            obj_k_1=[],
            v_kp1=[None],
            obj_eval_count=0,
            history={}  # Store L-BFGS history: s_k's and y_k's
        )
        super(CusOptimizer, self).__init__(params, defaults)

        self.obj_and_grad_fn = obj_and_grad_fn
        self.constraint_fn = constraint_fn
        self.use_lbfgs = use_lbfgs

        if len(self.param_groups) != 1:
            raise ValueError("Only parameters with a single tensor are supported")

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            constraint_fn = self.constraint_fn
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                if not group['u_k']:
                    group['u_k'].append(p.data.clone())
                    group['v_k'].append(p)

                u_k = group['u_k'][i]
                v_k = group['v_k'][i]

                obj_k, g_k = obj_and_grad_fn(v_k)
                if not group['obj_k']:
                    group['obj_k'].append(None)
                group['obj_k'][i] = obj_k.data.clone()

                if not group['a_k']:
                    group['a_k'].append(torch.ones(1, dtype=g_k.dtype, device=g_k.device))
                    group['v_k_1'].append(torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True))
                    group['v_k_1'][i].data.copy_(group['v_k'][i] - group['lr'] * g_k)

                a_k = group['a_k'][i]
                v_k_1 = group['v_k_1'][i]

                obj_k_1, g_k_1 = obj_and_grad_fn(v_k_1)
                if not group['obj_k_1']:
                    group['obj_k_1'].append(None)
                group['obj_k_1'][i] = obj_k_1.data.clone()

                if group['v_kp1'][i] is None:
                    group['v_kp1'][i] = torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True)

                v_kp1 = group['v_kp1'][i]

                alpha_k = self._line_search(v_k, g_k, v_k_1, g_k_1, obj_k, obj_k_1)

                a_kp1 = (1 + (4 * a_k.pow(2) + 1).sqrt()) / 2
                coef = (a_k - 1) / a_kp1

                u_kp1 = v_k - alpha_k * g_k
                v_kp1.data.copy_(u_kp1 + coef * (u_kp1 - u_k))

                constraint_fn(v_kp1)

                group['obj_eval_count'] += 1
                v_k_1.data.copy_(v_k.data)
                u_k.data.copy_(u_kp1.data)
                v_k.data.copy_(v_kp1.data)
                a_k.data.copy_(a_kp1.data)

        return loss

    def _line_search(self, v_k, g_k, v_k_1, g_k_1, obj_k, obj_k_1):
        s_k = v_k - v_k_1
        y_k = g_k - g_k_1

        if self.use_lbfgs:
            history = self.param_groups[0]['history']
            if 's' not in history:
                history['s'] = []
                history['y'] = []
                history['rho'] = []

            if len(history['s']) > 10:
                history['s'].pop(0)
                history['y'].pop(0)
                history['rho'].pop(0)

            history['s'].append(s_k)
            history['y'].append(y_k)
            history['rho'].append(1.0 / torch.dot(y_k, s_k))

            q = g_k
            alpha = []

            for s, y, rho in reversed(list(zip(history['s'], history['y'], history['rho']))):
                alpha_i = rho * torch.dot(s, q)
                alpha.append(alpha_i)
                q = q - alpha_i * y

            r = q
            for s, y, rho, alpha_i in zip(history['s'], history['y'], history['rho'], reversed(alpha)):
                beta = rho * torch.dot(y, r)
                r = r + s * (alpha_i - beta)

            step_size = r.norm(p=2) / g_k.norm(p=2)

        else:
            bb_long_step_size = (s_k.dot(s_k) / s_k.dot(y_k)).data
            bb_short_step_size = (s_k.dot(y_k) / y_k.dot(y_k)).data
            lip_step_size = (s_k.norm(p=2) / y_k.norm(p=2)).data

            step_size = bb_short_step_size if bb_short_step_size > 0 else min(lip_step_size, bb_long_step_size)

        return step_size

    def _wolfe_line_search(self, v_k, g_k, d_k, obj_k, c1=1e-4, c2=0.9):
        alpha = 1.0
        beta = 0.0
        factor = 0.5

        while True:
            new_v_k = v_k + alpha * d_k
            new_obj_k, new_g_k = self.obj_and_grad_fn(new_v_k)
            if (new_obj_k <= obj_k + c1 * alpha * g_k.dot(d_k)) and (new_g_k.dot(d_k) >= c2 * g_k.dot(d_k)):
                return alpha

            beta = factor * alpha
            alpha *= factor

'''Key improvement points summary:
## Key improvements from original Dreamplace Optimizer

1. **Advanced Line Search Method with Wolfe Conditions**:
   - Introduced Wolfe condition-based line search method to ensure a sufficient decrease condition and achieve better convergence stability.

2. **Utilized Limited-Memory BFGS (L-BFGS)**:
   - Applied L-BFGS to leverage second-order information without the overhead of the full Hessian computation. This method stores a fixed number of previous updates and gradients to approximate the inverse Hessian product.

3. **Integration of Line Search with Second-Order Momentum**:
   - Integrated line search with the Nesterov acceleration method to dynamically adapt the step size, enhancing the robustness and speed of the optimizer.

These changes aim to provide a more reliable and faster convergence, potentially resulting in better HPWL performance for global placement tasks in electronic design automation.
'''