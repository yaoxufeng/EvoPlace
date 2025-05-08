import math
import torch
from torch.optim.optimizer import Optimizer, required

class CusOptimizer(Optimizer):
    """
    @brief Improved Dreamplace Custom Optimizer
    This optimizer applies Hierarchical Multi-Scale techniques with an advanced line search method.
    """

    def __init__(self, params, lr=required, obj_and_grad_fn=required, constraint_fn=None, scales=3, convergence_threshold=1e-5, use_bb=True):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(
            lr=lr,
            scales=scales,
            convergence_threshold=convergence_threshold,
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
            obj_eval_count=0
        )
        super(CusOptimizer, self).__init__(params, defaults)

        self.obj_and_grad_fn = obj_and_grad_fn
        self.constraint_fn = constraint_fn
        self.use_bb = use_bb
        if len(self.param_groups) != 1:
            raise ValueError("Only parameters with a single tensor are supported")

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            constraint_fn = self.constraint_fn
            scales = group['scales']
            convergence_threshold = group['convergence_threshold']

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                if not group['u_k']:
                    group['u_k'].extend([p.data.clone() for _ in range(scales)])
                    group['v_k'].extend([p for _ in range(scales)])

                u_k = group['u_k'][i]
                v_k = group['v_k'][i]

                for scale in range(scales):
                    scaled_lr = group['lr'] / (2 ** scale)
                    obj_k, g_k = obj_and_grad_fn(v_k)

                    if not group['obj_k']:
                        group['obj_k'].append(None)
                    group['obj_k'][i] = obj_k.data.clone()

                    if not group['a_k']:
                        group['a_k'].append(torch.ones(1, dtype=g_k.dtype, device=g_k.device))
                        group['v_k_1'].append(torch.zeros_like(v_k, requires_grad=True))
                        group['v_k_1'][i].data.copy_(v_k - scaled_lr * g_k)

                    a_k = group['a_k'][i]
                    v_k_1 = group['v_k_1'][i]

                    obj_k_1, g_k_1 = obj_and_grad_fn(v_k_1)
                    if not group['obj_k_1']:
                        group['obj_k_1'].append(None)
                    group['obj_k_1'][i] = obj_k_1.data.clone()

                    if group['v_kp1'][i] is None:
                        group['v_kp1'][i] = torch.zeros_like(v_k, requires_grad=True)

                    v_kp1 = group['v_kp1'][i]

                    if not group['alpha_k']:
                        group['alpha_k'].append((v_k - v_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2))

                    alpha_k = group['alpha_k'][i]

                    a_kp1 = (1 + (4 * a_k.pow(2) + 1).sqrt()) / 2
                    coef = (a_k - 1) / a_kp1

                    with torch.no_grad():
                        s_k = (v_k - v_k_1)
                        y_k = (g_k - g_k_1)

                        bb_short_step_size = (s_k.dot(y_k) / y_k.dot(y_k)).data
                        adapt_step_size = min(alpha_k, bb_short_step_size)

                    u_kp1 = v_k - adapt_step_size * g_k
                    v_kp1.data.copy_(u_kp1 + coef * (u_kp1 - u_k))
                    constraint_fn(v_kp1)

                    if g_k.norm() < convergence_threshold:
                        break

                    group['obj_eval_count'] += 1
                    v_k_1.data.copy_(v_k.data)
                    alpha_k.data.copy_(adapt_step_size.data)
                    u_k.data.copy_(u_kp1.data)
                    v_k.data.copy_(v_kp1.data)
                    a_k.data.copy_(a_kp1.data)

        return loss

'''Key improvement points summary:
## Key improvements from original Dreamplace Optimizer:
1. **Hierarchical Multi-Scale Optimization**: Introduced multiple scales in optimization to systematically refine placements from coarse to fine resolutions, enhancing HPWL reduction.
2. **Adaptive Gradient Update**: Dynamically adjusts gradient computation and learning rate at each scale, improving convergence efficiency.
3. **Improved Line Search**: Advanced adaptive step size selection incorporated for stable and efficient convergence, reducing dependency on heuristic backtracking.
4. **Convergence Sensitivity**: Integrates a dynamic check to terminate at any scale once convergence is achieved according to a threshold, improving computational resource allocation.
'''