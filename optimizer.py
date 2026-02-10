from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
import math


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            max_grad_norm: float = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # TODO: Clip gradients if max_grad_norm is set
            if group['max_grad_norm'] is not None:
                #using pytorch version
                torch.nn.utils.clip_grad_norm_(group['params'], group['max_grad_norm'])

            
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data #grabbing calculated gradients
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # raise NotImplementedError()

                #using torch.optim.Optimizer params
                state = self.state[p]

                # State Initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                #Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                #step increment
                state['step'] += 1

                #Update first and second moments of the gradients
                # m_t update = beta1 * m_t-1 + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # v_t update = beta2 * v_t-1 + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                #Bias correction
                # efficient version of bias correction
                step_size = alpha
                if correct_bias:
                    # bias_correction1 = 1 - beta1^t
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    # bias_correction2 = 1 - beta2^t
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    #Adjust the step size (alpha) based on how "warmed up" the moving averages are.
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # TODO: Update parameter
                denom = exp_avg_sq.sqrt().add_(eps)
                # p = p - step_size * (m_t / (sqrt(v_t) + eps)). Using addcdiv_ which performs: tensor.add(value * tensor1 / tensor2)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # TODO: Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                # AdamW: p = p - lr * weight_decay * p
                if weight_decay > 0.0:
                    p.data.add_(p.data, alpha=-group['lr'] * weight_decay)

        return loss