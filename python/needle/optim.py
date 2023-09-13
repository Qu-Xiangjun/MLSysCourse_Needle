"""Optimization module"""
import needle as ndl


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            # param grad maybe None
            if param.grad is None:
                continue
            # Init the u
            if self.u.get(param, None) is None:
                self.u[param] = ndl.init.zeros(*param.shape)
            # Compute current u with momentum, last u, grad and weight decay
            grad = self.momentum * self.u[param] + (1 - self.momentum) * \
                   (param.grad + self.weight_decay * param)
            self.u[param] = ndl.Tensor(grad, dtype=param.type, requires_grad=False)
            # Update the param
            param.data -= self.lr * self.u[param]


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for p in self.params:
            if p is None:
                continue
            if self.m.get(p, None) is None:
                self.m[p] = ndl.init.zeros(*p.shape)
            if self.v.get(p, None) is None:
                self.v[p] = ndl.init.zeros(*p.shape)
            
            # Notice the weight decay
            grad_data = p.grad.detach() + p.detach() * self.weight_decay

            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * grad_data
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (grad_data ** 2)

            m_hat = self.m[p] / (1 - self.beta1 ** self.t) 
            v_hat = self.v[p] / (1 - self.beta2 ** self.t) 

            theta = self.lr * m_hat / (v_hat ** 0.5 + self.eps)

            p.data = p.data - ndl.Tensor(
                theta, dtype=p.dtype, requires_grad=False
            )  # outofmemory now
