import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate,
                                          betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, input, target, eta,
                                network_optimizer):  #,temp):
        loss = self.model._loss(input, target)  #,temp)
        parameters = [
            p for n, p in self.model.named_parameters() if p.requires_grad
        ]
        theta = _concat(parameters).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.model.parameters()).mul_(
                                 self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        #names = [n for n, p in self.model.named_parameters() if not p.requires_grad]
        #print(names)
        dtheta = _concat(torch.autograd.grad(
            loss, parameters)).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment + dtheta))  #,temp)
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta,
             network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train,
                                         input_valid, target_valid, eta,
                                         network_optimizer)  #,temp)
        else:
            self._backward_step(input_valid, target_valid)  #,temp)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):  #,temp):
        loss = self.model._loss(input_valid, target_valid)  #,temp)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid,
                                target_valid, eta,
                                network_optimizer):  #,temp):
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer)  #,temp)
        unrolled_loss = unrolled_model._loss(input_valid,
                                             target_valid)  #,temp)

        unrolled_loss.backward()
        parameters = [
            p for n, p in unrolled_model.named_parameters() if p.requires_grad
        ]
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in parameters]
        implicit_grads = self._hessian_vector_product(vector, input_train,
                                                      target_train)  #,temp)
        #print(dalpha)
        #print(vector)
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):  #,temp):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                v_length = np.prod(v.size())
                params[k] = theta[offset:offset + v_length].view(v.size())
                offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        parameters = [
            p for n, p in self.model.named_parameters() if p.requires_grad
        ]
        for p, v in zip(parameters, vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)  #),temp)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(parameters, vector):
            p.data.sub_(2 * R, v)
        loss = self.model._loss(input, target)  #,temp)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(parameters, vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
