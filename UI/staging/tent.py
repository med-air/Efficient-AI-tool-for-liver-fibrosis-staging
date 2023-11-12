from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x_0, x_1,mode = False):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x_0, x_1, self.model, self.optimizer, mode)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def softmax_entropy_(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    x[x<0.5] = 1 - x[x<0.5]
    return -(x[:,1:] * x[:,1:].log()).sum(1)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x_0, x_1, model, optimizer,mode):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    if mode == False:
        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-6)
        # forward
        outputs = model(x_0)
        # adapt
        outputs = outputs.view(-1,x_0.shape[1],4)
        outputs = torch.mean(outputs,dim=1)
        #output = output[:,-1,:]
        outputs = outputs.view(outputs.shape[0],4)
        outputs = torch.sigmoid(outputs)

        outputs_1 = model(x_1)
        # adapt
        outputs_1 = outputs_1.view(-1,x_1.shape[1],4)
        outputs_1 = torch.mean(outputs_1,dim=1)
        #output = output[:,-1,:]
        outputs_1 = outputs_1.view(outputs_1.shape[0],4)
        outputs_1 = torch.sigmoid(outputs_1)
        cos_similarity = cos(outputs,outputs_1).mean()

        optimizer.zero_grad()
        loss = 1-cos_similarity
        loss.backward()
        # print(loss)
        optimizer.step()
    else:
        model.eval()
        outputs = model(x_0)
        # adapt
        outputs = outputs.view(-1,x_0.shape[1],4)
        outputs = torch.mean(outputs,dim=1)
        #output = output[:,-1,:]
        outputs = outputs.view(outputs.shape[0],4)
        outputs = torch.sigmoid(outputs)
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            # m.track_running_stats = False
            # m.running_mean = None
            # m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
