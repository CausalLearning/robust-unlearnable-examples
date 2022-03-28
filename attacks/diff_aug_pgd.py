import torch


class DiffAugPGDAttacker():
    def __init__(self, samp_num, trans,
        radius, steps, step_size, random_start, ascending=True):
        self.samp_num     = samp_num
        self.trans        = trans

        self.radius       = radius / 255.
        self.steps        = steps
        self.step_size    = step_size / 255.
        self.random_start = random_start
        self.ascending    = ascending

    def perturb(self, model, criterion, x, y):
        ''' initialize noise '''
        delta = torch.zeros_like(x.data)
        if self.steps==0 or self.radius==0:
            return delta

        if self.random_start:
            delta.uniform_(-self.radius, self.radius)

        ''' temporarily shutdown autograd of model to improve pgd efficiency '''
        model.eval()
        for pp in model.parameters():
            pp.requires_grad = False

        delta.requires_grad_()
        for step in range(self.steps):
            delta.grad = None

            for i in range(self.samp_num):
                adv_x = (self.trans(x) + delta).clamp(0., 255.)
                _y = model(adv_x)
                lo = criterion(_y, y)
                lo.backward()

            with torch.no_grad():
                grad = delta.grad.data
                if not self.ascending: grad.mul_(-1)
                delta.add_(torch.sign(grad), alpha=self.step_size)
                delta.clamp_(-self.radius, self.radius)

        ''' reopen autograd of model after pgd '''
        for pp in model.parameters():
            pp.requires_grad = True

        return delta.data
