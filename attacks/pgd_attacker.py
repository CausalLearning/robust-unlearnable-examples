import torch


class PGDAttacker():
    def __init__(self, radius, steps, step_size, random_start, norm_type, ascending=True):
        self.radius = radius / 255.
        self.steps = steps
        self.step_size = step_size / 255.
        self.random_start = random_start
        self.norm_type = norm_type
        self.ascending = ascending

    def perturb(self, model, criterion, x, y):
        if self.steps==0 or self.radius==0:
            return x.clone()

        adv_x = x.clone()
        if self.random_start:
            if self.norm_type == 'l-infty':
                adv_x += 2 * (torch.rand_like(x) - 0.5) * self.radius
            else:
                adv_x += 2 * (torch.rand_like(x) - 0.5) * self.radius / self.steps
            self._clip_(adv_x, x)

        ''' temporarily shutdown autograd of model to improve pgd efficiency '''
        model.eval()
        for pp in model.parameters():
            pp.requires_grad = False

        for step in range(self.steps):
            adv_x.requires_grad_()
            _y = model(adv_x)
            loss = criterion(_y, y)
            grad = torch.autograd.grad(loss, [adv_x])[0]

            with torch.no_grad():
                if not self.ascending: grad.mul_(-1)

                if self.norm_type == 'l-infty':
                    adv_x.add_(torch.sign(grad), alpha=self.step_size)
                else:
                    if self.norm_type == 'l2':
                        grad_norm = (grad.reshape(grad.shape[0],-1)**2).sum(dim=1).sqrt()
                    elif self.norm_type == 'l1':
                        grad_norm = grad.reshape(grad.shape[0],-1).abs().sum(dim=1)
                    grad_norm = grad_norm.reshape( -1, *( [1] * (len(x.shape)-1) ) )
                    scaled_grad = grad / (grad_norm + 1e-10)
                    adv_x.add_(scaled_grad, alpha=self.step_size)

                self._clip_(adv_x, x)

        ''' reopen autograd of model after pgd '''
        for pp in model.parameters():
            pp.requires_grad = True

        return adv_x.data

    def _clip_(self, adv_x, x):
        adv_x -= x
        if self.norm_type == 'l-infty':
            adv_x.clamp_(-self.radius, self.radius)
        else:
            if self.norm_type == 'l2':
                norm = (adv_x.reshape(adv_x.shape[0],-1)**2).sum(dim=1).sqrt()
            elif self.norm_type == 'l1':
                norm = adv_x.reshape(adv_x.shape[0],-1).abs().sum(dim=1)
            norm = norm.reshape( -1, *( [1] * (len(x.shape)-1) ) )
            adv_x /= (norm + 1e-10)
            adv_x *= norm.clamp(max=self.radius)
        adv_x += x
        adv_x.clamp_(-0.5, 0.5)
