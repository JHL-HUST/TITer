import torch

class ReactiveBaseline(object):
    def __init__(self, config, update_rate):
        self.update_rate = update_rate
        self.value = torch.zeros(1)
        if config['cuda']:
            self.value = self.value.cuda()

    def get_baseline_value(self):
        return self.value

    def update(self, target):
        self.value = torch.add((1 - self.update_rate) * self.value, self.update_rate * target)