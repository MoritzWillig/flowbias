import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import torch.nn as nn

from flowbias.models.pwcnet import PWCNet

model = PWCNet({})
l = [module for module in model.modules() if type(module) != nn.Sequential]
print(l)