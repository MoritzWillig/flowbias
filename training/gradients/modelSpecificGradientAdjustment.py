
class ExpertAddModelGradientAdjustment:

    def __init__(self, args, num_experts):
        self._factor = 1.0 / num_experts

    def adjust_gradients(self, model):
        for p in model.feature_pyramid_extractor.convsBase.parameters():
            p.grad *= self._factor
        for p in model.flow_estimators.conv1_base.parameters():
            p.grad *= self._factor
        for p in model.flow_estimators.conv2_base.parameters():
            p.grad *= self._factor
        for p in model.flow_estimators.conv3_base.parameters():
            p.grad *= self._factor
        for p in model.flow_estimators.conv4_base.parameters():
            p.grad *= self._factor
        for p in model.flow_estimators.conv5_base.parameters():
            p.grad *= self._factor
        for p in model.flow_estimators.conv_last_base.parameters():
            p.grad *= self._factor
        for p in model.context_networks.convs_base.parameters():
            p.grad *= self._factor


class ExpertAddModelGradientAdjustment4(ExpertAddModelGradientAdjustment):

    def __init__(self, args):
        super().__init__(args, 4)


class FullModelGradientAdjustment:

    def __init__(self, args, adjustment_factor):
        self._adjustment_factor = adjustment_factor

    def adjust_gradients(self, model):
        for p in model.context_networks.convs_base.parameters():
            p.grad *= self._adjustment_factor


class FullModelGradientAdjustment025(FullModelGradientAdjustment):

    def __init__(self, args):
        super().__init__(args, 0.25)
