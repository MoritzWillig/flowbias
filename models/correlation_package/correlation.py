import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
import correlation_cuda

class CorrelationFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2,
                pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply):
        ctx.save_for_backward(input1, input2)
        ctx.non_tensor_params = (pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)

        with torch.cuda.device_of(input1):
            rbot1 = torch.empty_like(input1)
            rbot2 = torch.empty_like(input2)
            output = torch.empty_like(input1)

            correlation_cuda.forward(input1, input2, rbot1, rbot2, output, 
                pad_size, kernel_size, max_displacement,stride1, stride2, corr_multiply)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply = ctx.non_tensor_params

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            with torch.cuda.device_of(input1):
                rbot1 = torch.empty_like(input1)
                rbot2 = torch.empty_like(input2)

                grad_input1 = torch.empty_like(input1)
                grad_input2 = torch.empty_like(input2)

                correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                    pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)
        else:
            grad_input1 = None
            grad_input2 = None

        return grad_input1, grad_input2, None,  None, None, None, None, None


class Correlation(Module):
    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=1, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        result = CorrelationFunction.apply(input1, input2, self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)

        return result

