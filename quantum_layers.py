from quantum_circuits import *
import torch
import torch.nn as nn
from torch.autograd import Function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class HybridFunction(Function):
    
    @staticmethod
    def forward(ctx, inputs, quantum_circuit, shift):
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = [ctx.quantum_circuit.run(input.tolist()) for input in inputs]
        result = torch.tensor(expectation_z).cuda()

        ctx.save_for_backward(inputs, result)
        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift
        
        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left  = ctx.quantum_circuit.run(shift_left[i])
            
            gradient = torch.tensor([expectation_right]).cuda() - torch.tensor([expectation_left]).cuda()
            gradients.append(gradient)
        
        gradients = torch.tensor([gradients]).cuda()
        gradients = torch.transpose(gradients, 0, 1)

        return gradients.float() * grad_output.float(), None, None

class Hybrid(nn.Module):
    
    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift
        
    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)




#Custom forward/backward pass for Pytorch
class QuanvFunction(Function):
    @staticmethod
    def forward(ctx, inputs, in_channels, out_channels, kernel_size, quantum_circuits, shift, verbose=True):
        """
           input  shape : (batch_size, feature_size, length, length)
           otuput shape : (batch_size, feature_size', length, length')
        """
        ctx.in_channels      = in_channels
        ctx.out_channels     = out_channels
        ctx.kernel_size      = kernel_size
        ctx.quantum_circuits = quantum_circuits
        ctx.shift            = shift

        _, _, len_x, len_y = inputs.size()
        len_x = len_x - kernel_size + 1
        len_y = len_y - kernel_size + 1
        
        features = []
        ## loop over the images
        for input in inputs:
            feature = []
            ## loop over the circuits
            for circuit in quantum_circuits:
                # save the results
                xys = []
                for x in range(len_x):
                    ys = []
                    for y in range(len_y):
                        # get the patches
                        data = input[0, x:x+kernel_size, y:y+kernel_size]
                        # store the results
                        res=circuit.run(data)
                        ys.append(res)
                    xys.append(ys)
                feature.append(xys)
            features.append(feature)
        # construct the tensor
        result = torch.tensor(features)

        ctx.save_for_backward(inputs, result)
        return result
        
    @staticmethod
    def backward(ctx, grad_output): 
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift
        
        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left  = ctx.quantum_circuit.run(shift_left[i])
            
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None


#The actual Quantum convolutional layer
class Quanv(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 backend=qiskit.providers.aer.QasmSimulator(method = "statevector_gpu"), 
                 shots=100, 
                 shift=np.pi/2):
        
        super(Quanv, self).__init__()

        self.quantum_circuits = [QuantumVCircuit(kernel_size=kernel_size, backend=backend, shots=shots, threshold=0.5) for _ in range(out_channels)]

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.shift        = shift
        
    def forward(self, inputs):
        return QuanvFunction.apply(inputs, 
                                   self.in_channels, 
                                   self.out_channels, 
                                   self.kernel_size,
                                   self.quantum_circuits, 
                                   self.shift)