import numpy as np
import torch
from tests.frozen_poisson import build_input_spike_train, times_to_spikes
from snn import NeuralNet

gpu = True
if gpu:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'

target_data = []
for i in range(1024):
            target_data.append(#1)
                1 + 2 * np.sin(2 * i * 2 * np.pi / 1024) \
                    + 2 * np.sin((4 * i * 2 * np.pi / 1024))
                )

input_size = 100+1
hidden_size = [100, 100, 100]
output_size = 1

repeats = 100
lr = 0.0000015

net = NeuralNet(input_size, hidden_size, output_size, non_spiking_output=True)

duration = 1024
spike_times = build_input_spike_train(num_repeats=1,
                                       cycle_time=duration,
                                       pop_size=100,
                                       use_50=True)
input_spikes = times_to_spikes(spike_times, duration, stack_bias=50)

all_outputs = []
epoch_error = []
with torch.no_grad():
    for r in range(repeats):
        iteration_error = []
        for i in range(duration):
            output = net.timed_forward(torch.tensor(input_spikes[:, i], dtype=torch.float32))
            all_outputs.append(output)
            # error = torch.square(output - target_data[i])
            error = output - target_data[i]
            # sign = -1 if output - target_data[i] < 0 else 1
            net.scale_weights(-error * lr)
            # net.scale_weights(sign * -error * lr)
            iteration_error.append(error.item())
        # epoch_error.append(np.round(np.mean(np.sqrt(iteration_error)), 4))
        epoch_error.append(np.mean(iteration_error))
        print("epoch", r+1, "/", repeats, "error =", epoch_error)


print("done")