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
                0 + 2 * np.sin(2 * i * 2 * np.pi / 1024) \
                    + 2 * np.sin((4 * i * 2 * np.pi / 1024))
                )

input_size = 100+1
hidden_size = [50]#, 100, 100]
output_size = 1

repeats = 1
lr = 0.0015
stdev = 0.1

generations = 100
pop_size = 100
elite_size = 5
elite_mutate = 3
nets = []
for n in range(pop_size):
    nets.append(NeuralNet(input_size, hidden_size, output_size, non_spiking_output=True))
    nets[-1].set_pde_params(setting_type='normal')

duration = 1024
spike_times = build_input_spike_train(num_repeats=1,
                                       cycle_time=duration,
                                       pop_size=input_size-1,
                                       use_50=True)
input_spikes = times_to_spikes(spike_times, duration, stack_bias=50)

all_outputs = []
epoch_error = []
with torch.no_grad():
    for g in range(generations):
        for r in range(repeats):
            iteration_error = []
            clipped_duration = int((duration * 0.1) + (duration * 0.9 * g / (generations-1)))
            for i in range(clipped_duration):
                net_error = []
                for net in nets:
                    err = net.activation[-1].v - target_data[i]
                    if not torch.isnan(err):
                        output = net.pde_forward(torch.tensor(input_spikes[:, i], dtype=torch.float32),
                                                 err, lr)
                    # all_outputs.append(output)
                    # error = torch.square(output - target_data[i])
                    error = err#output - target_data[i]
                    # sign = -1 if output - target_data[i] < 0 else 1
                    # net.scale_weights(-error * lr)
                    # net.scale_weights(sign * -error * lr)
                    net_error.append(np.round(error.item(), 2))
                print(i+1, "/", clipped_duration, "iteration error = ", net_error)
                iteration_error.append(net_error)
            epoch_error.append(np.round(np.mean(np.square(iteration_error), axis=0), 4))
            # epoch_error.append(np.mean(iteration_error, axis=1))
            print("epoch", r+1, "/", repeats, "error =", epoch_error)
            for n in range(len(nets)):
                print("Coefficients =", np.round(nets[n].pde_coef, 2), "achieved an error of", epoch_error[-1][n])
            print("epoch", r+1, "/", repeats, "error =", epoch_error)
            print("Stats across epochs")
            for e, err in enumerate(epoch_error):
                print("{} - min: {} - ave: {} - max:{} - {} failed".format(e,
                                                               np.round(np.nanmin(err), 2),
                                                               np.round(np.nanmean(err), 2),
                                                               np.round(np.nanmax(err), 2),
                                                               np.sum(np.isnan(err))))
        min_err = np.nanmin(epoch_error[-1])
        max_err = np.nanmax(epoch_error[-1])
        err_range = max_err - min_err
        rescale_err = 1 - ((np.array(epoch_error[-1]) - min_err) / err_range)
        rescale_err /= np.nansum(rescale_err)
        failure = np.isnan(epoch_error[-1])
        weighted_pde_coef = np.zeros_like(nets[0].pde_coef)
        for n, fail in enumerate(failure):
            if not fail:
                weighted_pde_coef += nets[n].pde_coef * rescale_err[n]
        elite = torch.topk(torch.tensor(epoch_error[-1]), elite_size, largest=False)[1].cpu().numpy()
        elite_coef = []
        for e_idx in elite:
            elite_coef.append(nets[e_idx].pde_coef)
        nets = []
        for e_c in elite_coef:
            nets.append(NeuralNet(input_size, hidden_size, output_size, non_spiking_output=True))
            nets[-1].set_pde_params(setting_type=e_c)
            for i in range(elite_mutate):
                nets.append(NeuralNet(input_size, hidden_size, output_size, non_spiking_output=True))
                nets[-1].set_pde_params(setting_type='ES', start=e_c, stop=stdev)
        for n in range(pop_size-elite_size):
            nets.append(NeuralNet(input_size, hidden_size, output_size, non_spiking_output=True))
            nets[-1].set_pde_params(setting_type='ES', start=weighted_pde_coef, stop=stdev)


print("done")