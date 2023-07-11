import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from snn import NeuralNet

gpu = True
if gpu:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'

# weight_scale = [-2, -1, 10]
# weights = np.logspace(*weight_scale)
weight_scale = [0.03, 0.14, 100]
weights = np.linspace(*weight_scale)

repeats = 1000
prob = 0.1

width = 100
inputs = width
hidden_size = width
layers = 5
output = width
hidden = [hidden_size for i in range(layers)]

binomial_input = torch.distributions.binomial.Binomial(probs=prob)

with torch.no_grad():
    results = {}
    for weight in weights:
        results[weight] = {'input': [], 'output': []}
        for i in range(repeats):
            print(weight, '-', i+1, '/', repeats)
            net = NeuralNet(inputs, hidden, output)
            net.scale_weights(weight)
            input_spikes = binomial_input.sample([inputs])
            output_spikes = net.forward(input_spikes)
            results[weight]['input'].append(input_spikes)
            results[weight]['output'].append(output_spikes)
        # print(results)
        for w in results:
            output_spikes = results[w]['output']
            num_spikes = torch.stack([torch.sum(spikes) for spikes in output_spikes])
            ave_out = torch.mean(num_spikes)
            std_out = torch.std(num_spikes)
            print(w, 'mean:', ave_out, 'std:', std_out)

print('Setting up for plotting')
res_3d = []
for w in results:
    output_spikes = results[w]['output']
    num_spikes = torch.stack([torch.sum(spikes) for spikes in output_spikes])
    ave_out = torch.mean(num_spikes)
    std_out = torch.std(num_spikes)
    res_3d.append(np.array([w, np.float(ave_out), np.float(std_out)]))

res_3d = np.array(res_3d)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(res_3d[:, 0], res_3d[:, 1], res_3d[:, 2])
plt.show()

print('done')
