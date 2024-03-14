import torch
import numpy as np
from utils import load_yaml, initialize


def test(param, device, testloader, model, snr):
    variance = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in testloader:
            # Send input and target to the device
            x, y = x.to(device), y.to(device)

            if param['noise'] == 'gaussian':
                # snr = mean(signal**2)/var(noise)
                std = torch.sqrt((x**2).mean(dim=(1, 2))/(10**(snr/10))).unsqueeze(-1).unsqueeze(-1)
                x += std*torch.randn(*x.shape)

            if param['vmp']:
                # Forward pass
                if param['dataset'] == 'pems':
                    logit, var_logit = model(x)
                else:
                    raise NotImplementedError

            else:
                # Forward pass
                if param['dataset'] == 'pems':
                    logit = model(x)
                else:
                    raise NotImplementedError

            # Compute validation metrics
            if param['dataset'] == 'pems':
                pred = logit.argmax(dim=-1)
                correct += torch.eq(pred, y).int().sum().item()
                total += y.numel()
                if param['vmp']:
                    variance += torch.take_along_dim(var_logit, pred.unsqueeze(-1), dim=1).sum().item()
            else:
                raise NotImplementedError

        # Compute accuracy
        acc = 100*correct/total
        # Multiply the total by the number of classes
        var = variance/(total*param['output_dim'])
        print(f'Noise: {param["noise"]}, SNR: {snr:.1f} dB, Test Accuracy: {acc:.2f}%, Predictive Variance: {var:.3g}')


def one_test_run(param):
    device, trainloader, testloader, model, optimizer, scheduler = initialize(param)

    # Testing
    print('Start testing')
    snr_param = param['snr']
    for snr in np.arange(snr_param[0], snr_param[1], snr_param[2]):
        test(param, device, testloader, model, snr)


def main():
    param = load_yaml()
    one_test_run(param)


if __name__ == '__main__':
    main()
