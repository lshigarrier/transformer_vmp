import torch
import numpy as np
from utils import load_yaml, initialize
from vmp import softmax_vmp


def test(param, device, testloader, model, lvl):
    variance = 0
    correct = 0
    total = 0
    snr_sum = 0
    std_sum = 0

    with torch.no_grad():

        for x, y in testloader:
            # Send input and target to the device
            x, y = x.to(device), y.to(device)

            if param['noise'] == 'gaussian':

                if param['snr']:
                    # lvl is the snr
                    # snr = mean(signal**2)/var(noise)
                    std = torch.sqrt((x**2).mean(dim=(1, 2))/(10**(lvl/10))).unsqueeze(-1).unsqueeze(-1)
                    std_sum += std.sum().item()
                    x += std*torch.randn(*x.shape)
                else:
                    # lvl is the noise std
                    noise = lvl*torch.randn(*x.shape)
                    snr_sum += ((x**2).sum(dim=(1, 2))/(noise**2).sum(dim=(1, 2))).sum().item()
                    x += noise

            if param['vmp']:
                # Forward pass
                if param['dataset'] == 'pems':
                    logit, var_logit = model(x)
                    _, var_soft = softmax_vmp(logit, var_logit)
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
                    # variance += torch.take_along_dim(var_logit, pred.unsqueeze(-1), dim=1).sum().item()
                    variance += torch.take_along_dim(var_soft, pred.unsqueeze(-1), dim=1).sum().item()
            else:
                raise NotImplementedError

        # Compute accuracy
        acc = 100*correct/total
        if param['snr']:
            print(f'Noise: {param["noise"]}, SNR: {lvl:.1f} dB, std: {std_sum/total:.3g}, '
                  f'Test Accuracy: {acc:.2f}%, Predictive Variance: {variance/total:.3g}')
        else:
            mean_snr = 10 * np.log10(snr_sum / total)
            print(f'Noise: {param["noise"]}, SNR: {mean_snr:.3g} dB, std: {lvl:.3g}, '
                  f'Test Accuracy: {acc:.2f}%, Predictive Variance: {variance/total:.3g}')

        # Print samples
        if param['sample']:
            # nb_sample = min(5, len(y))
            correct_flag = torch.eq(pred, y).int()
            idx = [torch.argmax(correct_flag), torch.argmin(correct_flag)]
            for i in idx:
                print(f'------------------------------------------\nSample n°{i}')
                print(f'True class: {y[i]}')
                print(f'Predicted class: {pred[i]}')
                print(f'Softmax probabilities:\n{torch.nn.functional.softmax(logit[i], dim=0)}')
                if param['vmp']:
                    # print(f'Predictive variances:\n{var_logit[i]}')
                    print(f'Predictive variances:\n{var_soft[i]}')
                print('\n')


def one_test_run(param):
    device, trainloader, testloader, model, optimizer, scheduler = initialize(param)

    # Testing
    print('Start testing\n')
    level = param['level']
    for lvl in np.arange(level[0], level[1], level[2]):
        test(param, device, testloader, model, lvl)


def test_fct(lvl, batch):
    t = torch.rand(batch, 3, 3)
    noise = lvl*torch.randn(*t.shape)
    snr1 = ((t ** 2).sum(dim=(1, 2)) / (noise ** 2).sum(dim=(1, 2))).sum().item()
    snr1_db = 10 * np.log10(snr1 / batch)
    snr2_db = (10 * np.log10((t ** 2).sum(dim=(1, 2)) / (noise ** 2).sum(dim=(1, 2)))).mean().item()
    print(f'snr1_db={snr1_db}')
    print(f'snr2_db={snr2_db}')


def main():
    param = load_yaml()
    one_test_run(param)


if __name__ == '__main__':
    # for std in [0.01, 0.05, 0.1, 0.2]:
    #     print(f'std={std}')
    #     test_fct(std, 100)
    main()
