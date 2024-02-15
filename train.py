import time
import torch
import wandb
import torch.nn as nn
from utils import load_yaml, initialize
# from test import test
from vmp import loss_vmp


def compute_metrics(param, idx, trainloader, model, target, logit):
    # Compute average gradient norm
    grad_sum = 0
    grad_tot = 0
    grad_var_sum = 0
    grad_var_tot = 0
    grad_mu_sum = 0
    grad_mu_tot = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            grad_tot += 1
            grad_sum += p.grad.norm().item()
            if 'rho' in name:
                grad_var_tot += 1
                grad_var_sum += p.grad.norm().item()
            else:
                grad_mu_tot += 1
                grad_mu_sum += p.grad.norm().item()

    if param['wandb']:
        wandb.log({'average_gradient_norm': grad_sum/grad_tot,
                   'variance_gradient_norm': grad_var_sum/grad_var_tot, 'mu_gradient_norm': grad_mu_sum/grad_mu_tot})

    # Compute training set accuracy
    if param['dataset'] == 'pems':
        # Compute predictions
        pred = logit.argmax(dim=-1)

        # Compute the nb of correct and total examples
        train_corr = torch.eq(pred, target).int().sum().item()
        train_num = target.numel()

        if param['wandb']:
            onehot = nn.functional.one_hot(target, num_classes=param['output_dim'])
            prob = nn.functional.softmax(logit, dim=-1)
            wandb.log({'train_MSE': ((onehot - prob)**2).mean().item()})
    else:
        raise NotImplementedError

    # Printing ~4 times per epoch
    # The max(.,1) is here to avoid exception if int(len(trainloader)/4) = 0
    if idx % max(int(len(trainloader)/4), 1) == 0:
        if idx != len(trainloader)-1:
            total = idx*param['batch_size']
        else:
            total = (idx - 1)*param['batch_size'] + pred.shape[0]
        print(f'{total}/{len(trainloader.dataset)} {100*idx/len(trainloader):.0f}%')

    return train_corr, train_num


def validate(param, device, testloader, model):
    print('Validation')
    model.eval()

    tot_loss = 0
    tot_corr = 0
    tot_num = 0

    with torch.no_grad():

        for x, y in testloader:
            # Send input and target to the device
            x, y = x.to(device), y.to(device)

            if param['vmp']:
                # Forward pass
                if param['dataset'] == 'pems':
                    logit, var_logit = model(x)
                else:
                    raise NotImplementedError

                # Compute the ELBO
                nll, kl, var_prob, _ = loss_vmp(logit, var_logit, y, model, param)
                loss = nll + param['kl_factor']*kl

                if param['wandb']:
                    wandb.log({'test_nll': nll.item(), 'test_kl': kl.item(),
                               'test_var': var_prob.mean().item()})

            else:
                # Forward pass
                if param['dataset'] == 'pems':
                    logit = model(x)
                    # Compute the loss
                    loss = nn.functional.cross_entropy(logit, y)
                else:
                    raise NotImplementedError

            tot_loss += loss.item()
            if param['wandb']:
                wandb.log({'test_loss': loss.item()})

            # Compute validation metrics
            if param['dataset'] == 'pems':
                pred = logit.argmax(dim=-1)
                tot_corr += torch.eq(pred, y).int().sum().item()
                tot_num += y.numel()
                if param['wandb']:
                    onehot = nn.functional.one_hot(y, num_classes=param['output_dim'])
                    prob = nn.functional.softmax(logit, dim=-1)
                    wandb.log({'test_MSE': ((onehot - prob)**2).mean().item()})
            else:
                raise NotImplementedError

        # Final print
        acc = 100*tot_corr/tot_num
        if param['wandb']:
            wandb.log({'test_accuracy': acc})
        test_loss = tot_loss/len(testloader)
        print(f'Test loss: {test_loss:.3g}, Test Accuracy: {acc:.2f}%')

    return acc


def train(param, device, trainloader, testloader, model, optimizer):
    train_tot_corr = 0
    train_tot_num = 0

    for idx, (x, y) in enumerate(trainloader):
        # Gradients are set to 0
        optimizer.zero_grad()

        # Send input and target to the device
        x, y = x.to(device), y.to(device)

        if param['vmp']:
            # Forward pass
            if param['dataset'] == 'pems':
                logit, var_logit = model(x)
            else:
                raise NotImplementedError

            # Compute the ELBO
            nll, kl, var_prob, (var_par, log_var, mse_inv) = loss_vmp(logit, var_logit, y, model, param)
            loss = nll + param['kl_factor']*kl

            if param['wandb']:
                wandb.log({'train_nll': nll.item(), 'train_kl': kl.item(), 'train_var': var_prob.mean().item(),
                           'variance_param': var_par, 'log_var': log_var, 'mse_inv': mse_inv})

        else:
            # Forward pass
            if param['dataset'] == 'pems':
                logit = model(x)
                # Compute the loss
                loss = nn.functional.cross_entropy(logit, y)
            else:
                raise NotImplementedError

        # Backpropagation
        loss.backward()

        # Clipping the gradients
        nn.utils.clip_grad_norm_(model.parameters(), param['clip'])

        # Optimizer step
        optimizer.step()

        if param['wandb']:
            wandb.log({'train_loss': loss.item()})

        # Compute metrics
        train_corr, train_num = compute_metrics(param, idx, trainloader, model, y, logit)
        train_tot_corr += train_corr
        train_tot_num += train_num

    # Validation
    train_acc = 100*train_tot_corr/train_tot_num
    if param['wandb']:
        wandb.log({'train_accuracy': train_acc})
    return validate(param, device, testloader, model)


def training(param, device, trainloader, testloader, model, optimizer, scheduler):
    print('Start training')
    tac = time.time()
    best_acc = 0
    best_epoch = 0

    for epoch in range(1, param['epochs']+1):
        print(f'Epoch {epoch}')
        tic = time.time()

        # Train for one epoch
        test_acc = train(param, device, trainloader, testloader, model, optimizer)

        # Update the learning rate scheduler
        scheduler.step()
        if param['wandb']:
            wandb.log({'lr_scheduler': scheduler.get_last_lr()[0]})

        print(f'Epoch training time (s): {time.time() - tic:.0f}')

        # Save model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            print('Saving model')
            torch.save(model.state_dict(), f=f'models/{param["name"]}/weights.pt')

    print('Saving final model')
    torch.save(model.state_dict(), f=f'models/{param["name"]}/final.pt')
    print(f'Best epoch: {best_epoch}')
    print(f'Best accuracy: {best_acc:.2f}%')
    print(f'Training time (s): {time.time() - tac:.0f}')


def one_run(param):
    device, trainloader, testloader, model, optimizer, scheduler = initialize(param)

    # Training
    training(param, device, trainloader, testloader, model, optimizer, scheduler)

    # Testing
    # print('Start testing')
    # test(param, device, testloader, model)


def main():
    param = load_yaml()
    if param['wandb']:
        wandb.init(project='transformer_vmp', config=param)
    one_run(param)


if __name__ == '__main__':
    main()
