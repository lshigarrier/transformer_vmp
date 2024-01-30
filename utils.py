import yaml
import argparse
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from datasets import load_pems
from attention import TransformerClassifier
from attention_vmp import TransformerClassifierVMP


def load_yaml(file_name=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(name_or_flag='--config', type=str)
    args = parser.parse_args()
    if file_name:
        yaml_file = f'config/{file_name}.yml'
    elif args.config:
        yaml_file = f'config/{args.config}.yml'
    else:
        raise RuntimeError
    with open(yaml_file) as file:
        param = yaml.load(file, Loader=yaml.FullLoader)
    return param


def initialize(param):
    # Print param
    for key in param:
        print(f'{key}: {param[key]}')

    # Detect anomaly in autograd
    torch.autograd.set_detect_anomaly(True)

    # Deterministic
    torch.manual_seed(param['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    # Declare CPU/GPU usage
    if param['gpu_number'] is not None:
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = param['gpu_number']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # Create model
    print('Initialize model')

    # Load datasets and create model
    if param['dataset'] == 'pems':
        x_train, y_train = load_pems(dataset='TRAIN')
        x_test, y_test = load_pems(dataset='TEST')
        if param['vmp']:
            model = TransformerClassifierVMP(param, device)
        else:
            model = TransformerClassifier(param, device)
    else:
        raise NotImplementedError

    # Load model
    if param['load']:
        statedict = torch.load(f=f'models/{param["name"]}/{param["model"]}', map_location='cpu')
        model.load_state_dict(statedict)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()

    # Create data loaders
    trainset = TensorDataset(x_train, y_train)
    trainloader = DataLoader(trainset, batch_size=param['batch_size'],
                             shuffle=True, pin_memory=True, num_workers=param['workers'])
    testset = TensorDataset(x_test, y_test)
    testloader = DataLoader(testset, batch_size=param['batch_size'],
                            shuffle=False, pin_memory=True, num_workers=param['workers'])

    # Set optimizer
    if param['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    elif param['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'], weight_decay=param['l2_reg'])
    else:
        raise NotImplementedError

    # Set scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=param['end_factor'],
                                                  total_iters=param['epochs'])

    return device, trainloader, testloader, model, optimizer, scheduler
