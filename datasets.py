import numpy as np
import pandas as pd
import torch
from scipy.io import arff


def load_pems(dataset='TRAIN'):
    """
    PEMS-SF_<TRAIN-TEST>.arff --> data, meta
    meta :
    Dataset: 'PEMS-SF'
        relationalAtt's type is relational
        classAttribute's type is nominal, range is ('1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0')
    data :
    len(TRAIN) = 267 ; len(TEST) = 173
    data['relationalAtt'] : (len,) ; dtype : 963 x 144 x (att, <f8)
    data['classAttribute'] : (len,) ; dtype : S3
    """
    path = './data/PEMS-SF/PEMS-SF_'+dataset+'.arff'
    data, meta = arff.loadarff(path)
    # nb of features = 963 ; sequence length = 144
    x = np.zeros(shape=(len(data), 963, 144), dtype='<f8')
    # Conversion to ndarray
    for i in range(len(data)):
        x[i] = pd.DataFrame(data['relationalAtt'][i]).astype(float).to_numpy()
    y = pd.DataFrame(data['classAttribute']).astype(float).to_numpy()
    # y must be squeezed for proper functioning of one_hot
    # Remove 1 to have range [0, 6]
    y = y.squeeze() - 1
    # Normalization
    x_min = x.min()
    x = (x - x_min)/(x.max() - x_min)
    return torch.Tensor(x).transpose(dim0=1, dim1=2), torch.Tensor(y).long()


def main():
    x, y = load_pems(dataset='TEST')
    print(x[0, 0])
    print(y[:50])
    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    main()
