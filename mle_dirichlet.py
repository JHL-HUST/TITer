import pickle
import os
import argparse
from dataset.baseDataset import baseDataset
from model.dirichlet import MLE_Dirchlet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dirichlet MLE', usage='mle_dirichlet.py [<args>] [-h | --help]')
    parser.add_argument('--data_dir', default='data/ICEWS14', type=str)
    parser.add_argument('--outfile', default='dirchlet_alphas.pkl', type=str)
    parser.add_argument('--k', default=300, type=int)
    parser.add_argument('--time_span', default=24, type=int, help='24 for ICEWS, 1 for WIKI and YAGO')
    parser.add_argument('--tol', default=1e-7, type=float)
    parser.add_argument('--method', default='meanprecision', type=str)
    parser.add_argument('--maxiter', default=100, type=int)
    args = parser.parse_args()

    trainF = os.path.join(args.data_dir, 'train.txt')
    testF = os.path.join(args.data_dir, 'test.txt')
    statF = os.path.join(args.data_dir, 'stat.txt')
    validF = os.path.join(args.data_dir, 'valid.txt')
    if not os.path.exists(validF):
        validF = None
    dataset = baseDataset(trainF, testF, statF, validF)

    mle_d = MLE_Dirchlet(dataset.trainQuadruples, dataset.num_r, args.k, args.time_span,
                         args.tol, args.method, args.maxiter)
    pickle.dump(mle_d.alphas, open(os.path.join(args.data_dir, args.outfile), 'wb'))
