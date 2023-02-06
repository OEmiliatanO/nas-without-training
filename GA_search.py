import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
from scores import get_score_func
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
from statistics import mean
import time
from utils import add_dropout
from search import GA
from score import net_score


parser = argparse.ArgumentParser(description='NAS Without Training')

parser.add_argument('--maxn_pop', default=30, type=int, help='number of population')
parser.add_argument('--maxn_iter', default=50, type=int, help='number of iteration')
parser.add_argument('--prob_mut', default=0.07, type=float, help='probability of mutation')
parser.add_argument('--prob_cr', default=0.8, type = float, help='probability of crossover')

parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='../NAS-Bench-201.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results/ICML', type=str, help='folder to save results')
parser.add_argument('--save_string', default='naswot', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--kernel', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--activations', action='store_true')
parser.add_argument('--cosine', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=500, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
searchspace = nasspace.get_search_space(args)
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
os.makedirs(args.save_loc, exist_ok=True)


times     = []
chosen    = []
acc       = []
val_acc   = []
topscores = []
order_fn = np.nanargmax


if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'

scores_nas = []
scores_gu = []
arches = np.random.randint(0, 15625, 100)
for arch in arches:
    uid = searchspace[arch]
    network = searchspace.get_network(uid)
    scores_nas.append(net_score.score_nas(network, train_loader, device, args))
    scores_gu.append(net_score.score_gu(network, train_loader, device, args))

scores_nas = np.array(scores_nas)
scores_gu = np.array(scores_gu)
calstd = lambda x: np.ma.masked_invalid(x).std()
calmean = lambda x: np.ma.masked_invalid(x).mean()
stds = {"nas": calstd(scores_nas), "gu": calstd(scores_gu)}
means = {"nas": calmean(scores_nas), "gu": calmean(scores_gu)}

print(f"parameter:\nnumber of population={args.maxn_pop}\nnumber of iteration={args.maxn_iter}\nprobability of mutation={args.prob_mut}\nprobability of crossover={args.prob_cr}")

runs = trange(args.n_runs, desc='acc: nan  topscores: nan')
for N in runs:
    start = time.time()

    # nas-bench-201 spec
    sol = GA.GA(6, 5, searchspace, train_loader, device, stds, means, acc_type, args)
    score, acc_, uid = sol.find_best()
    chosen.append(uid)
    topscores.append(score)
    acc.append(acc_)

    if not args.dataset == 'cifar10' or args.trainval:
        val_acc.append(searchspace.get_final_accuracy(uid, val_acc_type, args.trainval))
    #    val_acc.append(info.get_metrics(dset, val_acc_type)['accuracy'])

    times.append(time.time()-start)
    runs.set_description(f"acc: {mean(acc):.2f}%  topscores:{mean(topscores):.2f}  time:{mean(times):.2f}")

print(f"Final mean test accuracy: {np.mean(acc)}")
#if len(val_acc) > 1:
#    print(f"Final mean validation accuracy: {np.mean(val_acc)}")

state = {'accs': acc,
         'chosen': chosen,
         'times': times,
         'topscores': topscores,
         }

dset = args.dataset if not (args.trainval and args.dataset == 'cifar10') else 'cifar10-valid'
fname = f"{args.save_loc}/{args.save_string}_{args.score}_{args.nasspace}_{dset}_{args.kernel}_{args.dropout}_{args.augtype}_{args.sigma}_{args.repeat}_{args.batch_size}_{args.n_runs}_{args.n_samples}_{args.seed}.t7"
torch.save(state, fname)
