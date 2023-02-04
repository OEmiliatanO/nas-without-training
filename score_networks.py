import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
from scores import get_score_func
from scipy import stats
from pycls.models.nas.nas import Cell
from utils import add_dropout, init_network
import score.net_score

parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='../NAS-Bench-201.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
parser.add_argument('--save_string', default='naswot', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--maxofn', default=1, type=int, help='score is the max of this many evaluations of the network')
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


def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y, out = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), out.detach()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
savedataset = args.dataset
dataset = 'fake' if 'fake' in args.dataset else args.dataset
args.dataset = args.dataset.replace('fake', '')
if args.dataset == 'cifar10':
    args.dataset = args.dataset + '-valid'
searchspace = nasspace.get_search_space(args)
if 'valid' in args.dataset:
    args.dataset = args.dataset.replace('-valid', '')
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
os.makedirs(args.save_loc, exist_ok=True)

filename = f'{args.save_loc}/{args.save_string}_{args.score}_{args.nasspace}_{savedataset}{"_" + args.init + "_" if args.init != "" else args.init}_{"_dropout" if args.dropout else ""}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}'
accfilename = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{savedataset}_{args.trainval}'

if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'


scores = np.zeros(len(searchspace))
try:
    accs = np.load(accfilename + '.npy')
except:
    accs = np.zeros(len(searchspace))

scores_nas = []
scores_gu = []
arches = np.random.randint(0, 15625, 100)
for arch in arches:
    uid = searchspace[arch]
    network = searchspace.get_net(uid)
    score_nas.append(score_.score_nas(network, train_loader, device, args))
    score_gu.append(score_.score_gu(network, train_loader, device, args))

scores_nas = np.array(scores_nas)
scores_gu = np.array(scores_gu)
calstd = lambda x: np.ma.masked_invalid(x).std()
calmean = lambda x: np.ma.masked_invalid(x).mean()
stds = {"nas": calstd(scores_nas), "gu": calstd(scores_gu)}
means = {"nas": calmean(scores_nas), "gu": calmean(scores_gu)}

for i, (uid, network) in enumerate(searchspace):
    # Reproducibility
    try:
        scores[i] = net_score.socres(network, train_loader, device, stds, means, args)
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        accs_ = accs[~np.isnan(scores)]
        scores_ = scores[~np.isnan(scores)]
        numnan = np.isnan(scores).sum()
        tau, p = stats.kendalltau(accs_[:max(i-numnan, 1)], scores_[:max(i-numnan, 1)])
        print(f'{tau}')
        if i % 1000 == 0:
            np.save(filename, scores)
            np.save(accfilename, accs)
    except Exception as e:
        print(e)
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        scores[i] = np.nan
np.save(filename, scores)
np.save(accfilename, accs)
