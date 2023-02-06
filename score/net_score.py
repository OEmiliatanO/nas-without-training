import numpy as np
import torch
import random
from torch.autograd import Variable

def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld

def random_score(jacob, label=None):
    return np.random.normal()


_scores = {
        'hook_logdet': hooklogdet,
        'random': random_score
        }

def get_score_func(score_name):
    return _scores[score_name]

def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y, out = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), out.detach()

def score_nas(network, train_loader, device, args):
    #try:
    if args.dropout:
        add_dropout(network, args.sigma)
    if args.init != '':
        init_network(network, args.init)
    if 'hook_' in args.score:
        network.K = np.zeros((args.batch_size, args.batch_size))
        def counting_forward_hook(module, inp, out):
            try:
                if not module.visited_backwards:
                    return
                if isinstance(inp, tuple):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1)
                x = (inp > 0).float()
                K = x @ x.t()
                K2 = (1.-x) @ (1.-x.t())
                network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
            except:
                pass
                
        def counting_backward_hook(module, inp, out):    
            module.visited_backwards = True    

                
        for name, module in network.named_modules():    
            if 'ReLU' in str(type(module)):    
                #hooks[name] = module.register_forward_hook(counting_hook)    
                module.register_forward_hook(counting_forward_hook)    
                module.register_backward_hook(counting_backward_hook)    

    network = network.to(device)    
    s = []

    data_iterator = iter(train_loader)
    x, target = next(data_iterator)
    x2 = torch.clone(x)
    x2 = x2.to(device)
    x, target = x.to(device), target.to(device)
    jacobs, labels, y, out = get_batch_jacobian(network, x, target, device, args)

    if 'hook_' in args.score:
        network(x2.to(device))
        s.append(get_score_func(args.score)(network.K, target))
    else:
        s.append(get_score_func(args.score)(jacobs, labels))
    
    return np.mean(s)
    #except Exception as e:
    #    print(e)
    #    return np.nan

def score_gu(network, train_loader, device, args):
    data_iter = iter(train_loader)
    x, target = next(data_iter)
    x, target = x.to(device), target.to(device)

    noise = x.new(x.size()).normal_(0, args.sigma).to(device)
    x2 = x + noise

    o, _ = network(x)
    o_, _ = network(x2)
    o = o.detach().cpu().numpy()
    o_ = o_.detach().cpu().numpy()
    return -np.sum(np.square(o-o_))

def scores(network, train_loader, device, stds, means, args):
    scoreNAS = score_nas(network, train_loader, device, args)
    scoreGU  = score_gu(network, train_loader, device, args)
    std_of_nas = stds["nas"]
    mean_of_nas = means["nas"]
    stand_score_nas = (scoreNAS - mean_of_nas) / std_of_nas
    std_of_gu = stds["gu"]
    mean_of_gu = means["gu"]
    stand_score_gu = (scoreGU - mean_of_gu) / std_of_gu
    return stand_score_nas*2+stand_score_gu
