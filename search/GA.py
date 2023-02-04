from score import net_score

class GA():
    def __init__(self, MAXN_POPULATION, MAXN_ITERATION, MAXN_CONNECTION, MAXN_OPERATION, searchspace, device, stds, means, args):
        self.MAXN_POPULATION = MAXN_POPULATION
        self.MAXN_ITERATION = MAXN_ITERATOPN
        self.MAXN_CONNECTION = MAXN_CONNECTION
        self.MAXN_OPERATION = MAXN_OPERATION
        self.DICT = {}
        self.population = []
        self.searchspace = searchspace
        self.args = args
        self.device = device
        self.stds = stds
        self.means = means
        self.NAS_201_ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

    def init_population(self,):
        for i in range(self.MAX_POPULATION):
            gene = ""
            for j in range(self.MAX_CONNECTION):
                op = random.randint(0, self.MAX_OPERATION-1)
                gene += f"{(bin(1<<op)[2:]):>05s}"
            self.population.append(gene)

    def evaluate(self):
        for i in range(self.MAXN_POPULATION):
            network = gene2net(self.population[i], self.NAS_201_ops, self.searchspace)
            if self.population[i] not in self.DICT:
                self.DICT[self.population[i]] = net_score(network, self.train_loader, self.device, self.stds, self.means, self.args)
            
                
    def run():
        pass

def gene2net(gene, ops, searchspace):
    gene_sect_len = len(ops)
    gene_sect = [ops[gene[i:i+gene_sect_len].find("1")] for i in range(0, len(gene), gene_sect_len)]
    arch = "|{}|+|{}|{}|+|{}|{}|{}|".format(*gene_sect)
    idx = searchspace.get_index_by_arch(arch)
    uid = searchspace[idx]
    network = searchspace.get_network(uid)
    return network

