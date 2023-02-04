from score import net_score
import copy

class chromosome():
    def __init__(self, gene = "", fitness = 0):
        self.gene = gene
        self.fitness = fitness
        self.acc = 0
        self.uid = 0

class GA():
    def __init__(self, MAXN_POPULATION, MAXN_ITERATION, MAXN_CONNECTION, MAXN_OPERATION, searchspace, device, stds, means, acc_type, args):
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
        self.acc_type = acc_type
        self.best_chrom = chromosome()
        self.NAS_201_ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

    def init_population(self):
        for i in range(self.MAXN_POPULATION):
            chrom = chromosome()
            for j in range(self.MAXN_CONNECTION):
                op = random.randint(0, self.MAXN_OPERATION-1)
                chrom.gene += (bin(1<<op)[2:]).zfill(self.MAXN_OPERATION)
            self.population.append(chrom)

    def evaluate(self):
        for i in range(len(self.population)):
            network, uid, acc = gene2net(self.population[i].gene, self.NAS_201_ops, self.searchspace, self.acc_type, self.args)
            self.population[i].acc = acc
            self.population[i].uid = uid
            if self.population[i].gene not in self.DICT:
                self.population[i].fitness = self.DICT[self.population[i].gene] = net_score(network, self.train_loader, self.device, self.stds, self.means, self.args)
            else:
                self.population[i].fitness = self.DICT[self.population[i].gene]
            if self.population[i].fitness > self.best_chrom.fitness or self.best_chrom.gene == "":
                self.best_chrom.fitness = self.popultation[i].fitness
                self.best_chrom.gene = copy.deepcopy(self.population[i].gene)

    def mutation(self, chrom):
        p = random.randint(0, self.MAXN_CONNECTION-1)
        newop = random.randint(0, self.MAXN_OPEARION-1)
        gene_sect_len = self.MAXN_OPERATION
        gene = chrom.gene
        genelist = [gene[i:i+gene_sect_len] for i in range(0, len(gene), gene_sect_len)]
        genelist[p] = (bin(1<<newop)[2:]).zfill(self.MAXN_OPERATION)
        chrom.gene = "".join(genelist)
        return chrom
    
    def select_2chrom_fromN(self, N=10):
        cand = np.sample(self.population, N)
        maxi = smaxi = cand[0]
        for chrom in cand:
            if maxi.fitness < chrom.fitness:
                smaxi = maxi
                maxi = chrom
            elif smaxi.fitness < chrom.fitness:
                smaxi = chrom
        return maxi, smaxi
    
    def run(self):
        self.init_population()
        self.evaluate()
        for _ in range(self.MAXN_ITERATION):
            for i in range(self.MAXN_POPULATION):
                p = select_2chrom_fromN(self)
                
                offspring = crossover(p[0], p[1])
                if random.randint(1,100) < self.PROB_MUTATION*100:
                    offspring = self.mutation(offspring)
                
                self.population.append(offspring)

            self.evaluation()
            self.population.sort(key = lambda chrom: chrom.fitness, reverse = True)
            self.population = self.population[:self.MAXN_POPULATION]
        network, uid, acc = gene2net(self.best_chrom.gene, self.ops, self.searchspace, self.acc_type, self.args)

def gene2sect(gene, ops):
    gene_sect_len = len(ops)
    return [ops[gene[i:i+gene_sect_len].find("1")] for i in range(0, len(gene), gene_sect_len)]

def gene2net(gene, ops, searchspace, acc_type, args):
    gene_sect = gene2sect(gene, ops)
    arch = "|{}|+|{}|{}|+|{}|{}|{}|".format(*gene_sect)
    idx = searchspace.get_index_by_arch(arch)
    uid = searchspace[idx]
    network = searchspace.get_network(uid)
    acc = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
    return network, uid, acc

