import numpy as np

class KernelGene(object):
    'Basic unit of genetic code as filter'

    def __init__(self, kernel, bias):

        if (type(kernel).__module__ != np.__name__):
            raise ValueError('Kernel Gene must be a numpy array')
        #elif (type(stride).__name__ != 'int' or stride <= 0):
        #    raise ValueError('Stride must be an integer bigger than 0')
        self.kernel = kernel
        self.bias = bias

    def __str__(self):
        return 'filter size = '+str(self.kernel.shape)+ ', bias = '+str(self.bias)

    def mutate(self,p):
        # TODO: set appropriate std to this gaussian
        if p >= np.random.rand():
            self.kernel = np.random.normal(size = self.kernel.shape)
            return True


class PoolingChromosome(object):
    'Basic unit of genetic code as pooling'
    # TODO: Check input parameters
    def __init__(self, pooling_size, stride):
        self.id_layer = 'pooling'
        self.size = pooling_size
        self.stride = stride

    def __str__(self):
        return self.id_layer +': size = ' + str(self.size) + ' stride = ' + str(self.stride)

    def mutate(self,p):
        return [] # dummy function


class KernelChromosome(object):
    'Many genes(filter) forming a chromosome(layer)'

    #TODO: Implement geneCrossover, set and get
    def __init__(self, kernels = [], biases = []):
        '''
        kernels: list of kernels n x m numpy arrays
        biases: list of biases value of each kernel
        '''
        if len(kernels) != len(biases):
            raise ValueError('kernels and biases must have same length')

        self.id_layer = 'convolution'
        self.n_kernels = len(kernels)
        genes = list()
        for i in range(self.n_kernels):
            genes.append(KernelGene(kernels[i], biases[i]))
        self.genes = genes
        if len(kernels) == 0:
            self.kernels_shape = (3,3,3)
        else:
            self.kernels_shape = self.genes[0].kernel.shape

    def __str__(self):
        str_structure = self.id_layer + ':\n'
        for i in range(self.n_kernels):
            str_structure += 'kernel '+str(i) +': '+str(self.genes[i])+'\n'
        return str_structure

    def setGenes(self, genes):
        self.n_kernels = len(genes)
        self.genes = genes
        self.kernels_shape = self.genes[0].kernel.shape

    def geneCrossover(self, chromosome):
        #Get crossover point
        cross_point1 = np.random.randint(len(self.genes))
        cross_point2 = np.random.randint(len(chromosome.genes))
        child1_genes = self.genes[:cross_point1] + chromosome.genes[cross_point1:]
        child2_genes = chromosome.genes[:cross_point1] + self.genes[cross_point1:]
        child1 = KernelChromosome()
        child2 = KernelChromosome()
        child1.setGenes(child1_genes)
        child2.setGenes(child2_genes)
        return child1, child2

    def mutate(self, p):
        for i in range(len(self.genes)):
            self.genes[i].mutate(p)


class Genome(object):
    'Many chromosomes(layers) forming a Genome(entire network)'
    #TODO: Implement constructor to given parameters ? and chromCrossover
    def __init__(self, parameters = []):
        if len(parameters) == 0:
            #self.chromosome_type = ['convolution', 'pooling']
            self.n_chromosomes = 0
            self.chromosomes = list()
        else:
            self.n_chromosomes= len(parameters)
            self.chromosomes = list()
            for i in range(self.n_chromosomes):
                filters = list()
                biases = list()
                for j in range(parameters[i][0].shape[3]):
                    filters.append(parameters[i][0][:,:,:,j])
                    biases.append(parameters[i][1][j])
                self.chromosomes.append(KernelChromosome(filters,biases))

    def __str__(self):
        str_structure = '------Genome------\n'
        for i in range(self.n_chromosomes):
            str_structure += '---Chromosome '+str(i)+': \n'+str(self.chromosomes[i])
        return str_structure

    def mutate(self, p):
        for i in range(len(self.chromosomes)):
            self.chromosomes[i].mutate(p)

    def crossover(self, genome):
        child1 = Genome()
        child2 = Genome()
        for i in range (self.n_chromosomes):
            chromo1, chromo2 = self.chromosomes[i].geneCrossover(genome.chromosomes[i])
            child1.add_chromosome(chromo1)
            child2.add_chromosome(chromo2)
        return child1, child2

    def set_parameters(self, parameters):
        self.n_chromosomes= len(parameters)
        self.chromosomes = list()
        for i in range(len(n_chromosomes)):
            filters = list()
            for j in range(parameters[i].shape[3]):
                filters.append(parameters[i][:,:,:,j])
            biases = parameters[i][1]
            self.chromosomes.append(KernelChromosome(filters,biases))

    def get_parameters(self):
        parameters = list()
        for i in range(self.n_chromosomes):
            parameter_chromosome = list()
            dim = self.chromosomes[i].kernels_shape + (self.chromosomes[i].n_kernels,)
            filters = np.zeros(shape = dim)
            biases = np.zeros(shape = (self.chromosomes[i].n_kernels,))
            for j in range(self.chromosomes[i].n_kernels):
                filters[:,:,:,j] = self.chromosomes[i].genes[j].kernel
                biases[j] = self.chromosomes[i].genes[j].bias
            parameters.append([filters, biases])
        return parameters

    def add_chromosome(self, chromosome):
        self.chromosomes.append(chromosome)
        self.n_chromosomes += 1












