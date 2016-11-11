import numpy as np


class KernelGene(object):
    'Basic unit of genetic code as filter'

    def __init__(self, kernel, stride):

        if (type(kernel).__module__ != np.__name__):
            raise ValueError('Kernel Gene must be a numpy array')
        elif (type(stride).__name__ != 'int' or stride <= 0):
            raise ValueError('Stride must be an integer bigger than 0')
        self.kernel = kernel
        self.stride = stride

    def __str__(self):
        return 'filter size = '+str(self.kernel.shape)+ ', stride = '+str(self.stride)

    def mutate(self,p):
        # TODO: set appropriate std to this gaussian
        if p < np.random.rand():
            self.kernel += np.random.normal(size = self.kernel.shape)


class PoolingChromosome(object):
    'Basic unit of genetic code as pooling'
    # TODO: Check input parameters
    def __init__(self, pooling_size, stride):
        self.id_layer = 'pooling'
        self.size = pooling_size
        self.stride = stride

    def __str__(self):
        return self.id_layer +': size = ' + str(self.size) + ' stride = ' + str(self.stride)


class KernelChromosome(object):
    'Many genes(filter) forming a chromosome(layer)'
    #TODO: Implement geneCrossover, set and get
    def __init__(self, kernels, strides):
        '''
        kernels: list of kernels n x m numpy arrays
        strides: list of stride value of each kernel
        '''
        if len(kernels) != len(strides):
            raise ValueError('kernels and strides must have same length')

        self.id_layer = 'convolution'
        self.n_kernels = len(kernels)
        genes = list()
        for i in range(self.n_kernels):
            genes.append(KernelGene(kernels[i], strides[i]))
        self.genes = genes

    def __str__(self):
        str_structure = self.id_layer + ':\n'
        for i in range(self.n_kernels):
            str_structure += 'kernel '+str(i) +': '+str(self.genes[i])+'\n'
        return str_structure

    def setGenes(self, genes):
        self.n_kernels = len(genes)
        self.genes = genes

    def geneCrossover(self, chromosome):
        #Get crossover point
        cross_point1 = np.random.randint(len(self.genes))
        cross_point2 = np.random.randint(len(chromosome.genes))
        child1_genes = self.genes[:cross_point1] + chromosome.genes[cross_point2:]
        child2_genes = self.genes[:cross_point2] + chromosome.genes[cross_point1:]
        child1 = KernelChromosome([],[])
        child2 = KernelChromosome([],[])
        child1.setGenes(child1_genes)
        child2.setGenes(child2_genes)
        return child1, child2


class Genome(object):
    'Many chromosomes(layers) forming a Genome(entire network)'
    #TODO: Implement constructor to given parameters ? and chromCrossover
    def __init__(self, chromosome_type = [], parameters = []):
        if len(chromosome_type) == 0:
            self.chromosome_type = ['convolution', 'pooling']
            self.n_chromosomes = len(self.chromosome_type)
            filters = [np.random.normal(size = (3,3)) for i in range(4)]
            strides = [1,1,1,1]
            self.chromosomes = list()
            self.chromosomes.append(KernelChromosome(filters,strides))
            self.chromosomes.append(PoolingChromosome((2,2),1))

    def __str__(self):
        str_structure = '------Genome------\n'
        for i in range(self.n_chromosomes):
            str_structure += '---Chromosome '+str(i)+': \n'+str(self.chromosomes[i])
        return str_structure












