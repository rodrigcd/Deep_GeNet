import numpy as np


class KernelGene:
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

    def mutate(self):
        self.kernel += np.random.normal(size = self.kernel.shape)


class PoolingChromosome:
    'Basic unit of genetic code as pooling'
    #TODO: Check input parameters
    def __init__(self, pooling_size, stride):
        self.id_layer = 'pooling'
        self.size = pooling_size
        self.stride = stride

    def __str__(self):
        return self.id_layer +': size = ' + str(self.size) + ' stride = ' + str(self.stride)


class KernelChromosome:
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

class Genome:
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












