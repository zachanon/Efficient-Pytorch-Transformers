'''
Input: Network N, hparams

output: Network N^q, the now quantized network

Hparams: Temperature, epochs, quantization set

params: alpha, beta, b

'''


def quantize_network(network, quantization_set='int8', epochs=1):
    """
    Inputs:
        - network: A pytorch network
        - quantization_set: the target quantized integer set
        - epochs: number of training epochs for the quantization process
    
    Returns:
        - quantized_network: the quantized pytorch network for 
            inference
    """
    
    training_network = copy.deepcopy(network)
    training_network = initialize_quantized_training_network(training_network,
                                                             quantization_set)
    
    trained_network = train_soft_quantized_network(training_network, epochs)
    
    quantized_network = quantize_trained_network(trained_network)

    
    return quantized_network

def initialize_quantized_training_network(training_network, quantization_set):
    
    for module in training_network.modules():
        
        #infer {s_i[m], o[m]}
        
        infer_scale_and_offset(module, quantization_set)
        
        #initialize {alpha[m], beta[m], b_i[m]}
        initialize_trainable_parameters(module,)
        
        #soft quantization function {Q_X[m], Q_Theta[m]}
            # based on {Y_X[m],Y_Theta[m], X[m], Theta[m]}
            
        #apply soft quantization to each x[m,d] in X[m]
            # and each theta[m,d] in Theta[m]
            
        #y[m,d] = Q_X[m]{alpha[m], beta[m], b_i[m]} (x[m,d])
        #theta_hat[m.d] = Q_Theta[m]{alpha[m], beta[m], b_i[m]} (theta[m,d])
        
        #forward propagate module with quantized weights
        
    return training_network

def train_soft_quantized_network(training_network, epochs):
    
    for epoch in range(epochs):

        #train quantized_network to optimize
            #Theta{alpha[m], beta[m], b[m]}, 
            #X{alpha[m], beta[m], b[m]}
        #gradually increase temperature
        
    return trained_network

def quantize_trained_network(trained_network):
     
    for module in trained_network.modules():

        #replace soft quantization with inference quantization function
        
    return quantized_network

def infer_scale_and_offset(module, quantization_set):
    """
    Infers scaling and offset for the given set. Assigns as buffers to module.
    
    Inputs:
        - module: the nn.Module to assign scale to
        - quantization_set: the set of integers to quantize into
    """
    
    assert quantization_set in QUANTIZATION_SETS, \
        '%s is not an implemented quantization set' % quantization_set
    
    values = QUANTIZATIZATION_SETS[quantization_set]
    scale = torch.zeros(len(values))
    offset = 0
    
    for i in range(values):
        scale[i] = values[i+1] - values[i]
        offset += scale[i]
    
    module.register_buffer('scale', scale)
    module.register_buffer('offset', offset/2)

def initialize_trainable_parameters(module, quantization_set):
    '''
    Initializes the trainable parameters alpha, beta and b for the passed module
    
    Input:
        - module: A nn.Module to assign scaling parameters
    '''
    
    #alpha = 1/beta
    #beta = 5*(max abs of qset) / 4*(max abs of activations X and params Theta)
    
    
QUANTIZATION_SETS = {
    "int8": sum([[(127-val) for val in range(127)],
           [(-val) for val in range(1,129)]],
           []),
    "binary": [0, 1],
    "ternary": [-1, 0, 1],
}
