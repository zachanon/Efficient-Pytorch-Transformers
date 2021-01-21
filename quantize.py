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
    training_network = initialize_quantized_training_network(training_network, quantization_set)
    
    trained_network = train_soft_quantized_network(training_network, epochs)
    
    quantized_network = quantize_trained_network(trained_network)

    
    return quantized_network

def initialize_quantized_training_network(training_network, quantization_set):
    
    for module in training_network.modules():
        
        #infer {s_i[m], o[m]}
        
        scale = infer_scale(quantization_set)
        offset = infer_offset(scale)
        
        #initialize {alpha[m], beta[m], b_i[m]}
        init
        
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

def infer_scale(quantization_set):
    """
    Inputs:
        - quantization_set: the set of integers to quantize into
        
    Returns:
        - scale: scale of the quantized step function
    """
    
    return scale

def infer_offset(scale):
    """
    Inputs:
        - scale: a vector of scaling factors for each quantized step function
        
    Returns:
        - offset: global offset to keep quantized output zero-centered
    """
    
    return offset