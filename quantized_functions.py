"""
TODO:
    build quantized replacement functions
"""

"""
LOOKUP:
    - pytorch floor function
    - pytorch casting precision
"""

def integer_second_order_polynomial(tensor, scale, constants):
"""
Input:
    - tensor: a torch.Tensor consisting of quantized values
    - scale: a scaling factor
    - constants: a tuple consisting of the scalars a, b, c 

Returns:
    - polynomial_approximation: elementwise integer approximation of the second order polynomial function, same dtype as passed tensor
            a(x+b)**2 + c
    - polynomial_scale: scaling factor used for future computations
"""

    a, b, c = constants
    
    q_b = torch.floor_divide(tensor, scale)
    q_c = torch.floor_divide(c, a*(scale**2))
    
    polynomial_approximation = (q + q_b)**2 + q_c
    polynomial_scale = torch.floor(a*(scale**2))
    
    return polynomial_approximation.type(tensor.dtype), polynomial_scale


    
def quantized_error_function(tensor, scale):
    
    dtype = tensor.dtype
    
    #store sign of tensor
    q_sgn = torch.zeros(tensor.shape)
    q_sgn = torch.where(tensor>=0, torch.tensor(1).type(dtype), q_sgn)
    q_sgn = torch.where(q_sgn==0, torch.tensor(-1).type(dtpye), q_sgn)
    
    #clip q so that scale of q is in interval [0, 1.769]
    
    constants = (-0.2888, -1.769, 1)
    q_L, scale_L = integer_second_order_polynomial(tensor, scale, constants)
    
    #restore sign
    q_out = q_sgn*q_L
    
    return q_out, scale_L

def quantized_gelu(tensor, scale):
    q_erf, scale_erf = quantized_error_function(tensor, scale/ (1.41421356237))
    
    q_one = torch.floor_divide(1 / scale_erf)
    
    q_out = q*(q_erf + q_one)
    scale_out = scale*scale_ef/2
    
    return q_out, scale_out
    
def quantized_softmax(tensor, )

def integer_exponential(q, scale):
    
    #fix
    log_two_q = floor( log(2/scale))
    
    #fix
    z = floor(-q/log_two_q)
    
    q_poly = q + z*log_two_q
    
    #fix
    q_L, s_L = integer_polynomial(q_poly, S)

    
def scalar_quantized_square_root(n, precision):
    
    if n==0:
        return 0
    
    x = 2**torch.ciel(precision/2)
    
    while True:
        numerator = x + torch.floor_divide(n, x)
        testvalue = torch.floor_divide(numerator, 2)
        
        if testvalue >= x:
            return x
        
        else:
            x = testvalue