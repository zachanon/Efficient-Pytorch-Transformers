"""
Implementation of quantized functions is based off the paper found here:

I-BERT: Integer-only BERT Quantization
https://arxiv.org/abs/2101.01321

"""


def integer_second_order_polynomial(tensor, scale, constants):
"""
Calculates a quantized second order polynomial of the form a(x+b)**2 + c

Input:
    - tensor: a torch.Tensor consisting of quantized values
    - scale: a scaling factor
    - constants: a tuple consisting of the scalars a, b, c 

Returns:
    - polynomial_approximation: elementwise integer approximation of the second order polynomial function, same dtype as passed tensor
            a(x+b)**2 + c
    - polynomial_scale: scaling factor used for future computations
"""
    #see algorithm 1 for more details
    
    a, b, c = constants
    
    q_b = torch.floor_divide(tensor, scale)
    q_c = torch.floor_divide(c, a*(scale**2))
    
    polynomial_approximation = (q + q_b)**2 + q_c
    polynomial_approximation = polynomial_approximation.type(tensor.dtype)
    
    polynomial_scale = torch.floor(a*(scale**2))
    
    return polynomial_approximation, polynomial_scale


    
def quantized_error_function(tensor, scale):
"""
Calculates a quantized version of the error function 

    erf(x) := 2/sqrt(pi) * (integrate from 0 to x)(exp(-(x**2)) dx) 
    
Input:
    - tensor: a torch.Tensor consisting of quantized values
    - scale: a scaling factor
    
Returns:
    - q_out: a torch.Tensor consisting of the elementwise applied quantized error function
    - scale_out: a scaling factor used for future computations
"""
#see algorithm 2 for more details

    if(scale > 1.769):
        scale = 1.769
        print("WARNING: Scale has been clipped to 1.769 for quantized erf computation")
    
    dtype = tensor.dtype
    
    #store sign of tensor
    q_sgn = torch.zeros(tensor.shape)
    q_sgn = torch.where(tensor>=0, torch.tensor(1).type(dtype), q_sgn)
    q_sgn = torch.where(q_sgn==0, torch.tensor(-1).type(dtpye), q_sgn)
    
    
    constants = (-0.2888, -1.769, 1)
    q_L, scale_out = integer_second_order_polynomial(torch.abs(tensor), scale, constants)
    
    #restore sign
    q_out = q_sgn*q_L
    
    return q_out, scale_out



def integer_gelu(tensor, scale):
"""
Implements the quantized gelu function elementwise for integer only computation
    
    gelu(x) := x/2 * (1 + error_function(x/sqrt(2)))
    
Input:
    - tensor: a torch.Tensor consisting of quantized values
    - scale: a scaling factor
    
Returns:
    - q_out: a torch.Tensor consisting of the elementwise applied integer gelu function
    - scale_out: a scaling factor used for future computations
"""
#see algorithm 2 for more details


    q_erf, scale_erf = quantized_error_function(tensor, scale/ (1.41421356237))
    
    q_one = torch.floor_divide(1,scale_erf)
    
    q_out = q*(q_erf + q_one)
    scale_out = scale*scale_erf/2
    
    return q_out, scale_out
    

def integer_exponential(tensor, scale):
"""
Implements the quantized exp function elementwise for integer only computation
    
Input:
    - tensor: a torch.Tensor consisting of quantized values
    - scale: a scaling factor
    
Returns:
    - q_out: a torch.Tensor consisting of the elementwise applied integer exp function
    - scale_out: a scaling factor used for future computations
"""
#see algorithm 3 for more details

    k = 5 #let k be a large enough integer? Not clear from the paper
    
    q_ln2 = torch.floor_divide(0.69314718056,scale)
    z = torch.floor_divide(-tensor, q_ln2)
    q_p = tensor + z*q_ln2
    
    constants = (0.3585, 1.353, 0.344)
    q_L, scale_L = integer_second_order_polynomial(q_p, scale, constants)
    
    q_out = q_L << (k-z)
    scale_out = 2**(-k)*scale_L
    
    return q_out, scale_out

def integer_softmax(tensor, scale):
"""
Implements the quantized softmax function elementwise for integer only computation
    
Input:
    - tensor: a torch.Tensor consisting of quantized values
    - scale: a scaling factor
    
Returns:
    - q_out: a torch.Tensor consisting of the elementwise applied integer softmax function
    - scale_out: a scaling factor used for future computations
"""
#see algorithm 3

    k = 5 #let k be a large enough integer?
    
    q_exp, scale_exp = integer_exponential(tensor, scale)
    
    q_sum = torch.sum(q_exp, dim=-1)
    
    q_out = (q_exp << k) / q_sum
    scale_out = 2**(-k)*scale_exp
    
    return q_out, scale_out
    
def integer_square_root(n, precision=8):
"""
Implements the quantized square root function for integer only computation.

NOTE: NOT Vectorized. Only takes integer input.
    
Input:
    - n: an integer
    - precison: a scaling factor equal to the bits of precison.
    
Returns:
    - x: the integer approximated square root of n
"""
#see algorithm 4
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