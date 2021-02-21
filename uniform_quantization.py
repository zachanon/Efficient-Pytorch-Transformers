import torch

PRECISION_TYPES = {
    'int8': 8,
    'int16': 16,
    'int32': 32,
}



def uniform_quantization(parameters, clip_value, bit_precision='int8'):
    """
    Input:
        - parameters: torch.Tensor parameters to quantize
        - clip_value: truncates parameters to [-clip_value, clip_value]
        - bit_precision: the final quantized bit precision

    Returns:
        - quantized_parameters: a torch.Tensor that has been uniformly quantized to the given precision
        - scale: scaling factor used for further computation

    """

    assert bit_precision in PRECISION_TYPES, "%s is not a valid bit precision" % bit_precision
    
    precision = PRECISION_TYPES[bit_precision]
    
    scale = clip_value / (2**(precision-1)-1)
    
    parameters = parameters.clamp(-clip_value, clip_value)
    parameters /= scale
    parameters = _cast_precision(parameters, bit_precision)
    
    return parameters, scale


def _cast_precision(parameters, bit_precision):
    """
    Helper method to recast parameter tensor
    """
    
    if bit_precision == 'int8':
        return parameters.type(torch.int8)
    
    if bit_precision == 'int16':
        return parameters.type(torch.int16)
    
    if bit_precision == 'int32':
        return parameters.type(torch.int32)

