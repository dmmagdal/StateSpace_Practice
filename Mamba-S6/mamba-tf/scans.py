#
# References: 
#   tensorflow probability - tfp.math.log_cumsum_exp (deprecated)
#       https://www.tensorflow.org/probability/api_docs/python/tfp/math/log_cumsum_exp
#   tensorflow - tf.math.cumulative_logsumexp (current)
#       https://www.tensorflow.org/api_docs/python/tf/math/cumulative_logsumexp
#   torch - torch.logcumsumexp
#       https://pytorch.org/docs/stable/generated/torch.logcumsumexp.html
#   tensorflow - tf.dtypes.complex 
#       https://www.tensorflow.org/api_docs/python/tf/dtypes/complex
#   torch - torch.complex
#       https://pytorch.org/docs/stable/generated/torch.complex.html
import tensorflow as tf


def complex_log(input, eps=1e-12):
    eps = input.new_tensor(eps)
    real = input.abs().maximum(eps).log()
    imag = (input < 0).to(input.dtype) * tf.math.pi
    return tf.dtypes.complex(real, imag)


def selective_scan(u, dt, A, B, C, D, mode='cumsum'):
    # match mode:
    #     case 'cumsum':
    if mode == "cumsum":
        dA = tf.einsum('bld,dn->bldn', dt, A)
        dB_u = tf.einsum('bld,bld,bln->bldn', dt, u, B)
        
        dA_cumsum = tf.pad(dA[:, 1:], (0, 0, 0, 0, 0, 1)).flip(1).cumsum(1).exp().flip(1)
        x = dB_u * dA_cumsum
        x = tf.math.cumsum(x, axis=1) / (dA_cumsum + 1e-12)
        y = tf.einsum('bldn,bln->bld', x, C)
    
        return y + u * D
        
        # case 'logcumsumexp':
    if mode == "logcumsumexp":
        dA = tf.einsum('bld,dn->bldn', dt, A)
        dB_u = tf.einsum('bld,bld,bln->bldn', dt, u, B)
        dB_u_log = complex_log(dB_u)
        
        dA_star = tf.cast(
            tf.pad(tf.math.cumsum(dA[:, 1:], axis=1), (0, 0, 0, 0, 1, 0)), 
            tf.complex64
        )
        x_log = tf.math.cumulative_logsumexp(dB_u_log - dA_star, 1) + dA_star
        
        y = tf.einsum('bldn,bln->bld', x_log.real.exp() * tf.math.cos(x_log.imag), C)
        return y + u * D
