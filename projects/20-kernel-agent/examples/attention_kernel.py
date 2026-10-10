"""Hand-written level-8 correctness fixture; not a model-generated solve."""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_attention_(q, k, v):
    seq, dim = q.shape
    q_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    k_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    v_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sb, src=q)
    nisa.dma_copy(dst=k_sb, src=k)
    nisa.dma_copy(dst=v_sb, src=v)

    qT_ps = nl.ndarray((dim, seq), dtype=nl.float32, buffer=nl.psum)
    kT_ps = nl.ndarray((dim, seq), dtype=nl.float32, buffer=nl.psum)
    qT_sb = nl.ndarray((dim, seq), dtype=nl.float32, buffer=nl.sbuf)
    kT_sb = nl.ndarray((dim, seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_transpose(dst=qT_ps, data=q_sb)
    nisa.nc_transpose(dst=kT_ps, data=k_sb)
    nisa.tensor_copy(dst=qT_sb, src=qT_ps)
    nisa.tensor_copy(dst=kT_sb, src=kT_ps)

    scores_ps = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.psum)
    scores_sb = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_matmul(dst=scores_ps, stationary=qT_sb, moving=kT_sb)
    nisa.tensor_scalar(dst=scores_sb, data=scores_ps,
                       op0=nl.multiply, operand0=1.0 / (dim ** 0.5))

    row_max = nl.max(scores_sb, axis=[1], keepdims=True)
    shifted = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    exp_scores = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=shifted, data=scores_sb, op0=nl.subtract, operand0=row_max)
    nisa.activation(dst=exp_scores, data=shifted, op=nl.exp)
    row_sum = nl.sum(exp_scores, axis=[1], keepdims=True)
    reciprocal = nl.ndarray((seq, 1), dtype=nl.float32, buffer=nl.sbuf)
    probabilities = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=reciprocal, data=row_sum)
    nisa.tensor_scalar(dst=probabilities, data=exp_scores,
                       op0=nl.multiply, operand0=reciprocal)

    pT_ps = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.psum)
    pT_sb = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_transpose(dst=pT_ps, data=probabilities)
    nisa.tensor_copy(dst=pT_sb, src=pT_ps)
    out_ps = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.psum)
    out_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    out = nl.ndarray((seq, dim), dtype=q.dtype, buffer=nl.shared_hbm)
    nisa.nc_matmul(dst=out_ps, stationary=pT_sb, moving=v_sb)
    nisa.tensor_copy(dst=out_sb, src=out_ps)
    nisa.dma_copy(dst=out, src=out_sb)
    return out
