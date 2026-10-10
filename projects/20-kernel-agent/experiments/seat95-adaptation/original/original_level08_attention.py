import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_attention_(q, k, v):
    seq, dim = q.shape
    # Allocate tiles
    q_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    k_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    v_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    t_ps = nl.ndarray((dim, seq), dtype=nl.float32, buffer=nl.psum)
    t_ps_sb = nl.ndarray((dim, seq), dtype=nl.float32, buffer=nl.sbuf)
    m_sb = nl.ndarray((seq, 1), dtype=nl.float32, buffer=nl.sbuf)
    e_sb = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    r_sb = nl.ndarray((seq, 1), dtype=nl.float32, buffer=nl.sbuf)
    out_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    out_hbm = nl.ndarray((seq, dim), dtype=q.dtype, buffer=nl.shared_hbm)

    # Copy inputs to sbuf
    nisa.dma_copy(dst=q_sb, src=q)
    nisa.dma_copy(dst=k_sb, src=k)
    nisa.dma_copy(dst=v_sb, src=v)

    # Compute Q @ K^T
    nisa.nc_transpose(dst=t_ps, data=k_sb)
    nisa.tensor_copy(dst=t_ps_sb, src=t_ps)
    nisa.nc_matmul(dst=t_ps, stationary=q_sb, moving=t_ps_sb)

    # Subtract row max
    m_sb = nl.max(t_ps, axis=[1], keepdims=True)
    nisa.tensor_scalar(dst=t_ps, data=t_ps, op0=nl.subtract, operand0=m_sb)

    # Compute exp
    nisa.activation(dst=e_sb, data=t_ps, op=nl.exp)

    # Compute softmax
    r_sb = nl.sum(e_sb, axis=[1], keepdims=True)
    nisa.tensor_scalar(dst=r_sb, data=r_sb, op0=nl.reciprocal, operand0=0.0)
    nisa.tensor_scalar(dst=e_sb, data=e_sb, op0=nl.multiply, operand0=r_sb)

    # Compute E @ V
    nisa.nc_transpose(dst=t_ps, data=v_sb)
    nisa.tensor_copy(dst=t_ps_sb, src=t_ps)
    nisa.nc_matmul(dst=t_ps, stationary=e_sb, moving=t_ps_sb)
    nisa.tensor_copy(dst=out_sb, src=t_ps)
    nisa.dma_copy(dst=out_hbm, src=out_sb)
    return out_hbm
