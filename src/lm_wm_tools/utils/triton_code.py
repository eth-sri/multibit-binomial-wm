import torch
import triton
import triton.language as tl

UNIFORM_RANDOM_BITS = 23
_BERNOULLI_CHUNK_OFFSET_STRIDE = 0x9E3779B9

# CUDA backend

@triton.jit
def _stateless_uniform_kernel(
    offsets_ptr,          # *int32/int64* offsets
    out_ptr,              # float32
    n_elements,           # int32/int64
    seed,                 # int32/64 scalar
    BLOCK: tl.constexpr,  # block size
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    idx = block_start + tl.arange(0, BLOCK)
    mask = idx < n_elements

    # Load offsets for this block
    offs = tl.load(offsets_ptr + idx, mask=mask)
    # tl.rand expects int32 offsets
    offs = offs.to(tl.int32)

    # Stateless RNG: depends only on (seed, offs)
    rnd = tl.rand(seed, offs)  # U(0,1) float32

    tl.store(out_ptr + idx, rnd, mask=mask)


def _stateless_uniform_cuda(offsets: torch.Tensor, seed: int, block_size: int = 1024) -> torch.Tensor:
    """
    Stateless Philox-based RNG using Triton's tl.rand. Pytorch has no stateless RNG.

    Args
    ----
    offsets : torch.Tensor
        CUDA tensor of integer offsets (any shape, any int dtype).
    seed : int
        Global seed (Python int or scalar convertible to int32/64).
    block_size : int
        Triton block size (tuning knob).

    Returns
    -------
    torch.Tensor
        Float32 tensor of same shape as `offsets`, values in [0, 1).
    """
    if not offsets.is_cuda:
        raise ValueError("offsets must be a CUDA tensor")

    if not offsets.is_contiguous():
        offsets = offsets.contiguous()

    orig_shape = offsets.shape
    flat_offsets = offsets.view(-1)

    if flat_offsets.dtype not in (torch.int32, torch.int64):
        flat_offsets = flat_offsets.to(torch.int64)

    n_elements = flat_offsets.numel()

    out_flat = torch.empty(n_elements, device=offsets.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)

    _stateless_uniform_kernel[grid](
        flat_offsets,
        out_flat,
        n_elements,
        seed,
        BLOCK=block_size,
    )

    return out_flat.view(orig_shape)


# CPU backend
@torch.jit.script
def _stateless_uniform_cpu(offsets: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Generates a tensor of stateless pseudo-random uniform variables in [0, 1).
    
    Args:
        offsets: A tensor of integer offsets (of any shape).
        seed: An integer seed.
    
    Returns:
        A tensor of the same shape as `offsets` with values in [0, 1).
    """

    u = offsets.to(torch.int32) + seed
    
    u ^= (u >> 16)
    u *= -2048144789
    u ^= (u >> 13)
    u *= -1028477387
    u ^= (u >> 16)
    
    return (u & 0x7FFFFF).float() * (1.0 / 8388608.0)

def stateless_uniform(offsets: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Stateless RNG similar to Triton's tl.rand(seed, offsets).

    Args
    ----
    offsets : torch.Tensor
        Integer tensor (any shape) whose values act as counters.
        Device: CPU or CUDA. Else will be cast to CPU.
    seed : int
        Scalar seed. Changing this changes the entire random field.

    Returns
    -------
    torch.Tensor
        float32 tensor in [0,1), same shape & device as `offsets`.

    """
    if offsets.device.type == "cuda":
        return _stateless_uniform_cuda(offsets, seed)
    else:

        if offsets.device.type != "cpu":
            device_type = offsets.device.type
            print(f"Warning: Using CPU backend for device type {device_type}")
            offsets = offsets.to("cpu")

            uniform = _stateless_uniform_cpu(offsets, seed).to(device_type)
            return uniform

        return _stateless_uniform_cpu(offsets, seed)


def stateless_bernoulli_bits(
    offsets: torch.Tensor,
    seed: int,
    k: int,
    bits_per_uniform: int = UNIFORM_RANDOM_BITS,
    chunk_offset_stride: int = _BERNOULLI_CHUNK_OFFSET_STRIDE,
) -> torch.Tensor:
    """
    Sample k pseudo-random Bernoulli(0.5) variables per offset by reusing bits from
    stateless uniforms.

    For each offset, one stateless uniform provides ``bits_per_uniform`` Bernoulli draws
    (from its quantized mantissa bits). When ``k`` exceeds this budget, extra uniforms
    are drawn using deterministically shifted offsets.

    Returns
    -------
    torch.Tensor
        int32 tensor with shape ``(*offsets.shape, k)`` containing 0/1 samples.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}.")
    if bits_per_uniform <= 0:
        raise ValueError(
            f"bits_per_uniform must be strictly positive, got {bits_per_uniform}."
        )
    if bits_per_uniform > UNIFORM_RANDOM_BITS:
        raise ValueError(
            f"bits_per_uniform cannot exceed {UNIFORM_RANDOM_BITS}, got {bits_per_uniform}."
        )

    if k == 0:
        return torch.empty((*offsets.shape, 0), device=offsets.device, dtype=torch.int32)

    orig_shape = offsets.shape
    flat_offsets = offsets.reshape(-1).to(torch.int64)

    n_chunks = (k + bits_per_uniform - 1) // bits_per_uniform
    chunk_ids = torch.arange(
        n_chunks, device=flat_offsets.device, dtype=torch.int64
    )
    chunk_offsets = flat_offsets[:, None] + chunk_ids[None, :] * int(chunk_offset_stride)

    uniforms = stateless_uniform(offsets=chunk_offsets.reshape(-1), seed=seed)
    uniforms = uniforms.reshape(flat_offsets.numel(), n_chunks)

    max_int = 1 << bits_per_uniform
    random_ints = torch.floor(uniforms * float(max_int)).to(torch.int64)

    bit_positions = torch.arange(
        bits_per_uniform, device=flat_offsets.device, dtype=torch.int64
    )
    bits = ((random_ints[..., None] >> bit_positions) & 1).to(torch.int32)

    bits = bits.reshape(flat_offsets.numel(), n_chunks * bits_per_uniform)[:, :k]
    return bits.reshape(*orig_shape, k)
