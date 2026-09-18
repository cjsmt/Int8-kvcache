import torch


def make_paged_cache(k, v, block_size=16):
    """k/v: [B, Hkv, T, D] 简化：每个 batch sequence 单独顺序分配 blocks。"""
    B, Hkv, T, D = k.shape
    blocks_per_seq = (T + block_size - 1) // block_size
    num_blocks = B * blocks_per_seq
    kc = torch.zeros(
        num_blocks,
        block_size,
        Hkv,
        D,
        dtype=k.dtype,
        device=k.device,
    )
    vc = torch.zeros_like(kc)
    block_tables = torch.empty(
        B,
        blocks_per_seq,
        dtype=torch.int32,
        device=k.device,
    )
    for b in range(B):
        for lb in range(blocks_per_seq):
            pb = b * blocks_per_seq + lb
            block_tables[b, lb] = pb
            s = lb * block_size
            e = min(s + block_size, T)
            n = e - s
            kc[pb, :n] = k[b, :, s:e, :].permute(1, 0, 2)
            vc[pb, :n] = v[b, :, s:e, :].permute(1, 0, 2)
    seq_lens = torch.full(
        (B,), T, dtype=torch.int32, device=k.device
    )
    return kc, vc, block_tables, seq_lens


def gather_from_paged_cache(kc, vc, block_tables, seq_lens):
    B = block_tables.shape[0]
    block_size = kc.shape[1]
    Hkv = kc.shape[2]
    D = kc.shape[3]
    ks, vs = [], []
    for b in range(B):
        T = int(seq_lens[b])
        parts_k, parts_v = [], []
        nblocks = (T + block_size - 1) // block_size
        for lb in range(nblocks):
            pb = int(block_tables[b, lb])
            parts_k.append(kc[pb])
            parts_v.append(vc[pb])
        k = torch.cat(parts_k, dim=0)[:T]
        v = torch.cat(parts_v, dim=0)[:T]

        # [T, Hkv, D] -> [Hkv, T, D]
        ks.append(k.permute(1, 0, 2))
        vs.append(v.permute(1, 0, 2))
    return torch.stack(ks), torch.stack(vs)
