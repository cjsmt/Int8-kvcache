import torch
from vllm_int8.vllm_int8_cache_ops import runtime_int8_cache_write


def torch_ref(key, value, kc, vc, slots, ks, vs):
    BS = kc.shape[1]
    for t in range(key.shape[0]):
        slot = int(slots[t])
        if slot < 0:
            continue
        b = slot // BS
        s = slot % BS
        kq = torch.round(key[t].float() / ks[:, None])
        vq = torch.round(value[t].float() / vs[:, None])
        kc[b, s] = torch.clamp(kq, -127, 127).to(torch.int8)
        vc[b, s] = torch.clamp(vq, -127, 127).to(torch.int8)


def main():
    torch.manual_seed(0)
    N, H, D = 23, 4, 128
    BS, NB = 16, 16

    key = torch.randn(N, H, D, device="cuda", dtype=torch.bfloat16)
    val = torch.randn_like(key)

    ks = (key.float().abs().amax(dim=(0, 2)) / 127).clamp_min(1e-6)
    vs = (val.float().abs().amax(dim=(0, 2)) / 127).clamp_min(1e-6)

    # 故意跨 block，且不从 0 开始。
    slots = torch.arange(7, 7 + N, device="cuda", dtype=torch.long)

    kc_ref = torch.zeros(NB, BS, H, D, device="cuda", dtype=torch.int8)
    vc_ref = torch.zeros_like(kc_ref)
    kc_tri = torch.zeros_like(kc_ref)
    vc_tri = torch.zeros_like(kc_ref)

    torch_ref(key, val, kc_ref, vc_ref, slots, ks, vs)
    runtime_int8_cache_write(key, val, kc_tri, vc_tri, slots, ks, vs)
    torch.cuda.synchronize()

    kd = (kc_ref.to(torch.int16) - kc_tri.to(torch.int16)).abs().max().item()
    vd = (vc_ref.to(torch.int16) - vc_tri.to(torch.int16)).abs().max().item()
    print("K max integer diff:", kd)
    print("V max integer diff:", vd)
    assert kd <= 1
    assert vd <= 1
    print("PASS")


if __name__ == "__main__":
    main()
