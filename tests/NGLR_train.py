import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zstandard as zstd

"""

/*
 * The NGLR training code implemented here is based on:
 * Zhu, L., Ranka, S., and Rangarajan, A.
 * "Residual Modeling for High-Fidelity Learned Compression of Scientific Data."
 * arXiv:2606.05389, 2026.
 * DOI: 10.48550/arXiv.2606.05389
 */

"""


def charbonnier(x, eps=1e-6):
    return torch.mean(torch.sqrt(x * x + eps * eps))


def model_path(args):
    return args.ckpt_path


def save_atomic(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def block_slices(shape, bt, bh, bw):
    B, C, T, H, W = shape
    for b in range(B):
        for c in range(C):
            for t0 in range(0, T, bt):
                for h0 in range(0, H, bh):
                    for w0 in range(0, W, bw):
                        yield b, c, t0, min(t0 + bt, T), h0, min(h0 + bh, H), w0, min(
                            w0 + bw, W
                        )


def recons_features(x):
    dt = torch.zeros_like(x)
    dh = torch.zeros_like(x)
    dw = torch.zeros_like(x)
    dt[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
    dh[:, :, :, 1:] = x[:, :, :, 1:] - x[:, :, :, :-1]
    dw[:, :, :, :, 1:] = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    return torch.cat([x, dt, dh, dw, dt.abs(), dh.abs(), dw.abs()], dim=1)


class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.n1 = nn.GroupNorm(min(4, ch), ch)
        self.n2 = nn.GroupNorm(min(4, ch), ch)

    def forward(self, x):
        y = F.gelu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return F.gelu(x + y)


class CausalNeuralLorenzoNet(nn.Module):
    def __init__(self, hidden=32, q_hidden=16, blocks=4):
        super().__init__()
        self.recons_in = nn.Sequential(
            nn.Conv3d(7, hidden, 3, padding=1),
            nn.GroupNorm(min(4, hidden), hidden),
            nn.GELU(),
        )
        self.recons_blocks = nn.Sequential(*[ResBlock3D(hidden) for _ in range(blocks)])
        self.q_branch = nn.Sequential(
            nn.Conv3d(8, q_hidden, 1),
            nn.GELU(),
            nn.Conv3d(q_hidden, q_hidden, 1),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(hidden + q_hidden, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, 1, 1),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def encode_recons(self, r):
        return self.recons_blocks(self.recons_in(recons_features(r)))

    def forward_from_feature(self, rf, qctx):
        return self.fusion(torch.cat([rf, self.q_branch(qctx)], dim=1))

    def forward(self, r, qctx):
        return self.forward_from_feature(self.encode_recons(r), qctx)


def lorenzo_pred(q):
    q = q.astype(np.int64, copy=False)
    p = np.zeros_like(q, dtype=np.int64)
    p[1:] += q[:-1]
    p[:, 1:] += q[:, :-1]
    p[:, :, 1:] += q[:, :, :-1]
    p[1:, 1:] -= q[:-1, :-1]
    p[1:, :, 1:] -= q[:-1, :, :-1]
    p[:, 1:, 1:] -= q[:, :-1, :-1]
    p[1:, 1:, 1:] += q[:-1, :-1, :-1]
    return p


def lorenzo_delta(q):
    return q.astype(np.int64, copy=False) - lorenzo_pred(q)


def q_context(q, scale):
    q = q.astype(np.int64, copy=False)
    ctx = np.zeros((8, *q.shape), dtype=np.float32)
    ctx[0, 1:] = q[:-1]
    ctx[1, :, 1:] = q[:, :-1]
    ctx[2, :, :, 1:] = q[:, :, :-1]
    ctx[3, 1:, 1:] = q[:-1, :-1]
    ctx[4, 1:, :, 1:] = q[:-1, :, :-1]
    ctx[5, :, 1:, 1:] = q[:, :-1, :-1]
    ctx[6, 1:, 1:, 1:] = q[:-1, :-1, :-1]
    ctx[7] = lorenzo_pred(q)
    return ctx / float(max(scale, 1.0))


def lorenzo_pred_torch(q):
    p = torch.zeros_like(q)
    p[1:] += q[:-1]
    p[:, 1:] += q[:, :-1]
    p[:, :, 1:] += q[:, :, :-1]
    p[1:, 1:] -= q[:-1, :-1]
    p[1:, :, 1:] -= q[:-1, :, :-1]
    p[:, 1:, 1:] -= q[:, :-1, :-1]
    p[1:, 1:, 1:] += q[:-1, :-1, :-1]
    return p


def lorenzo_delta_torch(q):
    return q - lorenzo_pred_torch(q)


def q_context_torch(q, scale):
    ctx = torch.zeros((8,) + tuple(q.shape), dtype=torch.float32, device=q.device)
    ctx[0, 1:] = q[:-1]
    ctx[1, :, 1:] = q[:, :-1]
    ctx[2, :, :, 1:] = q[:, :, :-1]
    ctx[3, 1:, 1:] = q[:-1, :-1]
    ctx[4, 1:, :, 1:] = q[:-1, :, :-1]
    ctx[5, :, 1:, 1:] = q[:, :-1, :-1]
    ctx[6, 1:, 1:, 1:] = q[:-1, :-1, :-1]
    ctx[7] = lorenzo_pred_torch(q)
    return ctx / max(float(scale), 1.0)


def decode_sse(x, rn, residual, step, qbuf, ybuf, ebuf, mean, scale):
    step32 = np.float32(step)
    mean32 = np.float32(mean)
    scale32 = np.float32(scale)
    np.divide(residual, step32, out=qbuf)
    np.rint(qbuf, out=qbuf)
    np.multiply(qbuf, step32, out=ybuf)
    np.add(rn, ybuf, out=ybuf)
    np.multiply(ybuf, scale32, out=ybuf)
    np.add(ybuf, mean32, out=ybuf)
    np.subtract(x, ybuf, out=ebuf)
    np.divide(ebuf, scale32, out=ebuf)
    return float(np.sum(ebuf * ebuf))


def zero_sse(x, rn, ybuf, ebuf, mean, scale):
    mean32 = np.float32(mean)
    scale32 = np.float32(scale)
    np.multiply(rn, scale32, out=ybuf)
    np.add(ybuf, mean32, out=ybuf)
    np.subtract(x, ybuf, out=ebuf)
    np.divide(ebuf, scale32, out=ebuf)
    return float(np.sum(ebuf * ebuf))


def quantize_with_step(residual, step):
    qbuf = np.empty_like(residual, dtype=np.float32)
    np.divide(residual, np.float32(step), out=qbuf)
    np.rint(qbuf, out=qbuf)
    return qbuf.astype(np.int32)


def decoded_nrmse(x, rn, q, step, mean, scale):
    y = np.empty_like(x, dtype=np.float32)
    e = np.empty_like(x, dtype=np.float32)
    np.multiply(q, np.float32(step), out=y, casting="unsafe")
    np.add(rn, y, out=y)
    np.multiply(y, np.float32(scale), out=y)
    np.add(y, np.float32(mean), out=y)
    np.subtract(x, y, out=e)
    np.divide(e, np.float32(scale), out=e)
    return float(np.sqrt(np.mean(e * e)))


def safe_global_quantize(x, rn, residual, target, iters, mean, scale):
    qbuf = np.empty_like(residual, dtype=np.float32)
    ybuf = np.empty_like(residual, dtype=np.float32)
    ebuf = np.empty_like(residual, dtype=np.float32)
    target_sse = float(target) * float(target) * residual.size

    if zero_sse(x, rn, ybuf, ebuf, mean, scale) <= target_sse:
        return 1.0, np.zeros(residual.shape, dtype=np.int32), 0.0

    low = 0.0
    high = max(float(target) * np.sqrt(12.0), 1e-12)
    while (
        decode_sse(x, rn, residual, high, qbuf, ybuf, ebuf, mean, scale) <= target_sse
    ):
        low = high
        high *= 2.0

    for _ in range(iters):
        mid = 0.5 * (low + high)
        if (
            decode_sse(x, rn, residual, mid, qbuf, ybuf, ebuf, mean, scale)
            <= target_sse
        ):
            low = mid
        else:
            high = mid

    step = max(low, 1e-12)
    decode_sse(x, rn, residual, step, qbuf, ybuf, ebuf, mean, scale)
    q = qbuf.astype(np.int32)
    final = decoded_nrmse(x, rn, q, step, mean, scale)
    while final > target:
        step *= 0.999999
        q = quantize_with_step(residual, step)
        final = decoded_nrmse(x, rn, q, step, mean, scale)
    return float(step), q, float(final)


def estimate_scales(q, args):
    q_sum = 0.0
    d_sum = 0.0
    n = 0
    for b, c, t0, t1, h0, h1, w0, w1 in block_slices(
        q.shape, args.block_t, args.block_h, args.block_w
    ):
        qb = np.ascontiguousarray(q[b, c, t0:t1, h0:h1, w0:w1])
        d = lorenzo_delta(qb)
        q_sum += float(np.abs(qb).sum())
        d_sum += float(np.abs(d).sum())
        n += qb.size
    return float(max(1.0, q_sum / n)), float(max(1.0, d_sum / n))


def load_npz(path, latent_bit_arg):
    z = np.load(path)
    x = z["original_data"].astype(np.float32, copy=False)
    r = z["recons_data"].astype(np.float32, copy=False)
    if x.ndim == 4:
        x = x[:, None]
    if r.ndim == 4:
        r = r[:, None]
    if x.shape != r.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {r.shape}")
    latent_bit = (
        int(z["latent_bit"]) if "latent_bit" in z.files else int(latent_bit_arg)
    )
    return x, r, latent_bit


def normalize(x, r, mean, scale):
    mean32 = np.float32(mean)
    scale32 = np.float32(scale)
    xn = ((x - mean32) / scale32).astype(np.float32)
    rn = ((r - mean32) / scale32).astype(np.float32)
    return xn, rn, (xn - rn).astype(np.float32)


def load_resume_checkpoint(args):
    path = model_path(args)
    if not args.resume:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    for k in ["hidden", "q_hidden", "model_blocks", "block_t", "block_h", "block_w"]:
        if k in ckpt and getattr(args, k) != int(ckpt[k]):
            print(f"Use checkpoint {k}: {ckpt[k]} instead of arg {getattr(args, k)}")
            setattr(args, k, int(ckpt[k]))
    return ckpt


def prepare_data(args, ckpt):
    x, r, latent_bit = load_npz(args.path, args.latent_bit)
    if ckpt is None:
        mean = float(x.mean())
        scale = float(x.max() - x.min())
    else:
        mean = float(ckpt["mean"])
        scale = float(ckpt["scale"])
    if scale <= 0:
        raise ValueError("invalid data scale")

    xn, rn, residual = normalize(x, r, mean, scale)
    if ckpt is None:
        step, q, final_nrmse = safe_global_quantize(
            x, rn, residual, args.nrmse, args.quant_iter, mean, scale
        )
    else:
        step = float(ckpt["step"])
        q = quantize_with_step(residual, step)
        final_nrmse = decoded_nrmse(x, rn, q, step, mean, scale)

    q_scale, d_scale = estimate_scales(q, args)
    meta = {
        "target": float(args.nrmse),
        "step": float(step),
        "q_scale": float(q_scale),
        "d_scale": float(d_scale),
        "mean": float(mean),
        "scale": float(scale),
        "shape": list(x.shape),
        "original_bytes": int(x.nbytes),
        "latent_bit": int(args.latent_bit or latent_bit),
        "base_nrmse": decoded_nrmse(
            x, rn, np.zeros_like(q, dtype=np.int32), 1.0, mean, scale
        ),
        "quant_nrmse": float(final_nrmse),
        "block_t": int(args.block_t),
        "block_h": int(args.block_h),
        "block_w": int(args.block_w),
    }
    return x, xn, rn, q, meta


def checkpoint(model, args, meta, best_loss, best_epoch):
    out = dict(meta)
    out.update(
        {
            "model": {
                k.replace("_orig_mod.", ""): v.detach().cpu()
                for k, v in model.state_dict().items()
            },
            "hidden": int(args.hidden),
            "q_hidden": int(args.q_hidden),
            "model_blocks": int(args.model_blocks),
            "best_loss": float(best_loss),
            "best_epoch": int(best_epoch),
        }
    )
    return out


def train(rn, q, meta, args, device, ckpt):
    path = model_path(args)
    slices = list(block_slices(q.shape, args.block_t, args.block_h, args.block_w))
    model = torch.compile(
        CausalNeuralLorenzoNet(args.hidden, args.q_hidden, args.model_blocks).to(device)
    )
    best_loss = float("inf")
    best_epoch = 0

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        best_loss = float(ckpt.get("best_loss", best_loss))
        best_epoch = int(ckpt.get("best_epoch", 0))
        print(f"Resume: {path}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    print(f"Target {args.nrmse:g} | blocks {len(slices)} | step {meta['step']:.8e}")
    print(
        f"Base NRMSE {meta['base_nrmse']:.8e} | Quant NRMSE {meta['quant_nrmse']:.8e}"
    )
    print(f"q_scale {meta['q_scale']:.6g} | d_scale {meta['d_scale']:.6g}")
    print(f"Model: {path}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # to do convert to torch
    q_t = torch.from_numpy(q).to(device=device, dtype=torch.int64)
    rn_t = torch.from_numpy(rn).to(device=device, dtype=torch.float32)

    groups = defaultdict(lambda: {"r": [], "c": [], "d": []})
    for b, c, t0, t1, h0, h1, w0, w1 in slices:
        qb = q_t[b, c, t0:t1, h0:h1, w0:w1].contiguous()
        rb = rn_t[b, c, t0:t1, h0:h1, w0:w1].contiguous()
        d = lorenzo_delta_torch(qb).to(torch.float32) / meta["d_scale"]
        ctx = q_context_torch(qb, meta["q_scale"])
        g = groups[tuple(qb.shape)]
        g["r"].append(rb)
        g["c"].append(ctx)
        g["d"].append(d)

    batch_groups = []
    for shape, g in groups.items():
        r_g = torch.stack(g["r"]).unsqueeze(1)  # [N,1,T,H,W]
        c_g = torch.stack(g["c"])  # [N,8,T,H,W]
        d_g = torch.stack(g["d"]).unsqueeze(1)  # [N,1,T,H,W]
        batch_groups.append((r_g, c_g, d_g))
    print(f"Precomputed {len(slices)} blocks into {len(batch_groups)} shape group(s)")

    def run_epoch(profiler=None):
        model.train()
        loss_sum = torch.zeros((), device=device)
        remain_sum = torch.zeros((), device=device)
        n_total = 0

        group_order = torch.randperm(len(batch_groups))
        for gi in group_order:
            r_g, c_g, d_g = batch_groups[gi]
            N = r_g.shape[0]
            perm = torch.randperm(N, device=device)
            for i in range(0, N, args.batch_size):
                idx = perm[i : i + args.batch_size]
                r_b, c_b, d_b = r_g[idx], c_g[idx], d_g[idx]

                pred = model(r_b, c_b)
                remain = d_b - pred
                loss = charbonnier(remain)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                # maybe remove the error_if_nonfinite ?????????
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip, error_if_nonfinite=False
                )
                opt.step()

                bs = idx.numel()
                loss_sum += loss.detach() * bs
                remain_sum += remain.detach().abs().mean() * bs
                n_total += bs
                if profiler is not None:
                    profiler.step()

        return (loss_sum / n_total).item(), (remain_sum / n_total).item() * meta[
            "d_scale"
        ]

    for epoch in range(1, args.train_epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.time()

        if args.profile and epoch == 1:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
            ) as prof:
                avg_loss, avg_remain = run_epoch(profiler=prof)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        else:
            avg_loss, avg_remain = run_epoch()

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t_start

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            if epoch % args.checkpoint_every == 0 or epoch == args.train_epochs:
                save_atomic(checkpoint(model, args, meta, best_loss, best_epoch), path)

        if epoch % args.print_interval == 0 or epoch == args.train_epochs:
            print(
                f"Epoch {epoch} | Loss {avg_loss:.6e} | RemainAbs {avg_remain:.6e} "
                f"| BestEpoch {best_epoch} | Time {epoch_time:.2f}s",
                flush=True,
            )

    if not os.path.exists(path):
        save_atomic(checkpoint(model, args, meta, best_loss, best_epoch), path)
    return path


_DIAG_CACHE = {}


def diagonals(shape):
    key = tuple(shape)
    if key in _DIAG_CACHE:
        return _DIAG_CACHE[key]
    T, H, W = key
    out = []
    for s in range(T + H + W - 1):
        ts_all, hs_all, ws_all = [], [], []
        for t in range(max(0, s - H - W + 2), min(T - 1, s) + 1):
            h0 = max(0, s - t - W + 1)
            h1 = min(H - 1, s - t)
            if h0 > h1:
                continue
            hs = np.arange(h0, h1 + 1, dtype=np.int64)
            ts_all.append(np.full_like(hs, t))
            hs_all.append(hs)
            ws_all.append((s - t - hs).astype(np.int64))
        if ts_all:
            out.append(
                (np.concatenate(ts_all), np.concatenate(hs_all), np.concatenate(ws_all))
            )
    _DIAG_CACHE[key] = out
    return out


def causal_context(qhat, ts, hs, ws, scale):
    n = len(ts)
    v = [np.zeros(n, dtype=np.int64) for _ in range(7)]
    masks = [
        ts > 0,
        hs > 0,
        ws > 0,
        (ts > 0) & (hs > 0),
        (ts > 0) & (ws > 0),
        (hs > 0) & (ws > 0),
        (ts > 0) & (hs > 0) & (ws > 0),
    ]
    idx = [
        (ts - 1, hs, ws),
        (ts, hs - 1, ws),
        (ts, hs, ws - 1),
        (ts - 1, hs - 1, ws),
        (ts - 1, hs, ws - 1),
        (ts, hs - 1, ws - 1),
        (ts - 1, hs - 1, ws - 1),
    ]
    for i, m in enumerate(masks):
        if np.any(m):
            a, b, c = idx[i]
            v[i][m] = qhat[a[m], b[m], c[m]]
    pred = v[0] + v[1] + v[2] - v[3] - v[4] - v[5] + v[6]
    ctx = np.stack(v + [pred], axis=1).astype(np.float32) / float(max(scale, 1.0))
    return ctx, pred


# simple test
@torch.inference_mode()
def strict_delta(qb, rb, model, q_scale, d_scale, device):
    qref = qb.astype(np.int64, copy=False)
    qhat = np.zeros_like(qref, dtype=np.int64)
    delta = np.zeros_like(qref, dtype=np.int64)
    r = torch.from_numpy(rb[None, None].astype(np.float32)).to(device)
    rf = model.encode_recons(r)
    ch = rf.shape[1]

    for ts, hs, ws in diagonals(qref.shape):
        ctx, pred = causal_context(qhat, ts, hs, ws, q_scale)
        qctx = torch.from_numpy(ctx[:, :, None, None, None]).to(device)
        tt = torch.as_tensor(ts, dtype=torch.long, device=device)
        hh = torch.as_tensor(hs, dtype=torch.long, device=device)
        ww = torch.as_tensor(ws, dtype=torch.long, device=device)
        rf_sel = rf[0, :, tt, hh, ww].transpose(0, 1).contiguous().view(-1, ch, 1, 1, 1)
        bias = (
            model.forward_from_feature(rf_sel, qctx)
            .reshape(-1)
            .cpu()
            .numpy()
            .astype(np.float64)
            * d_scale
        )
        ref = np.rint(pred.astype(np.float64) + bias).astype(np.int64)
        delta[ts, hs, ws] = qref[ts, hs, ws] - ref
        qhat[ts, hs, ws] = qref[ts, hs, ws]
    return delta


def zigzag(a):
    a = a.astype(np.int64, copy=False)
    return np.where(a >= 0, a * 2, -2 * a - 1).astype(np.uint64)


def bitplane_size(delta, level):
    flat = zigzag(delta).reshape(-1)
    bits = max(1, int(flat.max()).bit_length() if flat.size else 1)
    cctx = zstd.ZstdCompressor(level=level)
    size = 4 + 8 * bits
    for b in range(bits):
        packed = np.packbits(((flat >> b) & 1).astype(np.uint8), bitorder="little")
        size += len(cctx.compress(packed.tobytes()))
    return size, bits


# simple test
@torch.inference_mode()
def evaluate(model_file, x, rn, q, meta, args, device):
    ckpt = torch.load(model_file, map_location="cpu")
    model = CausalNeuralLorenzoNet(
        ckpt["hidden"], ckpt["q_hidden"], ckpt["model_blocks"]
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stream_bytes = 0
    d_abs_sum = 0.0
    d_zero = 0
    d_num = 0
    bit_counts = []
    slices = list(block_slices(q.shape, args.block_t, args.block_h, args.block_w))
    t0 = time.time()

    for i, (b, c, t0_, t1, h0, h1, w0, w1) in enumerate(slices, 1):
        qb = np.ascontiguousarray(q[b, c, t0_:t1, h0:h1, w0:w1])
        rb = np.ascontiguousarray(rn[b, c, t0_:t1, h0:h1, w0:w1])
        d = strict_delta(qb, rb, model, meta["q_scale"], meta["d_scale"], device)
        s, bc = bitplane_size(d, args.level)
        stream_bytes += s
        bit_counts.append(bc)
        d_abs_sum += float(np.abs(d).sum())
        d_zero += int(np.sum(d == 0))
        d_num += d.size
        if args.progress_interval and (
            i % args.progress_interval == 0 or i == len(slices)
        ):
            print(f"Eval block {i}/{len(slices)}", flush=True)

    final_nrmse = decoded_nrmse(x, rn, q, meta["step"], meta["mean"], meta["scale"])
    model_bytes = os.path.getsize(model_file)
    latent_bytes = meta["latent_bit"] / 8.0 if meta["latent_bit"] > 0 else 0.0
    total_bytes = latent_bytes + stream_bytes + model_bytes
    cr = meta["original_bytes"] / total_bytes if total_bytes > 0 else 0.0

    print("-" * 80)
    print("Best model evaluation")
    print("Best model:", model_file)
    print(
        f"Best loss: {ckpt.get('best_loss', 0.0):.6e} | Best epoch: {ckpt.get('best_epoch', 0)}"
    )
    print(f"Final NRMSE: {final_nrmse:.8e}")
    print(f"CR: {cr:.3f}")
    print("Latent bytes:", latent_bytes)
    print("Correction stream bytes:", stream_bytes)
    print("Model bytes:", model_bytes)
    print("Total bytes:", total_bytes)
    print("delta abs mean:", d_abs_sum / d_num)
    print("delta zero ratio:", d_zero / d_num)
    print("bit planes mean:", float(np.mean(bit_counts)))
    print("Eval time sec:", time.time() - t0)
    print("-" * 80)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--nrmse", type=float, default=1e-5)
    p.add_argument("--block_t", type=int, default=60)
    p.add_argument("--block_h", type=int, default=120)
    p.add_argument("--block_w", type=int, default=120)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--q_hidden", type=int, default=16)
    p.add_argument("--model_blocks", type=int, default=4)
    p.add_argument("--train_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--quant_iter", type=int, default=24)
    p.add_argument("--level", type=int, default=21)
    p.add_argument("--print_interval", type=int, default=1)
    p.add_argument("--progress_interval", type=int, default=80)
    p.add_argument("--device", default="")
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--latent_bit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--checkpoint_every", type=int, default=5)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Device:", device)

    ckpt = load_resume_checkpoint(args)
    x, xn, rn, q, meta = prepare_data(args, ckpt)
    mpath = train(rn, q, meta, args, device, ckpt)
    evaluate(mpath, x, rn, q, meta, args, device)


if __name__ == "__main__":
    main()
