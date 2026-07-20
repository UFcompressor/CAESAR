import argparse
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import struct
import time
from multiprocessing.pool import ThreadPool
from multiprocessing import get_context
# not needed that much above 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zstandard as zstd # can be nvcomp 


MAGIC = b"NGLR_V1\n"


def sync_if_needed(device):
    d = torch.device(device)
    if d.type == "cuda":
        torch.cuda.synchronize(d)


class StageTimer:
    def __init__(self, device):
        self.device = torch.device(device)
        self.times = {}

    def add(self, name, sec):
        self.times[name] = self.times.get(name, 0.0) + float(sec)

    def timed(self, name):
        timer = self

        class Ctx:
            def __enter__(self_inner):
                sync_if_needed(timer.device)
                self_inner.t0 = time.perf_counter()
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                sync_if_needed(timer.device)
                timer.add(name, time.perf_counter() - self_inner.t0)

        return Ctx()


def nrmse_range(x, y):
    return float(np.sqrt(np.mean((x - y) ** 2)))


def decoded_nrmse(original, decoded, scale):
    return float(np.sqrt(np.mean(((original - decoded) / float(scale)) ** 2)))


def make_recons_features(x):
    dt = torch.zeros_like(x)
    dh = torch.zeros_like(x)
    dw = torch.zeros_like(x)
    dt[:, :, 1:, :, :] = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dh[:, :, :, 1:, :] = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dw[:, :, :, :, 1:] = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    return torch.cat([x, dt, dh, dw, dt.abs(), dh.abs(), dw.abs()], dim=1)


class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        g = min(4, ch)
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(g, ch)
        self.norm2 = nn.GroupNorm(g, ch)

    def forward(self, x):
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(x + y)


class CausalNeuralLorenzoNet(nn.Module):
    def __init__(self, recons_hidden=32, q_hidden=16, blocks=4):
        super().__init__()
        g1 = min(4, recons_hidden)
        self.recons_in = nn.Sequential(
            nn.Conv3d(7, recons_hidden, 3, padding=1),
            nn.GroupNorm(g1, recons_hidden),
            nn.GELU(),
        )
        self.recons_blocks = nn.Sequential(*[ResBlock3D(recons_hidden) for _ in range(blocks)])
        self.q_branch = nn.Sequential(
            nn.Conv3d(8, q_hidden, 1),
            nn.GELU(),
            nn.Conv3d(q_hidden, q_hidden, 1),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(recons_hidden + q_hidden, recons_hidden, 1),
            nn.GELU(),
            nn.Conv3d(recons_hidden, recons_hidden, 1),
            nn.GELU(),
            nn.Conv3d(recons_hidden, 1, 1),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def encode_recons(self, recons):
        rf = make_recons_features(recons)
        rf = self.recons_in(rf)
        rf = self.recons_blocks(rf)
        return rf

    def forward_from_recons_feature(self, rf, qctx):
        qf = self.q_branch(qctx)
        x = torch.cat([rf, qf], dim=1)
        return self.fusion(x)

    def forward(self, recons, qctx):
        rf = self.encode_recons(recons)
        return self.forward_from_recons_feature(rf, qctx)


_DIAG_CACHE = {}


def diagonal_indices(shape):
    key = tuple(int(x) for x in shape)
    if key in _DIAG_CACHE:
        return _DIAG_CACHE[key]
    T, H, W = key
    out = []
    for s in range(T + H + W - 2 + 1):
        ts_all, hs_all, ws_all = [], [], []
        t_min = max(0, s - (H - 1) - (W - 1))
        t_max = min(T - 1, s)
        for t in range(t_min, t_max + 1):
            h_min = max(0, s - t - (W - 1))
            h_max = min(H - 1, s - t)
            if h_min > h_max:
                continue
            hs = np.arange(h_min, h_max + 1, dtype=np.int64)
            ws = (s - t - hs).astype(np.int64)
            ts = np.full_like(hs, t, dtype=np.int64)
            ts_all.append(ts)
            hs_all.append(hs)
            ws_all.append(ws)
        if ts_all:
            out.append((np.concatenate(ts_all), np.concatenate(hs_all), np.concatenate(ws_all)))
    _DIAG_CACHE[key] = out
    return out


def lorenzo_context_arrays(q, ts, hs, ws, q_context_scale):
    n = len(ts)
    z = np.zeros(n, dtype=np.int32)
    v1 = z.copy(); v2 = z.copy(); v3 = z.copy(); v4 = z.copy()
    v5 = z.copy(); v6 = z.copy(); v7 = z.copy()

    m = ts > 0
    if np.any(m):
        v1[m] = q[ts[m] - 1, hs[m], ws[m]]
    m = hs > 0
    if np.any(m):
        v2[m] = q[ts[m], hs[m] - 1, ws[m]]
    m = ws > 0
    if np.any(m):
        v3[m] = q[ts[m], hs[m], ws[m] - 1]
    m = (ts > 0) & (hs > 0)
    if np.any(m):
        v4[m] = q[ts[m] - 1, hs[m] - 1, ws[m]]
    m = (ts > 0) & (ws > 0)
    if np.any(m):
        v5[m] = q[ts[m] - 1, hs[m], ws[m] - 1]
    m = (hs > 0) & (ws > 0)
    if np.any(m):
        v6[m] = q[ts[m], hs[m] - 1, ws[m] - 1]
    m = (ts > 0) & (hs > 0) & (ws > 0)
    if np.any(m):
        v7[m] = q[ts[m] - 1, hs[m] - 1, ws[m] - 1]

    pred = v1 + v2 + v3 - v4 - v5 - v6 + v7
    ctx = np.stack([v1, v2, v3, v4, v5, v6, v7, pred], axis=1).astype(np.float32)
    ctx /= float(max(q_context_scale, 1.0))
    return ctx, pred


@torch.inference_mode()
def strict_encode_delta_block_cpu(q_block, r_block, model, meta):
    q_context_scale = float(meta["q_context_scale"])
    delta_scale = float(meta["delta_scale"])
    T, H, W = q_block.shape
    q_ref = q_block.astype(np.int32, copy=False)
    qhat = np.zeros((T, H, W), dtype=np.int32)
    delta = np.zeros((T, H, W), dtype=np.int32)

    r = torch.from_numpy(r_block[None, None].astype(np.float32))
    rf = model.encode_recons(r)
    ch = rf.shape[1]

    for ts, hs, ws in diagonal_indices(q_block.shape):
        ctx_np, pred = lorenzo_context_arrays(qhat, ts, hs, ws, q_context_scale)
        qctx = torch.from_numpy(ctx_np[:, :, None, None, None]).float()
        ts_t = torch.as_tensor(ts, dtype=torch.long)
        hs_t = torch.as_tensor(hs, dtype=torch.long)
        ws_t = torch.as_tensor(ws, dtype=torch.long)
        rf_sel = rf[0, :, ts_t, hs_t, ws_t].transpose(0, 1).contiguous().view(-1, ch, 1, 1, 1)
        bias_norm = model.forward_from_recons_feature(rf_sel, qctx)
        bias = bias_norm.reshape(-1).detach().numpy().astype(np.float64) * delta_scale
        ref = np.rint(pred.astype(np.float64) + bias).astype(np.int32)
        cur = q_ref[ts, hs, ws]
        d64 = cur.astype(np.int64) - ref.astype(np.int64)
        if d64.min() < np.iinfo(np.int32).min or d64.max() > np.iinfo(np.int32).max:
            raise OverflowError("delta does not fit int32")
        d = d64.astype(np.int32)
        delta[ts, hs, ws] = d
        qhat[ts, hs, ws] = (ref.astype(np.int64) + d64).astype(np.int32)

    if not np.array_equal(qhat, q_ref):
        maxdiff = int(np.max(np.abs(qhat - q_ref)))
        raise RuntimeError(f"CPU strict encode mismatch, maxdiff={maxdiff}")
    return delta


@torch.inference_mode()
def strict_decode_delta_block_cpu(delta, r_block, model, meta):
    q_context_scale = float(meta["q_context_scale"])
    delta_scale = float(meta["delta_scale"])
    T, H, W = delta.shape
    qhat = np.zeros((T, H, W), dtype=np.int32)

    r = torch.from_numpy(r_block[None, None].astype(np.float32))
    rf = model.encode_recons(r)
    ch = rf.shape[1]

    for ts, hs, ws in diagonal_indices(delta.shape):
        ctx_np, pred = lorenzo_context_arrays(qhat, ts, hs, ws, q_context_scale)
        qctx = torch.from_numpy(ctx_np[:, :, None, None, None]).float()
        ts_t = torch.as_tensor(ts, dtype=torch.long)
        hs_t = torch.as_tensor(hs, dtype=torch.long)
        ws_t = torch.as_tensor(ws, dtype=torch.long)
        rf_sel = rf[0, :, ts_t, hs_t, ws_t].transpose(0, 1).contiguous().view(-1, ch, 1, 1, 1)
        bias_norm = model.forward_from_recons_feature(rf_sel, qctx)
        bias = bias_norm.reshape(-1).detach().numpy().astype(np.float64) * delta_scale
        ref = np.rint(pred.astype(np.float64) + bias).astype(np.int32)
        qhat[ts, hs, ws] = (ref.astype(np.int64) + delta[ts, hs, ws].astype(np.int64)).astype(np.int32)
    return qhat.astype(np.int32, copy=False)


def torch_lorenzo_context(qhat, ts, hs, ws, q_context_scale):
    nb = qhat.shape[0]
    n = ts.numel()
    z = torch.zeros((nb, n), dtype=torch.int32, device=qhat.device)
    v1 = z.clone(); v2 = z.clone(); v3 = z.clone(); v4 = z.clone()
    v5 = z.clone(); v6 = z.clone(); v7 = z.clone()

    m = ts > 0
    if bool(m.any()):
        v1[:, m] = qhat[:, ts[m] - 1, hs[m], ws[m]]
    m = hs > 0
    if bool(m.any()):
        v2[:, m] = qhat[:, ts[m], hs[m] - 1, ws[m]]
    m = ws > 0
    if bool(m.any()):
        v3[:, m] = qhat[:, ts[m], hs[m], ws[m] - 1]
    m = (ts > 0) & (hs > 0)
    if bool(m.any()):
        v4[:, m] = qhat[:, ts[m] - 1, hs[m] - 1, ws[m]]
    m = (ts > 0) & (ws > 0)
    if bool(m.any()):
        v5[:, m] = qhat[:, ts[m] - 1, hs[m], ws[m] - 1]
    m = (hs > 0) & (ws > 0)
    if bool(m.any()):
        v6[:, m] = qhat[:, ts[m], hs[m] - 1, ws[m] - 1]
    m = (ts > 0) & (hs > 0) & (ws > 0)
    if bool(m.any()):
        v7[:, m] = qhat[:, ts[m] - 1, hs[m] - 1, ws[m] - 1]

    pred = v1 + v2 + v3 - v4 - v5 - v6 + v7
    ctx = torch.stack([v1, v2, v3, v4, v5, v6, v7, pred], dim=2).to(torch.float32)
    ctx = ctx / float(max(q_context_scale, 1.0))
    return ctx, pred


@torch.inference_mode()
def strict_encode_delta_blocks_gpu(q_blocks, r_blocks, model, meta, device):
    q_context_scale = float(meta["q_context_scale"])
    delta_scale = float(meta["delta_scale"])
    q_ref_np = np.stack([x.astype(np.int32, copy=False) for x in q_blocks], axis=0)
    nb, T, H, W = q_ref_np.shape

    q_ref = torch.from_numpy(q_ref_np).to(device=device, dtype=torch.int32)
    qhat = torch.zeros((nb, T, H, W), dtype=torch.int32, device=device)
    delta = torch.zeros((nb, T, H, W), dtype=torch.int32, device=device)

    r_np = np.stack([x.astype(np.float32, copy=False) for x in r_blocks], axis=0)[:, None]
    r = torch.from_numpy(r_np).to(device=device, dtype=torch.float32)
    rf = model.encode_recons(r)
    ch = rf.shape[1]

    for ts_np, hs_np, ws_np in diagonal_indices((T, H, W)):
        ts = torch.as_tensor(ts_np, dtype=torch.long, device=device)
        hs = torch.as_tensor(hs_np, dtype=torch.long, device=device)
        ws = torch.as_tensor(ws_np, dtype=torch.long, device=device)
        ctx, pred = torch_lorenzo_context(qhat, ts, hs, ws, q_context_scale)
        qctx = ctx.reshape(-1, 8, 1, 1, 1)
        rf_sel = rf[:, :, ts, hs, ws].permute(0, 2, 1).contiguous().view(-1, ch, 1, 1, 1)
        bias_norm = model.forward_from_recons_feature(rf_sel, qctx).reshape(nb, -1)
        ref = torch.round(pred.to(torch.float64) + bias_norm.to(torch.float64) * delta_scale).to(torch.int32)
        cur = q_ref[:, ts, hs, ws]
        d = cur - ref
        delta[:, ts, hs, ws] = d
        qhat[:, ts, hs, ws] = ref + d

    if not bool(torch.equal(qhat, q_ref)):
        maxdiff = int((qhat - q_ref).abs().max().item())
        raise RuntimeError(f"GPU strict encode mismatch, maxdiff={maxdiff}")
    return [np.ascontiguousarray(x.cpu().numpy().astype(np.int32, copy=False)) for x in delta]


@torch.inference_mode()
def strict_decode_delta_blocks_gpu(delta_blocks, r_blocks, model, meta, device):
    q_context_scale = float(meta["q_context_scale"])
    delta_scale = float(meta["delta_scale"])
    deltas_np = np.stack([x.astype(np.int32, copy=False) for x in delta_blocks], axis=0)
    nb, T, H, W = deltas_np.shape

    deltas = torch.from_numpy(deltas_np).to(device=device, dtype=torch.int32)
    qhat = torch.zeros((nb, T, H, W), dtype=torch.int32, device=device)

    r_np = np.stack([x.astype(np.float32, copy=False) for x in r_blocks], axis=0)[:, None]
    r = torch.from_numpy(r_np).to(device=device, dtype=torch.float32)
    rf = model.encode_recons(r)
    ch = rf.shape[1]

    for ts_np, hs_np, ws_np in diagonal_indices((T, H, W)):
        ts = torch.as_tensor(ts_np, dtype=torch.long, device=device)
        hs = torch.as_tensor(hs_np, dtype=torch.long, device=device)
        ws = torch.as_tensor(ws_np, dtype=torch.long, device=device)
        ctx, pred = torch_lorenzo_context(qhat, ts, hs, ws, q_context_scale)
        qctx = ctx.reshape(-1, 8, 1, 1, 1)
        rf_sel = rf[:, :, ts, hs, ws].permute(0, 2, 1).contiguous().view(-1, ch, 1, 1, 1)
        bias_norm = model.forward_from_recons_feature(rf_sel, qctx).reshape(nb, -1)
        ref = torch.round(pred.to(torch.float64) + bias_norm.to(torch.float64) * delta_scale).to(torch.int32)
        qhat[:, ts, hs, ws] = ref + deltas[:, ts, hs, ws]

    return [np.ascontiguousarray(x.cpu().numpy().astype(np.int32, copy=False)) for x in qhat]


def zigzag_encode_int32(a):
    a64 = np.ascontiguousarray(a.astype(np.int64, copy=False))
    zz = np.where(a64 >= 0, a64 * 2, -2 * a64 - 1)
    if zz.size > 0 and int(zz.max()) > np.iinfo(np.uint32).max:
        raise OverflowError("zigzag code does not fit uint32")
    return zz.astype(np.uint32, copy=False)


def zigzag_decode_uint32(u):
    u64 = u.astype(np.uint64, copy=False)
    out = np.where((u64 & 1) == 0, u64 >> 1, -((u64 + 1) >> 1))
    return out.astype(np.int32, copy=False)


def bitplane_encode_uint(u, level=21):
    u = np.ascontiguousarray(u)
    flat = u.reshape(-1)
    max_val = int(flat.max()) if flat.size > 0 else 0
    bit_count = max(1, max_val.bit_length())
    if max_val <= np.iinfo(np.uint16).max:
        flat = flat.astype(np.uint16, copy=False)
    else:
        flat = flat.astype(np.uint32, copy=False)
    cctx = zstd.ZstdCompressor(level=level)
    streams = []
    one = np.array(1, dtype=flat.dtype)
    for b in range(bit_count):
        bits = ((flat >> b) & one).astype(np.uint8, copy=False)
        packed = np.packbits(bits, bitorder="little")
        streams.append(cctx.compress(packed.tobytes()))
    return streams, bit_count


def bitplane_decode_uint32(streams, bit_count, shape):
    n = int(np.prod(shape))
    dctx = zstd.ZstdDecompressor()
    flat = np.zeros(n, dtype=np.uint32)
    for b in range(bit_count):
        raw = dctx.decompress(streams[b])
        packed = np.frombuffer(raw, dtype=np.uint8)
        bits = np.unpackbits(packed, bitorder="little")[:n].astype(np.uint32, copy=False)
        flat |= bits << np.uint32(b)
    return flat.reshape(shape)


def codec_encode_delta_block(delta, level=21):
    delta = np.ascontiguousarray(delta.astype(np.int32, copy=False))
    zz = zigzag_encode_int32(delta)
    streams, bit_count = bitplane_encode_uint(zz, level=level)
    delta_abs = np.abs(delta.astype(np.int64, copy=False))
    return {
        "streams": streams,
        "bytes_payload": int(sum(len(s) for s in streams)),
        "bit_count": int(bit_count),
        "num": int(delta.size),
        "delta_min": int(delta.min()),
        "delta_max": int(delta.max()),
        "delta_abs_sum": float(delta_abs.sum()),
        "delta_zero_count": int(np.sum(delta == 0)),
    }


def decode_delta_block_from_streams(streams, bit_count, shape):
    zz = bitplane_decode_uint32(streams, bit_count, shape)
    return zigzag_decode_uint32(zz)


def encode_worker_delta(task):
    idx, delta, level = task
    t0 = time.perf_counter()
    out = codec_encode_delta_block(delta.astype(np.int32, copy=False), level=level)
    out["idx"] = idx
    out["entropy_sec"] = float(time.perf_counter() - t0)
    return out


def decode_worker_delta(task):
    idx, streams, bit_count, shape = task
    delta = decode_delta_block_from_streams(streams, bit_count, shape)
    return idx, delta



_CPU_Q = None
_CPU_RECONS = None
_CPU_MODEL = None
_CPU_META = None


def cpu_pool_init(q_arr, recons_arr, ckpt_path, meta, torch_threads):
    global _CPU_Q, _CPU_RECONS, _CPU_MODEL, _CPU_META
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        torch.set_num_threads(max(1, int(torch_threads)))
        torch.set_num_interop_threads(max(1, int(torch_threads)))
    except Exception:
        pass
    _CPU_Q = q_arr
    _CPU_RECONS = recons_arr
    _CPU_META = meta
    ckpt = torch.load(ckpt_path, map_location="cpu")
    _CPU_MODEL = build_model_from_ckpt(ckpt, torch.device("cpu"))


def _slice_block(arr, sl):
    b, c, t0, t1, h0, h1, w0, w1 = sl
    return np.ascontiguousarray(arr[b, c, t0:t1, h0:h1, w0:w1])


def cpu_encode_block_worker(task):
    idx, sl, level = task
    t0 = time.perf_counter()
    q_block = _slice_block(_CPU_Q, sl)
    r_block = _slice_block(_CPU_RECONS, sl)
    t1 = time.perf_counter()
    delta = strict_encode_delta_block_cpu(q_block, r_block, _CPU_MODEL, _CPU_META)
    t2 = time.perf_counter()
    out = codec_encode_delta_block(delta, level=level)
    t3 = time.perf_counter()
    out["idx"] = idx
    out["prepare_sec"] = float(t1 - t0)
    out["neural_sec"] = float(t2 - t1)
    out["entropy_sec"] = float(t3 - t2)
    out["worker_sec"] = float(t3 - t0)
    return out


def cpu_decode_block_worker(task):
    idx, sl, streams, bit_count, shape = task
    t0 = time.perf_counter()
    delta = decode_delta_block_from_streams(streams, bit_count, shape)
    t1 = time.perf_counter()
    r_block = _slice_block(_CPU_RECONS, sl)
    qhat = strict_decode_delta_block_cpu(delta, r_block, _CPU_MODEL, _CPU_META)
    t2 = time.perf_counter()
    return idx, qhat, float(t1 - t0), float(t2 - t1), float(t2 - t0)


def iter_block_slices(shape, block_t, block_h, block_w):
    B, C, T, H, W = shape
    for b in range(B):
        for c in range(C):
            for t0 in range(0, T, block_t):
                t1 = min(t0 + block_t, T)
                for h0 in range(0, H, block_h):
                    h1 = min(h0 + block_h, H)
                    for w0 in range(0, W, block_w):
                        w1 = min(w0 + block_w, W)
                        yield b, c, t0, t1, h0, h1, w0, w1


def block_shape_from_slice(sl):
    _, _, t0, t1, h0, h1, w0, w1 = sl
    return (t1 - t0, h1 - h0, w1 - w0)


def same_shape_batch(slices, start, max_batch):
    first_shape = block_shape_from_slice(slices[start])
    end = start + 1
    while end < len(slices) and end - start < max_batch:
        if block_shape_from_slice(slices[end]) != first_shape:
            break
        end += 1
    return slices[start:end], first_shape, end


def get_batch_size(args):
    if args.engine == "gpu":
        return max(1, int(args.block_batch))
    return 1


def write_header(f, meta):
    payload = json.dumps(meta, sort_keys=True).encode("utf-8")
    f.write(MAGIC)
    f.write(struct.pack("<Q", len(payload)))
    f.write(payload)


def read_header(f):
    magic = f.read(len(MAGIC))
    if magic != MAGIC:
        raise RuntimeError("Invalid correction file magic.")
    n = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(n).decode("utf-8"))


def write_block_streams(f, streams, bit_count):
    f.write(struct.pack("<I", int(bit_count)))
    header_bytes = 4
    for s in streams:
        f.write(struct.pack("<Q", len(s)))
        f.write(s)
        header_bytes += 8
    return header_bytes + sum(len(s) for s in streams)


def read_or_skip_block_streams(f, keep=True):
    raw = f.read(4)
    if len(raw) != 4:
        raise RuntimeError("Unexpected EOF when reading block bit_count.")
    bit_count = struct.unpack("<I", raw)[0]
    streams = []
    for _ in range(bit_count):
        raw = f.read(8)
        if len(raw) != 8:
            raise RuntimeError("Unexpected EOF when reading stream length.")
        n = struct.unpack("<Q", raw)[0]
        if keep:
            s = f.read(n)
            if len(s) != n:
                raise RuntimeError("Unexpected EOF when reading stream bytes.")
            streams.append(s)
        else:
            f.seek(n, os.SEEK_CUR)
    return streams, bit_count


def load_npz_data(path):
    data = np.load(path)
    original = data["original_data"].astype(np.float32) if "original_data" in data else None
    recons = data["recons_data"].astype(np.float32)
    original_raw_shape = None if original is None else original.shape
    recons_raw_shape = recons.shape
    if original is not None and original.ndim == 4:
        original = original[:, None]
    if recons.ndim == 4:
        recons = recons[:, None]
    if original is not None and original.shape != recons.shape:
        raise ValueError(f"shape mismatch: {original.shape} vs {recons.shape}")
    return data, original, recons, original_raw_shape, recons_raw_shape


def build_model_from_ckpt(ckpt, device):
    model = CausalNeuralLorenzoNet(
        recons_hidden=int(ckpt.get("hidden", ckpt.get("recons_hidden", 32))),
        q_hidden=int(ckpt.get("q_hidden", 16)),
        blocks=int(ckpt.get("model_blocks", ckpt.get("blocks", 4))),
    ).to(device)
    state = ckpt["model"]
    try:
        model.load_state_dict(state)
    except RuntimeError:
        remap = {}
        for k, v in state.items():
            k2 = k.replace(".c1.", ".conv1.").replace(".c2.", ".conv2.")
            k2 = k2.replace(".n1.", ".norm1.").replace(".n2.", ".norm2.")
            remap[k2] = v
        model.load_state_dict(remap)
    model.eval()
    return model


def get_codec_meta(ckpt):
    def pick(*names):
        for name in names:
            if name in ckpt:
                return ckpt[name]
        raise ValueError("checkpoint missing key: " + "/".join(names))

    return {
        "x_mean": float(pick("x_mean", "mean")),
        "scale": float(pick("scale")),
        "step": float(pick("step")),
        "q_context_scale": float(pick("q_context_scale", "q_scale")),
        "delta_scale": float(pick("delta_scale", "d_scale")),
        "block_t": int(pick("block_t")),
        "block_h": int(pick("block_h")),
        "block_w": int(pick("block_w")),
    }


def maybe_check_shape(ckpt, arr_shape):
    expected = ckpt.get("original_shape", ckpt.get("shape", None))
    if expected is not None and list(arr_shape) != list(expected):
        raise ValueError(f"shape mismatch with checkpoint: data {arr_shape}, ckpt {expected}")


def default_correction_path(ckpt_path, engine):
    root, _ = os.path.splitext(ckpt_path)
    return root + f"_{engine}.cnlz"


def get_block_arrays(arr, slices):
    out = []
    for b, c, t0, t1, h0, h1, w0, w1 in slices:
        out.append(np.ascontiguousarray(arr[b, c, t0:t1, h0:h1, w0:w1]))
    return out


def encode_deltas_for_batch(q_blocks, r_blocks, model, meta, device, args):
    return strict_encode_delta_blocks_gpu(q_blocks, r_blocks, model, meta, device)


def decode_qhat_for_batch(delta_blocks, r_blocks, model, meta, device, args):
    return strict_decode_delta_blocks_gpu(delta_blocks, r_blocks, model, meta, device)



def codec_encode_and_write_cpu(q, recons_n, meta, args, original_bytes):
    slices = list(iter_block_slices(q.shape, meta["block_t"], meta["block_h"], meta["block_w"]))
    timer = StageTimer(torch.device("cpu"))
    tmp_path = args.correction_path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.correction_path)), exist_ok=True)

    file_meta = dict(meta)
    file_meta.update({
        "num_blocks": int(len(slices)),
        "zstd_level": int(args.level),
        "ckpt_file": os.path.basename(args.ckpt_path),
        "model_format": str(args.model_format),
        "engine": "cpu",
        "workers": int(args.workers),
        "delta_dtype": "int32",
        "format": "bitplane_zstd_cpu_optimized_multiprocess_int32_uint16_v1",
    })

    correction_payload_bytes = 0
    correction_stream_file_bytes = 0
    delta_abs_sum = 0.0
    delta_zero_count = 0
    total_num = 0
    bit_counts = []
    global_delta_min = None
    global_delta_max = None
    neural_cpu_sum = 0.0
    entropy_cpu_sum = 0.0
    worker_cpu_sum = 0.0
    done = 0
    next_to_write = 0
    buffer = {}

    ctx = get_context("fork")
    tasks = [(idx, sl, args.level) for idx, sl in enumerate(slices)]

    def write_out(f, out):
        nonlocal correction_payload_bytes, correction_stream_file_bytes
        nonlocal delta_abs_sum, delta_zero_count, total_num, bit_counts
        nonlocal global_delta_min, global_delta_max, done
        nonlocal neural_cpu_sum, entropy_cpu_sum, worker_cpu_sum
        neural_cpu_sum += float(out.get("neural_sec", 0.0))
        entropy_cpu_sum += float(out.get("entropy_sec", 0.0))
        worker_cpu_sum += float(out.get("worker_sec", 0.0))
        correction_payload_bytes += out["bytes_payload"]
        correction_stream_file_bytes += write_block_streams(f, out["streams"], out["bit_count"])
        delta_abs_sum += out["delta_abs_sum"]
        delta_zero_count += out["delta_zero_count"]
        total_num += out["num"]
        bit_counts.append(out["bit_count"])
        global_delta_min = out["delta_min"] if global_delta_min is None else min(global_delta_min, out["delta_min"])
        global_delta_max = out["delta_max"] if global_delta_max is None else max(global_delta_max, out["delta_max"])
        done += 1

    with ctx.Pool(
        processes=max(1, int(args.workers)),
        initializer=cpu_pool_init,
        initargs=(q, recons_n, args.ckpt_path, meta, args.torch_num_threads),
    ) as pool:
        with timer.timed("encode_total_wall"):
            with open(tmp_path, "wb") as f:
                with timer.timed("write_header"):
                    write_header(f, file_meta)
                iterator = pool.imap_unordered(cpu_encode_block_worker, tasks, chunksize=max(1, int(args.cpu_chunksize)))
                for out in iterator:
                    buffer[out["idx"]] = out
                    while next_to_write in buffer:
                        cur = buffer.pop(next_to_write)
                        with timer.timed("file_write"):
                            write_out(f, cur)
                        next_to_write += 1
    os.replace(tmp_path, args.correction_path)
    correction_file_bytes = os.path.getsize(args.correction_path)
    if total_num == 0:
        total_num = 1
    timer.times["neural_cpu_sum"] = neural_cpu_sum
    timer.times["entropy_cpu_sum"] = entropy_cpu_sum
    timer.times["worker_cpu_sum"] = worker_cpu_sum
    return {
        "num_blocks": int(len(slices)),
        "correction_payload_bytes": int(correction_payload_bytes),
        "correction_stream_file_bytes": int(correction_stream_file_bytes),
        "correction_file_bytes": int(correction_file_bytes),
        "delta_min": int(global_delta_min),
        "delta_max": int(global_delta_max),
        "delta_abs_mean": float(delta_abs_sum / total_num),
        "delta_zero_ratio": float(delta_zero_count / total_num),
        "bit_count_max": int(max(bit_counts)) if bit_counts else 0,
        "bit_count_mean": float(np.mean(bit_counts)) if bit_counts else 0.0,
        "timings": timer.times,
    }


def codec_encode_and_write(q, recons_n, model, meta, device, args, original_bytes):
    slices = list(iter_block_slices(q.shape, meta["block_t"], meta["block_h"], meta["block_w"]))
    batch_size = get_batch_size(args)
    timer = StageTimer(device)

    tmp_path = args.correction_path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.correction_path)), exist_ok=True)

    file_meta = dict(meta)
    file_meta.update({
        "num_blocks": int(len(slices)),
        "zstd_level": int(args.level),
        "ckpt_file": os.path.basename(args.ckpt_path),
        "model_format": str(args.model_format),
        "engine": str(args.engine),
        "block_batch": int(batch_size),
        "delta_dtype": "int32",
        "format": "bitplane_zstd_cpu_pipeline_int32_uint16_v2",
    })

    correction_payload_bytes = 0
    correction_stream_file_bytes = 0
    delta_abs_sum = 0.0
    delta_zero_count = 0
    total_num = 0
    bit_counts = []
    global_delta_min = None
    global_delta_max = None
    done_generated = 0
    done_written = 0
    entropy_cpu_sum = 0.0

    pool = ThreadPool(max(1, int(args.workers)))
    pending = {}
    next_to_write = 0

    def write_out(f, out):
        nonlocal correction_payload_bytes, correction_stream_file_bytes
        nonlocal delta_abs_sum, delta_zero_count, total_num, bit_counts
        nonlocal global_delta_min, global_delta_max, done_written, entropy_cpu_sum
        entropy_cpu_sum += float(out.get("entropy_sec", 0.0))
        correction_payload_bytes += out["bytes_payload"]
        correction_stream_file_bytes += write_block_streams(f, out["streams"], out["bit_count"])
        delta_abs_sum += out["delta_abs_sum"]
        delta_zero_count += out["delta_zero_count"]
        total_num += out["num"]
        bit_counts.append(out["bit_count"])
        global_delta_min = out["delta_min"] if global_delta_min is None else min(global_delta_min, out["delta_min"])
        global_delta_max = out["delta_max"] if global_delta_max is None else max(global_delta_max, out["delta_max"])
        done_written += 1

    def flush_ready(f, block=False):
        nonlocal next_to_write
        while next_to_write in pending:
            fut = pending[next_to_write]
            if (not block) and (not fut.ready()):
                break
            with timer.timed("entropy_wait"):
                out = fut.get()
            if out["idx"] != next_to_write:
                raise RuntimeError("Internal block order error.")
            del pending[next_to_write]
            with timer.timed("file_write"):
                write_out(f, out)
            next_to_write += 1

    try:
        with timer.timed("encode_total_wall"):
            with open(tmp_path, "wb") as f:
                with timer.timed("write_header"):
                    write_header(f, file_meta)

                i = 0
                while i < len(slices):
                    batch_slices, _, next_i = same_shape_batch(slices, i, batch_size)
                    q_blocks = get_block_arrays(q, batch_slices)
                    r_blocks = get_block_arrays(recons_n, batch_slices)

                    with timer.timed("neural_delta"):
                        deltas = encode_deltas_for_batch(q_blocks, r_blocks, model, meta, device, args)

                    with timer.timed("entropy_submit"):
                        for j, delta in enumerate(deltas):
                            idx = i + j
                            pending[idx] = pool.apply_async(encode_worker_delta, ((idx, delta.astype(np.int32, copy=False), args.level),))
                            done_generated += 1

                    flush_ready(f, block=False)
                    if args.max_pending > 0:
                        while len(pending) >= args.max_pending:
                            flush_ready(f, block=True)

                    i = next_i

                while next_to_write < len(slices):
                    flush_ready(f, block=True)
    finally:
        pool.close()
        pool.join()

    os.replace(tmp_path, args.correction_path)
    correction_file_bytes = os.path.getsize(args.correction_path)
    if total_num == 0:
        total_num = 1
    timer.times["entropy_cpu_sum"] = entropy_cpu_sum

    return {
        "num_blocks": int(len(slices)),
        "correction_payload_bytes": int(correction_payload_bytes),
        "correction_stream_file_bytes": int(correction_stream_file_bytes),
        "correction_file_bytes": int(correction_file_bytes),
        "delta_min": int(global_delta_min),
        "delta_max": int(global_delta_max),
        "delta_abs_mean": float(delta_abs_sum / total_num),
        "delta_zero_ratio": float(delta_zero_count / total_num),
        "bit_count_max": int(max(bit_counts)) if bit_counts else 0,
        "bit_count_mean": float(np.mean(bit_counts)) if bit_counts else 0.0,
        "timings": timer.times,
    }


def decode_full_correction_cpu(recons_n, meta, correction_path, args, original_bytes):
    slices = list(iter_block_slices(recons_n.shape, meta["block_t"], meta["block_h"], meta["block_w"]))
    qhat_all = np.zeros(recons_n.shape, dtype=np.int32)
    timer = StageTimer(torch.device("cpu"))
    tasks = []

    entropy_cpu_sum = 0.0
    neural_cpu_sum = 0.0
    worker_cpu_sum = 0.0
    done = 0
    ctx = get_context("fork")
    with ctx.Pool(
        processes=max(1, int(args.workers)),
        initializer=cpu_pool_init,
        initargs=(None, recons_n, args.ckpt_path, meta, args.torch_num_threads),
    ) as pool:
        with timer.timed("decode_total_wall"):
            with open(correction_path, "rb") as f:
                with timer.timed("read_header"):
                    _ = read_header(f)
                with timer.timed("file_read"):
                    for idx, sl in enumerate(slices):
                        streams, bit_count = read_or_skip_block_streams(f, keep=True)
                        tasks.append((idx, sl, streams, bit_count, block_shape_from_slice(sl)))

            with timer.timed("cpu_decode_workers_wall"):
                iterator = pool.imap_unordered(cpu_decode_block_worker, tasks, chunksize=max(1, int(args.cpu_chunksize)))
                for idx, qhat, ent_s, neu_s, work_s in iterator:
                    entropy_cpu_sum += ent_s
                    neural_cpu_sum += neu_s
                    worker_cpu_sum += work_s
                    sl = slices[idx]
                    b, c, t0, t1, h0, h1, w0, w1 = sl
                    with timer.timed("copy_output"):
                        qhat_all[b, c, t0:t1, h0:h1, w0:w1] = qhat
                    done += 1
            timer.times["entropy_decode_cpu_sum"] = entropy_cpu_sum
            timer.times["neural_decode_cpu_sum"] = neural_cpu_sum
            timer.times["worker_cpu_sum"] = worker_cpu_sum
    return qhat_all, timer.times


def decode_full_correction(recons_n, model, meta, correction_path, args, device, original_bytes):
    slices = list(iter_block_slices(recons_n.shape, meta["block_t"], meta["block_h"], meta["block_w"]))
    batch_size = get_batch_size(args)
    qhat_all = np.zeros(recons_n.shape, dtype=np.int32)
    timer = StageTimer(device)
    pool = ThreadPool(args.workers) if args.workers > 1 else None
    done = 0

    with timer.timed("decode_total_wall"):
        with open(correction_path, "rb") as f:
            with timer.timed("read_header"):
                _ = read_header(f)

            i = 0
            while i < len(slices):
                batch_slices, block_shape, next_i = same_shape_batch(slices, i, batch_size)

                read_tasks = []
                with timer.timed("file_read"):
                    for j in range(len(batch_slices)):
                        streams, bit_count = read_or_skip_block_streams(f, keep=True)
                        read_tasks.append((i + j, streams, bit_count, block_shape))

                with timer.timed("entropy_decode"):
                    if pool is None:
                        decoded = [decode_worker_delta(t) for t in read_tasks]
                    else:
                        decoded = list(pool.imap(decode_worker_delta, read_tasks, chunksize=1))
                    decoded.sort(key=lambda x: x[0])
                    delta_blocks = [x[1] for x in decoded]

                r_blocks = get_block_arrays(recons_n, batch_slices)
                with timer.timed("neural_decode"):
                    qhat_blocks = decode_qhat_for_batch(delta_blocks, r_blocks, model, meta, device, args)

                with timer.timed("copy_output"):
                    for sl, qhat in zip(batch_slices, qhat_blocks):
                        b, c, t0, t1, h0, h1, w0, w1 = sl
                        qhat_all[b, c, t0:t1, h0:h1, w0:w1] = qhat
                        done += 1

                i = next_i

    if pool is not None:
        pool.close()
        pool.join()
    return qhat_all, timer.times


def read_latent_bit_from_npz(data):
    if "latent_bit" not in data:
        return 0
    return int(data["latent_bit"])


def resolve_device(args):
    if args.engine == "cpu":
        return torch.device("cpu")
    if args.device:
        return torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU engine requires CUDA, but CUDA is not available.")
    return torch.device("cuda")


def run_encode(args):
    if not args.correction_path:
        args.correction_path = default_correction_path(args.ckpt_path, args.engine)

    device = resolve_device(args)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    args.model_format = ckpt.get("model_format", "strict_causal_no_qnorm_v2")
    meta = get_codec_meta(ckpt)
    model = None if args.engine == "cpu" else build_model_from_ckpt(ckpt, device)

    load_t0 = time.perf_counter()
    data, original, recons, original_raw_shape, _ = load_npz_data(args.path)
    load_time = time.perf_counter() - load_t0
    if original is None:
        raise ValueError("encode mode needs original_data in npz")
    maybe_check_shape(ckpt, original.shape)

    original_bytes = int(original.nbytes)
    norm_t0 = time.perf_counter()
    recons_n = ((recons - meta["x_mean"]) / meta["scale"]).astype(np.float32)
    residual_n = ((original - meta["x_mean"]) / meta["scale"] - recons_n).astype(np.float32)
    q = np.rint(residual_n / meta["step"]).astype(np.int32)
    norm_quant_time = time.perf_counter() - norm_t0

    quant_decoded = ((recons_n + q.astype(np.float32) * float(meta["step"])) * float(meta["scale"]) + float(meta["x_mean"])).astype(np.float32)
    quant_nrmse = decoded_nrmse(original, quant_decoded, meta["scale"])

    if args.engine == "cpu":
        out = codec_encode_and_write_cpu(q, recons_n, meta, args, original_bytes)
    else:
        out = codec_encode_and_write(q, recons_n, model, meta, device, args, original_bytes)
    out["timings"]["load_npz"] = load_time
    out["timings"]["normalize_quantize"] = norm_quant_time

    latent_bit = read_latent_bit_from_npz(data)
    latent_bytes = latent_bit / 8.0 if latent_bit > 0 else 0.0
    model_bytes = os.path.getsize(args.ckpt_path)
    total_bytes = latent_bytes + out["correction_file_bytes"] + model_bytes
    cr = original_bytes / total_bytes if total_bytes > 0 else 0.0

    encode_wall = out["timings"].get("encode_total_wall", 0.0)
    end_to_end = load_time + norm_quant_time + encode_wall
    out["timings"]["end_to_end_estimated"] = end_to_end

    print("Encode complete")
    print(f"Correction file: {args.correction_path}")
    print(f"Final NRMSE: {quant_nrmse:.8e}")
    print(f"CR: {cr:.3f}")
    print(f"Correction bytes: {out['correction_file_bytes']}")
    print(f"Model bytes: {model_bytes}")
    print(f"Latent bytes: {latent_bytes:.1f}")
    print(f"Total bytes: {total_bytes:.1f}")
    print(f"Encode time: {end_to_end:.3f} sec")

def run_decode(args):
    if not args.correction_path:
        raise ValueError("decode mode requires --correction_path")

    device = resolve_device(args)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    args.model_format = ckpt.get("model_format", "strict_causal_no_qnorm_v2")
    meta = get_codec_meta(ckpt)
    model = None if args.engine == "cpu" else build_model_from_ckpt(ckpt, device)

    load_t0 = time.perf_counter()
    data, original, recons, _, recons_raw_shape = load_npz_data(args.path)
    load_time = time.perf_counter() - load_t0
    original_bytes = int(original.nbytes) if original is not None else int(recons.nbytes)

    norm_t0 = time.perf_counter()
    recons_n = ((recons - meta["x_mean"]) / meta["scale"]).astype(np.float32)
    norm_time = time.perf_counter() - norm_t0

    if args.engine == "cpu":
        qhat, timings = decode_full_correction_cpu(recons_n, meta, args.correction_path, args, original_bytes)
    else:
        qhat, timings = decode_full_correction(recons_n, model, meta, args.correction_path, args, device, original_bytes)
    timings["load_npz"] = load_time
    timings["normalize_recons"] = norm_time

    recon_t0 = time.perf_counter()
    decoded_n = recons_n + qhat.astype(np.float32) * float(meta["step"])
    decoded = (decoded_n * float(meta["scale"]) + float(meta["x_mean"])).astype(np.float32)
    recon_time = time.perf_counter() - recon_t0
    timings["final_reconstruct"] = recon_time

    final_nrmse = None
    if original is not None:
        eval_t0 = time.perf_counter()
        final_nrmse = decoded_nrmse(original, decoded, meta["scale"])
        timings["eval_nrmse"] = time.perf_counter() - eval_t0

    if len(recons_raw_shape) == 4 and decoded.ndim == 5 and decoded.shape[1] == 1:
        decoded_to_save = decoded[:, 0]
    else:
        decoded_to_save = decoded

    latent_bit = read_latent_bit_from_npz(data)
    latent_bytes = latent_bit / 8.0 if latent_bit > 0 else 0.0
    correction_file_bytes = os.path.getsize(args.correction_path)
    model_bytes = os.path.getsize(args.ckpt_path)
    total_bytes = latent_bytes + correction_file_bytes + model_bytes
    cr = original_bytes / total_bytes if total_bytes > 0 else 0.0

    decode_wall = timings.get("decode_total_wall", 0.0)
    end_to_end = load_time + norm_time + decode_wall + timings.get("final_reconstruct", 0.0)
    if "eval_nrmse" in timings:
        end_to_end += timings["eval_nrmse"]
    timings["end_to_end_estimated"] = end_to_end

    print("Decode complete")
    if final_nrmse is not None:
        print(f"Final NRMSE: {final_nrmse:.8e}")
    else:
        print("Final NRMSE: N/A")
    print(f"CR: {cr:.3f}")
    print(f"Correction bytes: {correction_file_bytes}")
    print(f"Model bytes: {model_bytes}")
    print(f"Latent bytes: {latent_bytes:.1f}")
    print(f"Total bytes: {total_bytes:.1f}")
    print(f"Decode time: {end_to_end:.3f} sec")

    if args.save_decode:
        out_path = args.decode_out or os.path.splitext(args.correction_path)[0] + "_decoded.npz"
        if final_nrmse is not None:
            np.savez(out_path, decoded_data=decoded_to_save, final_nrmse=final_nrmse)
        else:
            np.savez(out_path, decoded_data=decoded_to_save)
        print("Saved decode output:", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["encode", "decode"])
    parser.add_argument("--engine", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--correction_path", type=str, default="")
    parser.add_argument("--save_decode", action="store_true")
    parser.add_argument("--decode_out", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--level", type=int, default=21)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--block_batch", type=int, default=8)
    parser.add_argument("--max_pending", type=int, default=0)
    parser.add_argument("--torch_num_threads", type=int, default=1)
    parser.add_argument("--cpu_chunksize", type=int, default=1)
    args = parser.parse_args()

    try:
        torch.set_num_threads(max(1, int(args.torch_num_threads)))
        torch.set_num_interop_threads(max(1, int(args.torch_num_threads)))
    except Exception:
        pass

    if args.mode == "encode":
        run_encode(args)
    else:
        run_decode(args)


if __name__ == "__main__":
    main()
