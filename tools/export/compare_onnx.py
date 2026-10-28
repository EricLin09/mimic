"""Compare two exports: weights, graph structure and outputs on fixed inputs.

Usage: compare_onnx.py <dir_a> <dir_b> [graph ...]
       (graphs: synth synth_det retrieval, a voice's; contentvec rmvpe, the shared ones in models/)

For every graph present in both directories, prints how many initializers are bit-identical (matched
by name, then by position), the op-type counts when they differ, and the element-wise difference of
the outputs on fixed inputs. synth.onnx draws its own noise, so every random node of both graphs gets
the same seed, by order of appearance, before running.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

RANDOM_OPS = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike"}


def seeded(model: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(model.SerializeToString())
    k = 0
    for node in model.graph.node:
        if node.op_type in RANDOM_OPS:
            for attr in list(node.attribute):
                if attr.name == "seed":
                    node.attribute.remove(attr)
            node.attribute.append(onnx.helper.make_attribute("seed", float(1000 + k)))
            k += 1
    return model


def weights(model: onnx.ModelProto):
    return [(t.name, numpy_helper.to_array(t)) for t in model.graph.initializer]


def compare_weights(a, b):
    wa, wb = weights(a), weights(b)
    by_name = dict(wb)
    pairs = [(x, by_name[n]) for n, x in wa if n in by_name]
    how = "name"
    if len(pairs) != len(wa) or len(wa) != len(wb):  # names differ (e.g. renamed scopes): fall back to order
        pairs, how = [(x, y) for (_, x), (_, y) in zip(wa, wb)], "position"
    same = sum(x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes() for x, y in pairs)
    worst = max((float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())
                 for x, y in pairs if x.shape == y.shape and x.size and x.dtype.kind == "f"), default=0.0)
    total = sum(x.size for _, x in wa)
    print(f"  initializers: {len(wa)} vs {len(wb)} ({total:,} values), matched by {how}: "
          f"{same}/{len(pairs)} bit-identical, max|diff| {worst:.3e}")


def compare_nodes(a, b):
    ca, cb = Counter(n.op_type for n in a.graph.node), Counter(n.op_type for n in b.graph.node)
    if ca == cb:
        print(f"  nodes: {sum(ca.values())} in both, same op types")
    else:
        diff = {op: (ca[op], cb[op]) for op in sorted(set(ca) | set(cb)) if ca[op] != cb[op]}
        print(f"  nodes: {sum(ca.values())} vs {sum(cb.values())}, differing op counts {diff}")


def inputs(graph, rng, frames=300, skip=150, hop=400):
    if graph == "contentvec":
        return {"audio": (0.1 * rng.standard_normal((1, 48000))).astype(np.float32)}
    if graph == "rmvpe":  # log-mel-like values, frames a multiple of 32
        return {"mel": (2 * rng.standard_normal((1, 128, 128)) - 6).astype(np.float32)}
    f0 = (100 + 60 * rng.random((1, frames))).astype(np.float32)
    f0[:, 40:58] = 0  # an unvoiced stretch
    mel = 1127.0 * np.log(1.0 + f0 / 700.0)
    coarse = np.clip(np.rint((mel - 50.0) * 254 / 1050.0 + 1), 1, 255).astype(np.int64)
    common = {
        "phone": rng.standard_normal((1, frames, 768)).astype(np.float32),
        "phone_lengths": np.array([frames], np.int64),
        "pitch": coarse,
        "pitchf": f0,
        "sid": np.array([0], np.int64),
        "skip_head": np.array(skip, np.int64),
    }
    if graph == "synth":
        return common
    if graph == "synth_det":
        return {**common,
                "zp_noise": rng.standard_normal((1, 192, frames)).astype(np.float32),
                "src_noise": rng.standard_normal((1, (frames - skip) * hop, 1)).astype(np.float32),
                "phase0": np.array(0.37, np.float32)}
    return None


def retrieval_inputs(model, rng):
    vectors = next(numpy_helper.to_array(t) for t in model.graph.initializer if len(t.dims) == 2 and t.dims[1] == 768)
    feats = vectors[rng.integers(0, vectors.shape[0], 200)] + 0.05 * rng.standard_normal((200, 768))
    return {"feats": feats.astype(np.float32), "index_rate": np.array(0.75, np.float32)}


def run(model, feed):
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, feed)[0]


def main():
    dir_a, dir_b = Path(sys.argv[1]), Path(sys.argv[2])
    graphs = sys.argv[3:] or ["synth", "synth_det", "retrieval"]
    for graph in graphs:
        pa, pb = dir_a / f"{graph}.onnx", dir_b / f"{graph}.onnx"
        if not (pa.exists() and pb.exists()):
            print(f"{graph}: missing in one of the directories, skipped")
            continue
        a, b = onnx.load(str(pa)), onnx.load(str(pb))
        print(f"{graph}: {pa.stat().st_size:,} vs {pb.stat().st_size:,} bytes")
        compare_weights(a, b)
        compare_nodes(a, b)
        n_random = sum(n.op_type in RANDOM_OPS for n in a.graph.node)
        if n_random:
            a, b = seeded(a), seeded(b)
        rng = np.random.default_rng(7)
        feed = retrieval_inputs(a, rng) if graph == "retrieval" else inputs(graph, rng)
        ya, yb = run(a, feed), run(b, feed)
        if ya.shape != yb.shape:
            print(f"  output shapes differ: {ya.shape} vs {yb.shape}")
            continue
        d = np.abs(ya.astype(np.float64) - yb.astype(np.float64))
        peak = float(np.abs(ya).max())
        print(f"  output {ya.shape}{f', {n_random} random nodes seeded' if n_random else ''}: "
              f"bit-identical {ya.tobytes() == yb.tobytes()}, max|diff| {d.max():.3e}, mean|diff| {d.mean():.3e}, "
              f"peak |a| {peak:.3f}")


if __name__ == "__main__":
    main()
