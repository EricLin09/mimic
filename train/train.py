"""Fine-tune an RVC v2 voice (40 kHz, f0, NSF HiFi-GAN) on a `mimic prepare` folder, then export it.

  train\\.venv\\Scripts\\python.exe train\\train.py <prepare_dir> <run_dir> [--epochs 300] [--export <voice_id>]

A minimal single-GPU version of Applio's rvc/train/train.py (MIT, see rvc/LICENSE): the same networks,
losses and weights (mel x45, KL x1, feature matching, least-squares GAN), AdamW with per-epoch decay,
fp16 autocast with one GradScaler, starting from the f0G40k / f0D40k base models, and Applio's
length-bucketed batches. Left out: multi-GPU, TensorBoard, other vocoders, the UI hooks.

<run_dir> receives config.json, metrics.csv (per-epoch means), G_<step>.pth / D_<step>.pth (full state;
running the same command again resumes from them), <name>_<epoch>e_<step>s.pth inference checkpoints
every --save-every epochs in Applio's format (what tools/export reads) and preview/<epoch>e.wav.
With --export, the last checkpoint and the prepare folder's retrieval.npy are exported to
models/voices/<voice_id> at the end.
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import pretrained  # noqa: E402
from dataset import BucketSampler, PreparedSet, collate  # noqa: E402
from rvc import commons  # noqa: E402
from rvc.discriminators import MultiPeriodDiscriminator  # noqa: E402
from rvc.losses import discriminator_loss, feature_loss, generator_loss, kl_loss  # noqa: E402
from rvc.mel_processing import mel_spectrogram_torch  # noqa: E402
from rvc.synthesizers import Synthesizer  # noqa: E402
from wavio import write_wav  # noqa: E402

# Applio's rvc/configs/40000.json, the RVC v2 40 kHz recipe (one speaker)
HPS = {
    "train": {"seed": 1234, "learning_rate": 1e-4, "betas": [0.8, 0.99], "eps": 1e-9, "lr_decay": 0.999875,
              "segment_size": 12800, "c_mel": 45, "c_kl": 1.0},
    "data": {"sample_rate": 40000, "filter_length": 2048, "hop_length": 400, "win_length": 2048,
             "n_mel_channels": 125, "mel_fmin": 0.0, "mel_fmax": None},
    "model": {"inter_channels": 192, "hidden_channels": 192, "filter_channels": 768, "text_enc_hidden_dim": 768,
              "n_heads": 2, "n_layers": 6, "kernel_size": 3, "p_dropout": 0, "resblock": "1",
              "resblock_kernel_sizes": [3, 7, 11], "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
              "upsample_rates": [10, 10, 2, 2], "upsample_initial_channel": 512,
              "upsample_kernel_sizes": [16, 16, 4, 4], "use_spectral_norm": False, "gin_channels": 256,
              "spk_embed_dim": 1},
}
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def model_config(hps):
    """The positional `config` list RVC / Applio checkpoints store (segment fixed at 32 frames, as Applio)."""
    d, m = hps["data"], hps["model"]
    return [d["filter_length"] // 2 + 1, hps["train"]["segment_size"] // d["hop_length"], m["inter_channels"],
            m["hidden_channels"], m["filter_channels"], m["n_heads"], m["n_layers"], m["kernel_size"], m["p_dropout"],
            m["resblock"], m["resblock_kernel_sizes"], m["resblock_dilation_sizes"], m["upsample_rates"],
            m["upsample_initial_channel"], m["upsample_kernel_sizes"], m["spk_embed_dim"], m["gin_channels"],
            d["sample_rate"]]


def rename_keys(d, old, new):
    """Recursively replace `old` with `new` in string keys (weight-norm naming, both directions)."""
    out = OrderedDict() if isinstance(d, OrderedDict) else {}
    for k, v in d.items():
        k = k.replace(old, new) if isinstance(k, str) else k
        out[k] = rename_keys(v, old, new) if isinstance(v, dict) else v
    return out


def to_legacy_names(d):
    """parametrizations.weight.original0/1 -> weight_g / weight_v, the names RVC checkpoints use."""
    return rename_keys(rename_keys(d, ".parametrizations.weight.original1", ".weight_v"),
                       ".parametrizations.weight.original0", ".weight_g")


def from_legacy_names(d):
    return rename_keys(rename_keys(d, ".weight_v", ".parametrizations.weight.original1"),
                       ".weight_g", ".parametrizations.weight.original0")


def save_full(net, optim, scaler, lr, epoch, path):
    """Training state as Applio's save_checkpoint writes it (resumable by either)."""
    torch.save(to_legacy_names({"model": net.state_dict(), "iteration": epoch, "optimizer": optim.state_dict(),
                                "learning_rate": lr, "scaler": scaler.state_dict()}), path)


def load_full(net, optim, path):
    ckpt = from_legacy_names(torch.load(path, map_location="cpu", weights_only=True))
    own = net.state_dict()
    net.load_state_dict({k: ckpt["model"].get(k, v) for k, v in own.items()})
    optim.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"], ckpt.get("scaler", {})


def save_small(net_g, hps, name, epoch, step, path, dataset_seconds):
    """Inference checkpoint in Applio's extract_model format: fp16 weights without the posterior encoder."""
    weights = {k: v.half() for k, v in net_g.state_dict().items() if "enc_q" not in k}
    config = model_config(hps)
    opt = OrderedDict(weight=weights)
    opt.update(config=config, epoch=epoch, step=step, sr=hps["data"]["sample_rate"], f0=True, version="v2",
               creation_date=datetime.datetime.now().isoformat(),
               model_hash=hashlib.sha256(f"{name}-{epoch}-{step}-{hps['data']['sample_rate']}-v2-{config}".encode()).hexdigest(),
               dataset_length=str(datetime.timedelta(seconds=int(dataset_seconds))), model_name=name, author=None,
               embedder_model="contentvec", speakers_id=1, vocoder="HiFi-GAN", trainer="mimic")
    torch.save(to_legacy_names(opt), path)


def latest(run_dir: Path, prefix: str):
    """Newest <prefix>_<step>.pth full checkpoint, by step."""
    found = [(int(p.stem[len(prefix) + 1:]), p) for p in run_dir.glob(f"{prefix}_*.pth") if p.stem[len(prefix) + 1:].isdigit()]
    return max(found)[1] if found else None


def export(ckpt: Path, retrieval: Path, voice: str):
    for script, source in [("export_synth.py", ckpt), ("export_synth_det.py", ckpt), ("export_retrieval.py", retrieval)]:
        print(f"export: {script} {source} {voice}", flush=True)
        subprocess.run([sys.executable, str(REPO / "tools" / "export" / script), str(source), voice], check=True)


def main():
    p = argparse.ArgumentParser(description="Fine-tune an RVC v2 40 kHz voice on a `mimic prepare` folder.")
    p.add_argument("prepare_dir", type=Path)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--save-every", type=int, default=25, help="epochs between checkpoints")
    p.add_argument("--precision", choices=DTYPES, default="fp16", help="autocast dtype (fp16 as Applio's default)")
    p.add_argument("--mute", type=int, default=2, help="copies of the silent sample in the training set")
    p.add_argument("--name", help="checkpoint name (default: the run folder's name)")
    p.add_argument("--pretrained-g", type=Path, help="default: train/pretrained/f0G40k.pth, fetched if missing")
    p.add_argument("--pretrained-d", type=Path, help="default: train/pretrained/f0D40k.pth, fetched if missing")
    p.add_argument("--seed", type=int, default=HPS["train"]["seed"])
    p.add_argument("--export", metavar="VOICE_ID", help="export the final checkpoint to models/voices/VOICE_ID")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available: training needs an NVIDIA GPU and a CUDA build of PyTorch")
    hps = json.loads(json.dumps(HPS))
    hps["train"]["seed"] = args.seed
    t_hps, d_hps = hps["train"], hps["data"]
    name = args.name or args.run_dir.name
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "preview").mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**hps, "args": {k: str(v) for k, v in vars(args).items()}}, indent=2),
                                         encoding="utf-8")
    device = torch.device("cuda")
    dtype = DTYPES[args.precision]
    use_amp = dtype != torch.float32
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        sys.exit("this GPU does not support bf16")
    # Applio's backend settings
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)

    data = PreparedSet(args.prepare_dir, d_hps, mute_copies=args.mute)
    sampler = BucketSampler(data.lengths, args.batch_size)
    steps_per_epoch = len(sampler)
    dropped = sampler.dropped()
    print(f"{len(data)} samples ({len(data) - args.mute} slices + {args.mute} mute), {len(dropped)} outside the length "
          f"buckets, {steps_per_epoch} batches of {args.batch_size} per epoch", flush=True)
    if steps_per_epoch < 3:
        sys.exit("not enough data for a batch size of %d" % args.batch_size)

    m = hps["model"]
    net_g = Synthesizer(*model_config(hps), text_enc_hidden_dim=m["text_enc_hidden_dim"]).to(device)
    net_d = MultiPeriodDiscriminator(m["use_spectral_norm"]).to(device)
    optim_g = torch.optim.AdamW(net_g.parameters(), t_hps["learning_rate"], betas=t_hps["betas"], eps=t_hps["eps"])
    optim_d = torch.optim.AdamW(net_d.parameters(), t_hps["learning_rate"], betas=t_hps["betas"], eps=t_hps["eps"])
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    g_path, d_path = latest(run_dir, "G"), latest(run_dir, "D")
    if g_path and d_path:
        epoch_d, scaler_state = load_full(net_d, optim_d, d_path)
        epoch_g, _ = load_full(net_g, optim_g, g_path)
        assert epoch_g == epoch_d, (g_path, d_path)
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        first_epoch = epoch_g + 1
        print(f"resuming after epoch {epoch_g} ({g_path.name}, {d_path.name})", flush=True)
    else:
        first_epoch = 1
        pg = args.pretrained_g or pretrained.ensure("f0G40k.pth")
        pd = args.pretrained_d or pretrained.ensure("f0D40k.pth")
        ckpt = torch.load(pg, map_location="cpu", weights_only=True)["model"]
        # one speaker: the base model's 109-row speaker table is replaced by this model's own row, as in Applio
        ckpt["emb_g.weight"] = net_g.emb_g.weight.detach().clone()
        net_g.load_state_dict(ckpt)
        net_d.load_state_dict(torch.load(pd, map_location="cpu", weights_only=True)["model"])
        print(f"starting from {pg.name} / {pd.name}", flush=True)
    global_step = (first_epoch - 1) * steps_per_epoch
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=t_hps["lr_decay"], last_epoch=first_epoch - 2)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=t_hps["lr_decay"], last_epoch=first_epoch - 2)

    hop, segment = d_hps["hop_length"], t_hps["segment_size"]
    mel_args = (d_hps["filter_length"], d_hps["n_mel_channels"], d_hps["sample_rate"], hop, d_hps["win_length"],
                d_hps["mel_fmin"], d_hps["mel_fmax"])
    # preview: the longest slice, converted back to itself through the prior
    ref_index = max(range(len(data) - args.mute), key=lambda i: data[i][0].shape[1])
    _, _, ref_phone, ref_pitch, ref_pitchf = data[ref_index]
    reference = (ref_phone.unsqueeze(0).to(device), torch.LongTensor([ref_phone.shape[0]]).to(device),
                 ref_pitch.unsqueeze(0).to(device), ref_pitchf.unsqueeze(0).to(device), torch.LongTensor([0]).to(device))
    metrics_path = run_dir / "metrics.csv"
    columns = ["epoch", "step", "lr", "loss_disc", "loss_gen", "loss_fm", "loss_mel", "loss_kl", "loss_gen_all",
               "grad_norm_d", "grad_norm_g", "seconds"]
    if not metrics_path.exists():
        metrics_path.write_text(",".join(columns) + "\n", encoding="utf-8")

    for epoch in range(first_epoch, args.epochs + 1):
        started = time.time()
        sampler.set_epoch(epoch)
        net_g.train()
        net_d.train()
        sums = {c: 0.0 for c in columns[3:-1]}
        for batch_ids in sampler:
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, _, sid = (
                t.to(device, non_blocking=True) for t in collate([data[i] for i in batch_ids]))
            with torch.autocast("cuda", enabled=use_amp, dtype=dtype):
                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                    phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                wave = commons.slice_segments(wave, ids_slice * hop, segment, dim=3)
            # discriminator step
            with torch.autocast("cuda", enabled=use_amp, dtype=dtype):
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)
            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.grad_norm(net_d.parameters())
            scaler.step(optim_d)
            # generator step
            net_d.requires_grad_(False)
            with torch.autocast("cuda", enabled=use_amp, dtype=dtype):
                _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            wave_mel = mel_spectrogram_torch(wave.float().squeeze(1), *mel_args)
            y_hat_mel = mel_spectrogram_torch(y_hat.float().squeeze(1), *mel_args)
            loss_mel = torch.nn.functional.l1_loss(wave_mel, y_hat_mel) * t_hps["c_mel"]
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * t_hps["c_kl"]
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.grad_norm(net_g.parameters())
            scaler.step(optim_g)
            scaler.update()
            net_d.requires_grad_(True)
            global_step += 1
            for key, value in [("loss_disc", loss_disc), ("loss_gen", loss_gen), ("loss_fm", loss_fm),
                               ("loss_mel", loss_mel), ("loss_kl", loss_kl), ("loss_gen_all", loss_gen_all)]:
                sums[key] += float(value.detach())
            sums["grad_norm_d"] += grad_norm_d
            sums["grad_norm_g"] += grad_norm_g

        lr = optim_g.param_groups[0]["lr"]
        means = {k: v / steps_per_epoch for k, v in sums.items()}
        seconds = time.time() - started
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(",".join([str(epoch), str(global_step), f"{lr:.6e}"] + [f"{means[c]:.5f}" for c in columns[3:-1]]
                             + [f"{seconds:.1f}"]) + "\n")
        print(f"{name} | epoch {epoch}/{args.epochs} | step {global_step} | d {means['loss_disc']:.3f} "
              f"adv {means['loss_gen']:.3f} fm {means['loss_fm']:.3f} mel {means['loss_mel']:.3f} "
              f"kl {means['loss_kl']:.3f} total {means['loss_gen_all']:.3f} | lr {lr:.3e} | {seconds:.1f} s", flush=True)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            old = [latest(run_dir, "G"), latest(run_dir, "D")]
            save_full(net_g, optim_g, scaler, t_hps["learning_rate"], epoch, run_dir / f"G_{global_step}.pth")
            save_full(net_d, optim_d, scaler, t_hps["learning_rate"], epoch, run_dir / f"D_{global_step}.pth")
            for o in old:  # keep only the newest full state: it is 1.3 GB a pair
                if o is not None and o.stem.split("_")[1] != str(global_step):
                    o.unlink()
            small = run_dir / f"{name}_{epoch}e_{global_step}s.pth"
            save_small(net_g, hps, name, epoch, global_step, small, data.manifest["sliced_seconds"])
            net_g.eval()
            with torch.no_grad():
                audio = net_g.infer(*reference)[0, 0].float().cpu().numpy()
            write_wav(run_dir / "preview" / f"{epoch:04d}e.wav", audio, d_hps["sample_rate"])
            print(f"saved {small.name}", flush=True)

        sched_g.step()
        sched_d.step()

    final = run_dir / f"{name}_{args.epochs}e_{global_step}s.pth"
    if args.export:
        export(final, args.prepare_dir / "retrieval.npy", args.export)


if __name__ == "__main__":
    main()
