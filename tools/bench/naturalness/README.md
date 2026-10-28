# Offline naturalness proxies

Run with a research Python containing torch, torchaudio, numpy, soundfile and
onnxruntime. No implicit model download, torch.hub call, GPU, or package install.

1. `python -I -B tools/bench/naturalness/download.py` downloads about 411 MB of
   UTMOS22 weights and 1.16 MB of DNSMOS weights plus small reviewed source/license
   files. Downloads are data only; the downloader never executes them.
2. `python -I -B tools/bench/naturalness/score.py candidate.wav --reference source.wav`
   scores full 2.5 s source-time windows after global envelope alignment. With a
   reference, source RMS below -45 dBFS is excluded for every candidate equally;
   without a reference, all complete windows are scored.
3. `--models utmos` or `--models dnsmos` selects a subset. `--segment-s` must be >=1.
   Short inputs with no full window return an empty summary; choose a smaller valid
   window explicitly. `--json result.json` saves per-segment values and statistics.

UTMOS is a naturalness proxy, not speaker similarity. DNSMOS is a noise/quality
proxy: each short segment is repeated to its official 9.01 s input, so these are
repeat-proxy scores, not a validated native 2.5 s metric. Both use 16 kHz and cannot
assess frequencies above 8 kHz. Do not replace listening tests or optimize a single
predictor blindly.

`weights/`, `cache/` and logs are ignored. No private audio belongs in this folder.
Source audit and license details are in `THIRD_PARTY.md` and `vendor/LICENSE_*`.
