# Third-party notices and reviewed runtime boundary

- SpeechMOS v1.2.0: https://github.com/tarepan/SpeechMOS/tree/v1.2.0
  Copyright 2023 tarepan, MIT (`vendor/LICENSE_SpeechMOS`).
  The authors state in `hubconf.py` that the converted strong checkpoint derives
  from UTMOS22's MIT checkpoint. Original UTMOS22 copyright 2022
  Saruwatari&Koyama laboratory, The University of Tokyo (`LICENSE_UTMOS22`).
- `fairseq_alt.py` derives from Fairseq, copyright Facebook, Inc. and affiliates,
  MIT (`vendor/LICENSE_fairseq`).
- DNS-Challenge inference code: Microsoft Corporation, MIT
  (`vendor/LICENSE_DNSMOS_code`). The repository's separate root LICENSE is
  CC BY 4.0. We conservatively retain that license for the downloaded ONNX weight;
  no separate more permissive weight license was found. The downloader saves its
  exact license alongside the ignored weight. Source and polynomial calibration:
  https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS

Fully reviewed SpeechMOS runtime source:

- `vendor/fairseq_alt.py` SHA256
  `a554d47bebef0eb764a048d58dbb7e1e1fca316d627c571add54faa42fe5f45b`
- `vendor/utmos_model.py` SHA256
  `bffb8eba14e74e4f228684280cbd295ecffbfcd0414a5a460ca55c94f2f0a0e4`

These contain tensor-model definitions and torch/torchaudio calls, no network,
subprocess launch, installation or file writes. Runtime verifies their hashes,
loads the self-contained fairseq module and extracts only the reviewed UTMOS class.
The downloaded `hubconf_reference.py` and `dnsmos_reference.py` are provenance/
review copies and are never imported or executed by the scorer. There is no
`torch.hub.load`, `trust_repo`, remote-code loader or pickle fallback.

Runtime also verifies both fixed weight SHA256 values (see `score.py` and the
report). UTMOS uses `torch.load(weights_only=True)` and `load_state_dict(strict=True)`.
The downloader only transfers bounded files, records URL/size/hash, and does not
execute downloaded content. Weight/source licenses do not grant rights to the
user's recordings, private speaker embeddings or game dialogue.
