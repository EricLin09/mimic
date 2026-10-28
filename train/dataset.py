"""Training samples from a `mimic prepare` folder, batched the way Applio batches them.

PreparedSet does what Applio's TextAudioLoaderMultiNSFsid does per item, once up front (the data is
small); BucketSampler and collate are Applio's DistributedBucketSampler (one replica) and
TextAudioCollateMultiNSFsid, from rvc/train/data_utils.py (MIT, see rvc/LICENSE).
"""
import json
from pathlib import Path

import numpy as np
import torch

from rvc.mel_processing import spectrogram_torch
from wavio import read_wav

F0_BIN = 256
F0_MEL_MIN = 1127 * np.log(1 + 50.0 / 700)
F0_MEL_MAX = 1127 * np.log(1 + 1100.0 / 700)
# length buckets, in Applio's file-size units (see applio_length)
BOUNDARIES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
MAX_FRAMES = 900


def coarse_f0(f0):
    """RVC's 255-step mel-scale pitch for the content encoder's pitch embedding (1 = unvoiced), as
    Applio's training extraction computes it (rvc/train/extract/extract.py, coarse_f0)."""
    f0_mel = 1127.0 * np.log(1.0 + np.asarray(f0, np.float64) / 700.0)
    f0_mel = np.clip((f0_mel - F0_MEL_MIN) * (F0_BIN - 2) / (F0_MEL_MAX - F0_MEL_MIN) + 1, 1, F0_BIN - 1)
    return np.rint(f0_mel).astype(np.int64)


def applio_length(samples: int, hop: int, mute: bool) -> int:
    """Applio buckets a sample by its WAV file size // (3 * hop). Its slices are float32 WAVs with a
    58-byte header and its mute sample a 16-bit WAV with a 78-byte header; computing the same number
    here gives the same buckets, so the same items are dropped (under ~0.38 s at 40 kHz) and the epochs
    have the same length."""
    size = 78 + 2 * samples if mute else 58 + 4 * samples
    return size // (3 * hop)


class PreparedSet:
    """Every slice of a prepare folder plus `mute_copies` of its silent sample, each as
    (spec [bins, T], wave [1, T * hop], phone [T, 768], pitch [T], pitchf [T]), where T is the shorter of
    the feature frames (x2 to 100 fps, at most 900) and the spectrogram frames."""

    def __init__(self, folder, data_hps, mute_copies: int = 2):
        folder = Path(folder)
        manifest = json.loads((folder / "prepare.json").read_text(encoding="utf-8"))
        if manifest["sample_rate"] != data_hps["sample_rate"]:
            raise ValueError(f"{folder}: prepared at {manifest['sample_rate']} Hz, training at {data_hps['sample_rate']} Hz")
        self.manifest = manifest
        self.hop = data_hps["hop_length"]
        names = [(it["name"], False) for it in manifest["items"]] + [(manifest["mute"], True)] * mute_copies
        self.items, self.lengths, self.names = [], [], []
        cache = {}
        for name, mute in names:
            if name not in cache:
                cache[name] = self._load(folder, name, data_hps)
            item, samples = cache[name]
            self.items.append(item)
            self.lengths.append(applio_length(samples, self.hop, mute))
            self.names.append(name)

    def _load(self, folder, name, hps):
        """(item, samples in the WAV before trimming)."""
        wave, rate = read_wav(folder / "audio" / f"{name}.wav")
        assert rate == hps["sample_rate"], (name, rate)
        samples = len(wave)
        phone = np.repeat(np.load(folder / "feats" / f"{name}.npy"), 2, axis=0)
        pitchf = np.load(folder / "f0" / f"{name}.npy").astype(np.float64)
        pitch = coarse_f0(pitchf)
        n = min(phone.shape[0], MAX_FRAMES)
        phone, pitch, pitchf = phone[:n], pitch[:n], pitchf[:n]
        wave = torch.from_numpy(wave).unsqueeze(0)
        spec = spectrogram_torch(wave, hps["filter_length"], hps["hop_length"], hps["win_length"], center=False).squeeze(0)
        if phone.shape[0] != spec.shape[-1]:
            t = min(phone.shape[0], spec.shape[-1])
            spec, wave = spec[:, :t], wave[:, : t * self.hop]
            phone, pitch, pitchf = phone[:t], pitch[:t], pitchf[:t]
        return (spec, wave, torch.FloatTensor(phone), torch.LongTensor(pitch), torch.FloatTensor(pitchf)), samples

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    """Pads a batch, longest first: (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave,
    wave_lengths, sid); the sid is always 0, one speaker."""
    _, order = torch.sort(torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True)
    n = len(batch)
    max_spec = max(x[0].size(1) for x in batch)
    max_wave = max(x[1].size(1) for x in batch)
    max_phone = max(x[2].size(0) for x in batch)
    spec_padded = torch.zeros(n, batch[0][0].size(0), max_spec)
    wave_padded = torch.zeros(n, 1, max_wave)
    phone_padded = torch.zeros(n, max_phone, batch[0][2].shape[1])
    pitch_padded = torch.zeros(n, max_phone, dtype=torch.long)
    pitchf_padded = torch.zeros(n, max_phone)
    spec_lengths, wave_lengths, phone_lengths = (torch.LongTensor(n) for _ in range(3))
    for i, k in enumerate(order.tolist()):
        spec, wave, phone, pitch, pitchf = batch[k]
        spec_padded[i, :, : spec.size(1)] = spec
        spec_lengths[i] = spec.size(1)
        wave_padded[i, :, : wave.size(1)] = wave
        wave_lengths[i] = wave.size(1)
        phone_padded[i, : phone.size(0), :] = phone
        phone_lengths[i] = phone.size(0)
        pitch_padded[i, : pitch.size(0)] = pitch
        pitchf_padded[i, : pitchf.size(0)] = pitchf
    sid = torch.zeros(n, dtype=torch.long)
    return phone_padded, phone_lengths, pitch_padded, pitchf_padded, spec_padded, spec_lengths, wave_padded, wave_lengths, sid


class BucketSampler:
    """Batches of similar length. Every bucket is padded with repeats to a whole number of batches,
    then the batches are shuffled; set_epoch seeds the shuffle, as in Applio."""

    def __init__(self, lengths, batch_size, boundaries=BOUNDARIES, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.boundaries = list(boundaries)
        self.shuffle = shuffle
        self.epoch = 0
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.num_samples = sum(self.num_samples_per_bucket)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i, length in enumerate(self.lengths):
            b = self._bisect(length)
            if b != -1:
                buckets[b].append(i)
        for i in range(len(buckets) - 1, -1, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)
        per_bucket = [len(b) + (self.batch_size - len(b) % self.batch_size) % self.batch_size for b in buckets]
        return buckets, per_bucket

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1
        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        return -1

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = [torch.randperm(len(b), generator=g).tolist() for b in self.buckets]
        else:
            indices = [list(range(len(b))) for b in self.buckets]
        batches = []
        for bucket, ids, total in zip(self.buckets, indices, self.num_samples_per_bucket):
            rem = total - len(bucket)
            ids = ids + ids * (rem // len(bucket)) + ids[: rem % len(bucket)]
            for j in range(len(ids) // self.batch_size):
                batches.append([bucket[k] for k in ids[j * self.batch_size : (j + 1) * self.batch_size]])
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]
        assert len(batches) * self.batch_size == self.num_samples
        return iter(batches)

    def __len__(self):
        return self.num_samples // self.batch_size

    def dropped(self):
        """Items outside every bucket (too short or too long), which no batch ever contains."""
        kept = {i for b in self.buckets for i in b}
        return [i for i in range(len(self.lengths)) if i not in kept]
