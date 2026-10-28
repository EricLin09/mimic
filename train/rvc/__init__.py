"""RVC v2 network definitions: the f0 synthesizer with the NSF HiFi-GAN vocoder (768-dim content
features) and, for training, its multi-period discriminator, losses and spectrograms.

Taken from Applio's rvc/lib/algorithm and rvc/train and trimmed to that one configuration: no v1 /
no-f0 paths, no other vocoders, no UI glue. Module and parameter names are unchanged, so Applio / RVC
checkpoints load as they are. MIT licensed, see LICENSE in this directory.
"""
