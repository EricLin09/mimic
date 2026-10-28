"""GAN, feature-matching and KL losses (from Applio rvc/train/losses.py)."""
import torch


def feature_loss(fmap_r, fmap_g):
    """L1 between real and generated discriminator feature maps, summed over every layer, times 2."""
    return 2 * sum(torch.mean(torch.abs(rl - gl)) for dr, dg in zip(fmap_r, fmap_g) for rl, gl in zip(dr, dg))


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Least-squares GAN loss for the discriminator: real -> 1, generated -> 0."""
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr.float()) ** 2)
        g_loss = torch.mean(dg.float() ** 2)
        loss += r_loss + g_loss
    return loss


def generator_loss(disc_outputs):
    """Least-squares GAN loss for the generator: generated -> 1."""
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean((1 - dg.float()) ** 2)
    return loss


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """KL divergence between the posterior (flowed to z_p, log-std logs_q) and the prior (m_p, logs_p),
    averaged over the valid frames.

    Args:
        z_p: posterior latent after the flow [b, h, t].
        logs_q: posterior log-std [b, h, t].
        m_p: prior mean [b, h, t].
        logs_p: prior log-std [b, h, t].
        z_mask: valid-frame mask [b, 1, t].
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
    kl = (kl * z_mask).sum()
    loss = kl / z_mask.sum()
    return loss
