# -*- coding: utf-8 -*-
"""Quantum Variational Autoencoder (QVAE) Model"""

import torch
import numpy as np

from .qvae_dist_util import MixtureGeneric, FactorialBernoulliUtil
from .abstract_boltzmann_machine import AbstractBoltzmannMachine


class QVAE_BM(torch.nn.Module):
    """Quantum Variational Autoencoder (QVAE) Model

    Args:
        encoder: Encoder module
        decoder: Decoder module
        bm (AbstractBoltzmannMachine): Boltzmann machine
        sampler: Sampler
        dist_beta: Beta parameter for the distribution
        mean_x (torch.Tensor): Bias of training data
        num_vis (int): Number of visible variables in the Boltzmann machine
    """

    def __init__(
        self,
        encoder,
        decoder,
        bm: AbstractBoltzmannMachine,
        sampler,
        dist_beta,
        mean_x: float,
        num_vis: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.bm = bm
        self.sampler = sampler
        self.dist_beta = dist_beta
        # Convert train_bias to PyTorch tensor
        self.register_buffer(
            "train_bias",
            torch.tensor(
                -np.log(1.0 / np.clip(mean_x, 0.001, 0.999) - 1.0).astype(np.float32)
            ),
        )
        self.is_training = True
        self.num_var1 = num_vis

    def posterior(self, q_logits, beta):
        """Compute posterior distribution and its reparameterized sample

        Args:
            q_logits (torch.Tensor): Encoder output, log-odds
            beta: Beta parameter for the distribution

        Returns:
            tuple: (Posterior distribution object, Sampled result zeta)
        """
        posterior_dist = MixtureGeneric(q_logits, beta)
        zeta = posterior_dist.reparameterize(self.is_training)
        return posterior_dist, zeta

    def _cross_entropy(
        self, logit_q: torch.Tensor, log_ratio: torch.Tensor
    ) -> torch.Tensor:
        """Compute the cross-entropy term for the overlap distribution proposed in DVAE++
        [MODIFIED FOR FULL BOLTZMANN MACHINE]
        """
        # 1. (同前) 检查维度
        if self.bm.num_nodes != logit_q.shape[1]:
            raise ValueError(
                f"The number of variables in the Boltzmann machine {self.bm.num_nodes}"
                f" does not match the shape of logit_q {logit_q.shape[1]}."
            )
        
        # 2. (同前) 分割 logit_q
        logit_q1 = logit_q[:, : self.num_var1]
        logit_q2 = logit_q[:, self.num_var1 :]

        # 3. (同前) 计算基础概率
        q1 = torch.sigmoid(logit_q1)
        q2 = torch.sigmoid(logit_q2)
        log_ratio1 = log_ratio[:, : self.num_var1]
        # q1_pert 是 DVAE++ 技巧的一部分，用于计算梯度
        q1_pert = torch.sigmoid(logit_q1 + log_ratio1)

        # 4. [修改] 计算线性和二次型期望

        # 线性项 E[s^T h]
        # (这部分同前，使用原始的 q1, q2)
        q_all_original = torch.cat([q1, q2], dim=-1)
        linear_term = -torch.matmul(q_all_original, self.bm.linear_bias)

        # 二次项 E[s^T J s]
        # 我们使用 q1_pert 和 q2 来组合完整的概率向量
        # q_all_pert 形状为 (Batch, num_nodes)
        q_all_pert = torch.cat([q1_pert, q2], dim=-1)

        # self.bm.quadratic_coef 现在是 (num_nodes, num_nodes)
        # 计算完整的二次型 s^T J s
        tmp = q_all_pert.matmul(self.bm.quadratic_coef) # 形状 (B, num_nodes)
        # (B, num_nodes) * (B, num_nodes) -> 逐元素乘法后求和
        quadratic_term = -torch.sum(tmp * q_all_pert, dim=1) # 形状 (B,)

        # 5. (同前) 组合并减去负样本能量
        cross_entropy = linear_term + quadratic_term
        
        s_neg = self.bm.sample(self.sampler)
        cross_entropy = cross_entropy - self.bm(s_neg).mean()
        return cross_entropy

    def _kl_dist_from(self, posterior, post_samples):
        """Compute KL divergence

        Args:
            posterior: Posterior distribution object
            post_samples: Posterior distribution samples

        Returns:
            torch.Tensor: KL divergence tensor
        """
        entropy = 0
        logit_q = 0
        log_ratio = 0
        entropy += torch.sum(posterior.entropy(), dim=1)

        logit_q = posterior.logit_mu
        log_ratio = posterior.log_ratio(post_samples)
        cross_entropy = self._cross_entropy(logit_q, log_ratio)
        kl = cross_entropy - entropy

        return kl

    def neg_elbo(self, x, kl_beta):
        """Compute negative ELBO loss

        Args:
            x (torch.Tensor): Input data
            kl_beta (float): Weight coefficient for KL term

        Returns:
            tuple: (output, recon_x, neg_elbo, wd_loss, total_kl, cost, q, zeta)
                output: Reconstructed output (sigmoid activated)
                recon_x: Reconstructed data
                neg_elbo: Negative ELBO loss
                wd_loss: Weight decay loss
                total_kl: KL divergence
                cost: Reconstruction loss
                q: Encoder output
                zeta: Posterior sample
        """
        # Subtract mean from input
        encoder_x = x - self.train_bias
        recon_x, posterior, q, zeta = self(encoder_x)

        # Add data bias
        recon_x = recon_x + self.train_bias

        output_dist = FactorialBernoulliUtil(recon_x)

        # Apply sigmoid
        output = torch.sigmoid(output_dist.logit_mu)

        # Compute KL
        total_kl = self._kl_dist_from(posterior, zeta)
        total_kl = torch.mean(total_kl)
        # Expected log prob p(x| z)
        cost = -output_dist.log_prob_per_var(x)  # [256, 784]
        cost = torch.sum(cost, dim=1)  # [256], reconstruction loss per sample
        cost = torch.mean(cost)

        # Compute negative ELBO per sample, then average
        neg_elbo = total_kl * kl_beta + cost  # scalar

        # Weight decay loss
        w_weight_decay = 0.01 * torch.sum(self.bm.quadratic_coef**2)
        b_weight_decay = 0.005 * torch.sum(self.bm.linear_bias**2)
        wd_loss = w_weight_decay + b_weight_decay

        return output, recon_x, neg_elbo, wd_loss, total_kl, cost, q, zeta

    def forward(self, x):
        """Forward propagation

        Args:
            x (torch.Tensor): Input data

        Returns:
            tuple: (recon_x, posterior, q, zeta)
                recon_x: Reconstructed data
                posterior: Posterior distribution object
                q: Encoder output
                zeta: Posterior sample
        """
        q = self.encoder(x)
        posterior, zeta = self.posterior(q, self.dist_beta)
        recon_x = self.decoder(zeta)

        return recon_x, posterior, q, zeta
