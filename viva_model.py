import torch
import torch.nn as nn
import logging
from typing import Dict, Optional
from pathlib import Path

from wan_model import WanVideoModel
from wan.utils.fm import FlowMatchScheduler
from wan.modules.model import sinusoidal_embedding_1d
from viva_utils import (
    replace_latent_with_state, replace_latent_with_value,
    extract_value_from_latent, extract_state_from_latent,
    get_condition_mask, apply_condition_mask,
    BLANK_IDX, STATE_IDX, CAM_LEFT_WRIST_IDX,
    CAM_RIGHT_WRIST_IDX, CAM_HIGH_IDX,
    FUTURE_STATE_IDX, VALUE_IDX,
    FUTURE_IMAGE_INDICES, TARGET_INDICES, NUM_LATENT_FRAMES,
)

logger = logging.getLogger(__name__)


class VivaModel(nn.Module):
    """
    7-frame latent sequence:
    [blank, state, cam_left, cam_right, cam_high, future_state, value]

    Condition mask: [1, 1, 1, 1, 1, 0, 0]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dtype = torch.bfloat16

        self.video_model = WanVideoModel.from_pretrained(
            checkpoint_path=config.model.wan.checkpoint_path,
            vae_path=config.model.wan.vae_path,
            config_path=config.model.wan.config_path,
            precision=config.model.wan.precision
        )

        logger.info("VivaModel initialized")
        self.device = next(self.video_model.parameters()).device

        self.num_latent_frames = NUM_LATENT_FRAMES  # 7
        self.latent_h = config.common.video_height // 16
        self.latent_w = config.common.video_width // 16
        self.latent_c = 48

        self.loss_weight_future_state = getattr(
            config.training, 'loss_weight_future_state', 1.0)
        self.loss_weight_value = getattr(
            config.training, 'loss_weight_value', 1.0)
        if not hasattr(config.training, 'loss_weight_value') and hasattr(config.training, 'loss_weight_reward'):
            self.loss_weight_value = getattr(config.training, 'loss_weight_reward')

        logger.info(f"  Latent: [{self.latent_c}, {self.num_latent_frames}, "
                     f"{self.latent_h}, {self.latent_w}]")
        logger.info(f"  Loss weights: fut_state={self.loss_weight_future_state}"
                     f", value={self.loss_weight_value}")

        self.fm_scheduler = FlowMatchScheduler(
            shift=5.0, sigma_min=0.0,
            extra_one_step=True, num_train_timesteps=1000
        )
        self.fm_scheduler.set_timesteps(num_inference_steps=1000, training=True)

        total_p = sum(p.numel() for p in self.parameters())
        train_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"  Params: {total_p/1e9:.2f}B total, {train_p/1e9:.2f}B trainable")

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(dtype=self.dtype)
        images_normalized = images * 2.0 - 1.0
        images_normalized = images_normalized.unsqueeze(2)
        with torch.no_grad():
            latents = self.video_model.encode_video(images_normalized)
        return latents.to(dtype=self.dtype)

    def _inject_state_to_latent(self, latent, state):
        B = latent.shape[0]
        C, H, W = latent.shape[1], latent.shape[3], latent.shape[4]
        state = state.to(dtype=self.dtype)
        flat = state.flatten(start_dim=1)
        sd = flat.shape[1]
        le = C * H * W
        nr = (le + sd - 1) // sd
        repeated = flat.repeat(1, nr)[:, :le]
        latent[:, :, 0, :, :] = repeated.reshape(B, C, H, W)
        return latent

    def _inject_value_to_latent(self, latent, value):
        B = latent.shape[0]
        C, H, W = latent.shape[1], latent.shape[3], latent.shape[4]
        value = value.to(dtype=self.dtype)
        if value.dim() == 1:
            value = value.unsqueeze(1)
        latent[:, :, 0, :, :] = value.reshape(B, 1, 1, 1).expand(B, C, H, W)
        return latent

    def build_latent_sequence(
        self, state, cam_left_wrist, cam_right_wrist, cam_high,
        future_state, value_normalized,
    ) -> torch.Tensor:
        """Build 7-frame latent sequence [B, 48, 7, H', W']."""
        B = state.shape[0]
        device = state.device

        cam_left_lat = self.encode_images(cam_left_wrist)
        cam_right_lat = self.encode_images(cam_right_wrist)
        cam_high_lat = self.encode_images(cam_high)

        def _zeros():
            return torch.zeros(B, self.latent_c, 1, self.latent_h,
                               self.latent_w, dtype=self.dtype, device=device)

        blank_lat = _zeros()
        state_lat = self._inject_state_to_latent(_zeros(), state)
        fut_state_lat = self._inject_state_to_latent(_zeros(), future_state)
        value_lat = self._inject_value_to_latent(_zeros(), value_normalized)

        return torch.cat([
            blank_lat,                                # 0: blank
            state_lat,                                # 1: current state
            cam_left_lat.to(dtype=self.dtype),         # 2: curr cam_left
            cam_right_lat.to(dtype=self.dtype),        # 3: curr cam_right
            cam_high_lat.to(dtype=self.dtype),         # 4: curr cam_high
            fut_state_lat,                             # 5: future state
            value_lat,                                # 6: value
        ], dim=2)

    def get_time_embedding(self, timesteps, seq_len):
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1).expand(timesteps.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = timesteps.size(0)
            t_flat = timesteps.flatten()
            freq_dim = self.video_model.wan_model.freq_dim
            dim = 3072
            t_emb = self.video_model.wan_model.time_embedding(
                sinusoidal_embedding_1d(freq_dim, t_flat)
                .unflatten(0, (bt, seq_len)).float()
            )
            t_proj = self.video_model.wan_model.time_projection(t_emb) \
                .unflatten(2, (6, dim))
        return t_emb, t_proj

    def forward_denoise(self, noisy_latent, timesteps, t5_embeddings,
                        condition_mask, clean_latent):
        model_input = apply_condition_mask(
            clean_latent, noisy_latent, condition_mask).to(dtype=self.dtype)

        patched = self.video_model.wan_model.patch_embedding(model_input)
        video_tokens = patched.flatten(2).transpose(1, 2)
        seq_len = video_tokens.shape[1]
        B = patched.shape[0]

        _, _, T_p, H_p, W_p = patched.shape
        grid_sizes = torch.tensor(
            [T_p, H_p, W_p], dtype=torch.long, device=patched.device
        ).unsqueeze(0).expand(B, -1)

        time_emb, time_proj = self.get_time_embedding(timesteps, seq_len)

        text_len = self.video_model.wan_model.text_len
        if t5_embeddings.shape[1] < text_len:
            pad = t5_embeddings.new_zeros(
                t5_embeddings.shape[0],
                text_len - t5_embeddings.shape[1],
                t5_embeddings.shape[2])
            t5_embeddings = torch.cat([t5_embeddings, pad], dim=1)
        elif t5_embeddings.shape[1] > text_len:
            t5_embeddings = t5_embeddings[:, :text_len]

        t5_embeddings = t5_embeddings.to(dtype=self.dtype)
        context = self.video_model.wan_model.text_embedding(t5_embeddings)

        freqs = self.video_model.wan_model.freqs
        if freqs.device != video_tokens.device:
            freqs = freqs.to(video_tokens.device)
        seq_lens = torch.full(
            (B,), seq_len, dtype=torch.long, device=video_tokens.device)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            for block in self.video_model.wan_model.blocks:
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    modulation = (
                        block.modulation.unsqueeze(0) + time_proj
                    ).chunk(6, dim=2)
                norm_x = (block.norm1(video_tokens).float()
                          * (1 + modulation[1].squeeze(2))
                          + modulation[0].squeeze(2))
                attn_out = block.self_attn(norm_x, seq_lens, grid_sizes, freqs)
                video_tokens = video_tokens + attn_out * modulation[2].squeeze(2)

                cross_out = block.cross_attn(
                    block.norm3(video_tokens), context, None)
                video_tokens = video_tokens + cross_out

                ffn_in = (block.norm2(video_tokens).float()
                          * (1 + modulation[4].squeeze(2))
                          + modulation[3].squeeze(2))
                ffn_out = block.ffn(ffn_in)
                video_tokens = video_tokens + ffn_out * modulation[5].squeeze(2)

            pred = self.video_model.wan_model.head(video_tokens, time_emb)
            pred = self.video_model.wan_model.unpatchify(pred, grid_sizes)
            pred = torch.stack([u for u in pred], dim=0)

        return pred.to(dtype=self.dtype)

    def training_step(
        self, state, cam_left_wrist, cam_right_wrist, cam_high,
        future_state, value_normalized=None, t5_embeddings=None, return_dict=True,
    ) -> Dict[str, torch.Tensor]:
        B = state.shape[0]
        device = state.device
        if B == 0:
            return self._dummy_output(device)

        if value_normalized is None:
            raise ValueError("value_normalized must be provided")

        clean_latent = self.build_latent_sequence(
            state, cam_left_wrist, cam_right_wrist, cam_high,
            future_state, value_normalized)

        condition_mask = get_condition_mask(B, device=device, dtype=self.dtype)

        timestep_ids = torch.randint(
            0, self.fm_scheduler.num_train_timesteps, (B,), device='cpu')
        timesteps = self.fm_scheduler.timesteps[timestep_ids].to(
            dtype=self.dtype, device=device)
        sigma = self.fm_scheduler.sigmas[timestep_ids].to(
            dtype=self.dtype, device=device).view(B, 1, 1, 1, 1)

        noise = torch.randn_like(clean_latent, dtype=self.dtype)
        noisy_latent = clean_latent * (1 - sigma) + noise * sigma
        noisy_latent = apply_condition_mask(
            clean_latent, noisy_latent, condition_mask)

        target = (noise - clean_latent) * (1 - condition_mask)

        pred = self.forward_denoise(
            noisy_latent, timesteps, t5_embeddings,
            condition_mask, clean_latent)
        pred_masked = pred * (1 - condition_mask)

        fs = nn.functional.mse_loss(
            pred_masked[:, :, FUTURE_STATE_IDX].float(),
            target[:, :, FUTURE_STATE_IDX].float())
        value_loss = nn.functional.mse_loss(
            pred_masked[:, :, VALUE_IDX].float(),
            target[:, :, VALUE_IDX].float())

        total_loss = (self.loss_weight_future_state * fs
                      + self.loss_weight_value * value_loss)

        if return_dict:
            return {
                'total_loss': total_loss,
                'future_state_loss': fs,
                'value_loss': value_loss,
                'sigma_mean': sigma.mean().item(),
            }
        return total_loss

    def _dummy_output(self, device):
        z = lambda: torch.zeros([], device=device, requires_grad=True)
        return {
            'total_loss': z(), 'future_state_loss': z(),
            'value_loss': z(), 'sigma_mean': 0.0,
        }

    @torch.no_grad()
    def predict_value(
        self, state, cam_left_wrist, cam_right_wrist, cam_high,
        t5_embeddings, num_inference_steps=50, num_samples=1,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict future state + value via iterative denoising.

        num_samples>1 runs that many independent denoising rollouts (different
        noise seeds) per input and reports the spread of the resulting values as
        `value_ensemble_std` — a generative-entropy / confidence proxy in the
        spirit of Diffusion Reward's entropy reward. Lower spread = the conditional
        generative distribution is more peaked = the model is more certain.

        Returns:
          'value':              [B] in [0, 1]  (ensemble mean when num_samples>1)
          'value_std':          [B]  intra-sample latent-field dispersion
          'value_ensemble_std': [B]  std across the K samples (0 when num_samples==1)
          'future_state':       [B, 14]  (denormalized to [-1, 1])
        """
        B0 = state.shape[0]
        device = state.device

        if num_samples > 1:
            state = state.repeat_interleave(num_samples, dim=0)
            cam_left_wrist = cam_left_wrist.repeat_interleave(num_samples, dim=0)
            cam_right_wrist = cam_right_wrist.repeat_interleave(num_samples, dim=0)
            cam_high = cam_high.repeat_interleave(num_samples, dim=0)
            t5_embeddings = t5_embeddings.repeat_interleave(num_samples, dim=0)
        B = state.shape[0]

        dummy_value = torch.zeros(B, device=device)
        dummy_state = torch.zeros_like(state)

        clean_latent = self.build_latent_sequence(
            state, cam_left_wrist, cam_right_wrist, cam_high,
            dummy_state, dummy_value)

        condition_mask = get_condition_mask(B, device=device, dtype=self.dtype)

        latent = clean_latent.clone()
        for idx in TARGET_INDICES:
            latent[:, :, idx:idx+1] = torch.randn_like(latent[:, :, idx:idx+1])

        self.fm_scheduler.set_timesteps(num_inference_steps)
        for i, t in enumerate(self.fm_scheduler.timesteps):
            ts = torch.full((B,), t, device=device, dtype=self.dtype)
            pred = self.forward_denoise(
                latent, ts, t5_embeddings, condition_mask, clean_latent)
            for idx in TARGET_INDICES:
                latent[:, :, idx] = self.fm_scheduler.step(
                    pred[:, :, idx], t, latent[:, :, idx])

        value_frame = latent[:, :, VALUE_IDX, :, :]      # [B, 48, H', W']
        value_norm = value_frame.mean(dim=(1, 2, 3))
        value_std = value_frame.std(dim=(1, 2, 3))        # latent-space dispersion = confidence proxy
        value = torch.clamp((value_norm + 1) / 2, 0, 1)
        predicted_future_state = extract_state_from_latent(
            latent, state_dim=14, state_index=FUTURE_STATE_IDX)

        if num_samples > 1:
            value = value.view(B0, num_samples)
            value_std = value_std.view(B0, num_samples)
            predicted_future_state = predicted_future_state.view(B0, num_samples, -1)
            # generative-entropy proxies: spread across the K rollouts
            value_ensemble_std = value.std(dim=1)                              # [B0]
            fs_ensemble_std = predicted_future_state.std(dim=1).mean(dim=1)    # [B0] mean over 14 dims
            return {
                'value': value.mean(dim=1),
                'value_samples': value,
                'value_std': value_std.mean(dim=1),
                'value_ensemble_std': value_ensemble_std,
                'future_state_ensemble_std': fs_ensemble_std,
                'future_state': predicted_future_state.mean(dim=1),
            }

        return {
            'value': value,
            'value_std': value_std,
            'value_ensemble_std': torch.zeros_like(value),
            'future_state_ensemble_std': torch.zeros_like(value),
            'future_state': predicted_future_state,
        }

    @torch.no_grad()
    def estimate_generative_entropy(
        self, state, cam_left_wrist, cam_right_wrist, cam_high,
        t5_embeddings, num_samples=6, num_score_levels=6, num_inference_steps=1,
        num_noise_draws=1,
    ) -> Dict[str, torch.Tensor]:
        """Diffusion-Reward-style entropy for the value / future_state targets.

        1) Generate K self-samples of the target frames by denoising from noise.
        2) Score each sample's negative-log-likelihood with the flow-matching
           denoising residual ||v_pred - (noise - x_hat)||^2 integrated over M
           noise levels (the continuous-FM analog of Diffusion Reward accumulating
           the VLB/KL across the diffusion trajectory on self-generated samples).
        Averaging that NLL over the K samples estimates the conditional entropy
        H = -E_{x~p}[log p(x|obs)] (higher = the model is less certain).

        Returns: 'value' (ensemble mean), 'value_entropy', 'fs_entropy'.
        """
        B0 = state.shape[0]
        device = state.device
        K = num_samples
        rep = lambda x: x.repeat_interleave(K, dim=0)
        s, cl, cr, ch, t5 = (rep(state), rep(cam_left_wrist), rep(cam_right_wrist),
                             rep(cam_high), rep(t5_embeddings))
        B = s.shape[0]

        clean = self.build_latent_sequence(
            s, cl, cr, ch, torch.zeros_like(s), torch.zeros(B, device=device))
        cmask = get_condition_mask(B, device=device, dtype=self.dtype)

        # 1) generate K self-samples of the target frames
        latent = clean.clone()
        for idx in TARGET_INDICES:
            latent[:, :, idx:idx+1] = torch.randn_like(latent[:, :, idx:idx+1])
        self.fm_scheduler.set_timesteps(num_inference_steps)
        for t in self.fm_scheduler.timesteps:
            ts = torch.full((B,), t, device=device, dtype=self.dtype)
            pred = self.forward_denoise(latent, ts, t5, cmask, clean)
            for idx in TARGET_INDICES:
                latent[:, :, idx] = self.fm_scheduler.step(pred[:, :, idx], t, latent[:, :, idx])
        x_hat = clean.clone()
        for idx in TARGET_INDICES:
            x_hat[:, :, idx] = latent[:, :, idx]
        value = torch.clamp((x_hat[:, :, VALUE_IDX].mean(dim=(1, 2, 3)) + 1) / 2, 0, 1)

        # 2) M-level noise grid on the shifted training schedule (state-independent)
        shift = self.fm_scheduler.shift
        base = torch.linspace(1.0, self.fm_scheduler.sigma_min, num_score_levels, device=device)
        sig = shift * base / (1 + (shift - 1) * base)                       # [M]
        M = num_score_levels

        # tile self-samples across the M levels -> one batched scoring forward
        x_rep = x_hat.repeat(M, 1, 1, 1, 1)
        t5_rep = t5.repeat(M, 1, 1)
        cmask_rep = get_condition_mask(M * B, device=device, dtype=self.dtype)
        sig_lvl = sig.repeat_interleave(B)                                  # [M*B] level-major
        sig_b = sig_lvl.view(M * B, 1, 1, 1, 1).to(self.dtype)
        tvec = (sig_lvl * self.fm_scheduler.num_train_timesteps).to(self.dtype)
        # The per-level VLB integrand (paper Eq. 2) is an expectation over q(z_t|z_0);
        # one noise draw is a noisy 1-sample estimate. Average N independent draws to
        # suppress that aleatoric MC variance (the paper avoids it via closed-form KL).
        val_acc = torch.zeros(M * B, device=device)
        fs_acc = torch.zeros(M * B, device=device)
        for _ in range(num_noise_draws):
            noise = torch.randn_like(x_rep)
            x_t = x_rep * (1 - sig_b) + noise * sig_b
            x_t = apply_condition_mask(x_rep, x_t, cmask_rep)
            pred = self.forward_denoise(x_t, tvec, t5_rep, cmask_rep, x_rep)
            tgt = noise - x_rep
            val_acc += ((pred[:, :, VALUE_IDX] - tgt[:, :, VALUE_IDX]).float() ** 2).mean(dim=(1, 2, 3))
            fs_acc += ((pred[:, :, FUTURE_STATE_IDX] - tgt[:, :, FUTURE_STATE_IDX]).float() ** 2).mean(dim=(1, 2, 3))
        val_res = (val_acc / num_noise_draws).view(M, B).mean(dim=0)        # avg over levels -> [B]
        fs_res = (fs_acc / num_noise_draws).view(M, B).mean(dim=0)

        # average over the K self-samples -> entropy proxy per original input
        self.fm_scheduler.set_timesteps(num_inference_steps=1000, training=True)  # restore schedule
        return {
            'value': value.view(B0, K).mean(dim=1),
            'value_entropy': val_res.view(B0, K).mean(dim=1),
            'fs_entropy': fs_res.view(B0, K).mean(dim=1),
        }
