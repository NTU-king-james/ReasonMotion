"""
NOTE: Please use the MetricsEvaluator class for evaluation instead of using the helper functions (ampjpe, fmpjpe) directly.

Usage:
    from utils.metrics import MetricsEvaluator

    evaluator = MetricsEvaluator(config, device='cuda')
    metrics = evaluator.evaluate(model, dataloader, nsample=1)
"""
import torch
import numpy as np
from tqdm import tqdm
from scipy import fft

from utils.text_encoder import TextEncoder


def compute_ldlj(positions, fps=25):
    """
    Compute Log Dimensionless Jerk (LDLJ) for motion smoothness.
    
    LDLJ = -ln|DLJ|
    
    where DLJ = -((t2-t1)^5 / v_peak^2) * integral(|d^2v/dt^2|^2 dt)
    
    Args:
        positions: Motion positions array of shape (T, J, 3) or (T, J*3)
                   where T is time steps, J is number of joints
        fps: Frames per second of the motion data
    
    Returns:
        ldlj: Log Dimensionless Jerk value (higher/less negative = smoother)
    
    Reference:
        Balasubramanian et al. "On the analysis of movement smoothness" 
        Journal of NeuroEngineering and Rehabilitation (2015)
    """
    positions = np.asarray(positions)
    
    # Reshape to (T, J, 3) if needed
    if positions.ndim == 2:
        T, K = positions.shape
        positions = positions.reshape(T, K // 3, 3)
    
    T = positions.shape[0]
    dt = 1.0 / fps
    duration = (T - 1) * dt
    
    if T < 4:
        return 0.0  # Not enough frames to compute jerk
    
    # Compute velocity: v(t) = dx/dt
    velocity = np.diff(positions, axis=0) / dt  # (T-1, J, 3)
    
    # Compute speed (magnitude of velocity) for each joint
    speed = np.linalg.norm(velocity, axis=-1)  # (T-1, J)
    
    # Average speed across all joints
    avg_speed = speed.mean(axis=1)  # (T-1,)
    
    # Peak speed
    v_peak = avg_speed.max()
    
    if v_peak < 1e-10:
        return 0.0  # No movement
    
    # Compute acceleration: a(t) = dv/dt (second derivative)
    acceleration = np.diff(avg_speed) / dt  # (T-2,)
    
    # Compute jerk: j(t) = da/dt (third derivative)  
    jerk = np.diff(acceleration) / dt  # (T-3,)
    
    # Compute integral of |jerk|^2 using trapezoidal rule
    jerk_squared_integral = np.trapz(jerk ** 2, dx=dt)
    
    # Dimensionless Jerk (DLJ)
    dlj = -((duration ** 5) / (v_peak ** 2)) * jerk_squared_integral
    
    # Log Dimensionless Jerk (LDLJ)
    if abs(dlj) < 1e-10:
        return 0.0
    
    ldlj = -np.log(abs(dlj))
    
    return ldlj


def compute_sparc(positions, fps=30, padlevel=4, fc=10.0, amp_th=0.05):
    """
    Compute SPectral ARC length (SPARC) for motion smoothness.
    
    SPARC = -integral_0^wc sqrt((1/wc)^2 + (dV_hat/dw)^2) dw
    
    where V_hat(w) = V(w) / V(0) is the normalized Fourier magnitude spectrum
    
    Args:
        positions: Motion positions array of shape (T, J, 3) or (T, J*3)
                   where T is time steps, J is number of joints
        fps: Frames per second of the motion data
        padlevel: Zero padding level for FFT (power of 2 multiplier)
        fc: Maximum cutoff frequency in Hz (default 10 Hz for human motion)
        amp_th: Amplitude threshold for adaptive cutoff (default 0.05)
    
    Returns:
        sparc: SPARC value (higher/less negative = smoother)
                Typical values: ~-1.6 for healthy movements, more negative for impaired
    
    Reference:
        Balasubramanian et al. "On the analysis of movement smoothness" 
        Journal of NeuroEngineering and Rehabilitation (2015)
    """
    positions = np.asarray(positions)
    
    # Reshape to (T, J, 3) if needed
    if positions.ndim == 2:
        T, K = positions.shape
        positions = positions.reshape(T, K // 3, 3)
    
    T = positions.shape[0]
    dt = 1.0 / fps
    
    if T < 4:
        return 0.0  # Not enough frames
    
    # Compute velocity: v(t) = dx/dt
    velocity = np.diff(positions, axis=0) / dt  # (T-1, J, 3)
    
    # Compute speed (magnitude of velocity) for each joint
    speed = np.linalg.norm(velocity, axis=-1)  # (T-1, J)
    
    # Average speed across all joints
    avg_speed = speed.mean(axis=1)  # (T-1,)
    
    if avg_speed.max() < 1e-10:
        return 0.0  # No movement
    
    # Zero padding for better frequency resolution
    nfft = int(2 ** (np.ceil(np.log2(len(avg_speed))) + padlevel))
    
    # Compute FFT
    speed_spectrum = np.abs(fft.fft(avg_speed, n=nfft))
    
    # Frequency axis
    freq = fft.fftfreq(nfft, d=dt)
    
    # Only positive frequencies up to fc
    pos_freq_mask = (freq >= 0) & (freq <= fc)
    freq_pos = freq[pos_freq_mask]
    spectrum_pos = speed_spectrum[pos_freq_mask]
    
    # Normalize spectrum
    if spectrum_pos[0] < 1e-10:
        return 0.0
    
    spectrum_norm = spectrum_pos / spectrum_pos[0]
    
    # Adaptive cutoff: find where normalized spectrum drops below threshold
    below_th = np.where(spectrum_norm < amp_th)[0]
    if len(below_th) > 0:
        cutoff_idx = below_th[0]
    else:
        cutoff_idx = len(freq_pos) - 1
    
    # Ensure at least some frequency content
    cutoff_idx = max(cutoff_idx, 2)
    
    freq_sel = freq_pos[:cutoff_idx + 1]
    spectrum_sel = spectrum_norm[:cutoff_idx + 1]
    
    if len(freq_sel) < 2:
        return 0.0
    
    # Compute arc length: integral of sqrt((1/wc)^2 + (dV_hat/dw)^2)
    # Use numerical differentiation for dV_hat/dw
    dfreq = np.diff(freq_sel)
    dspectrum = np.diff(spectrum_sel)
    
    # Avoid division by zero
    dfreq = np.where(dfreq < 1e-10, 1e-10, dfreq)
    
    # Derivative dV_hat/dw
    d_spectrum_d_freq = dspectrum / dfreq
    
    # Cutoff frequency
    wc = freq_sel[-1] if freq_sel[-1] > 0 else 1.0
    
    # Arc length integrand: sqrt((1/wc)^2 + (dV_hat/dw)^2)
    integrand = np.sqrt((1.0 / wc) ** 2 + d_spectrum_d_freq ** 2)
    
    # Compute arc length using trapezoidal integration
    arc_length = np.trapz(integrand, freq_sel[:-1])
    
    # SPARC is negative arc length
    sparc = -arc_length
    
    return sparc

def ampjpe(batch_pred, batch_gt, target_dim):
    """Calculates A-MPJPE by reshaping (B, N, target_dim, T) to (B, N, T, J, 3).
    Computes best-of-N error: min error among N samples for each batch item.
    """
    B, N, K, T = batch_pred.shape
    p = batch_pred.transpose(-1, -2).reshape(B, N, T, target_dim // 3, 3)
    g = batch_gt.transpose(-1, -2).reshape(B, N, T, target_dim // 3, 3)
    
    # Calculate Euclidean distance: (B, N, T, J)
    dist = torch.norm(g - p, p=2, dim=-1)
    
    # Average over time (T) and joints (J) to get error per sample: (B, N)
    error_per_sample = dist.mean(dim=(-1, -2))
    
    # Best of N: min error over samples -> (B,)
    best_error = error_per_sample.min(dim=1)[0]
    
    # Average over batch
    return best_error.mean() * 1000

def fmpjpe(batch_pred, batch_gt, target_dim):
    """Calculates F-MPJPE for the final frame of the sequence.
    Computes best-of-N error: min error among N samples for each batch item.
    """
    B, N, K, T = batch_pred.shape
    p = batch_pred[..., -1].reshape(B, N, target_dim // 3, 3)
    g = batch_gt[..., -1].reshape(B, N, target_dim // 3, 3)
    
    # Calculate Euclidean distance: (B, N, J)
    dist = torch.norm(g - p, p=2, dim=-1)
    
    # Average over joints (J) to get error per sample: (B, N)
    error_per_sample = dist.mean(dim=-1)
    
    # Best of N: min error over samples -> (B,)
    best_error = error_per_sample.min(dim=1)[0]
    
    # Average over batch
    return best_error.mean() * 1000

def get_diversity(activation, diversity_times):
    """Implementation of average Euclidean distance of M pairs."""
    num_samples = activation.shape[0]
    if num_samples < 2: return 0.0
    
    # Generate M unique unordered pairs (i, j) where i < j
    if num_samples < 200:
        indices = np.array([(i, j) for i in range(num_samples) for j in range(i + 1, num_samples)])
    else:
        # Randomly sample pairs for efficiency
        indices = np.random.randint(0, num_samples, size=(diversity_times * 2, 2))
        indices = indices[indices[:, 0] < indices[:, 1]] # Ensure i < j and unique

    if len(indices) > diversity_times:
        indices = indices[np.random.choice(len(indices), diversity_times, replace=False)]
    
    # Calculate ||z_i - z_j||_2
    diff = activation[indices[:, 0]] - activation[indices[:, 1]]
    return np.linalg.norm(diff, axis=1).mean()

class MetricsEvaluator:
    def __init__(self, config, device='cuda'):
        self.device = device
        self.target_dim = config['model']['target_dim'] if 'target_dim' in config['model'] else 72
        self.input_n = config['data'].get('input_n', 0)
        data_name = config['data'].get('name', 'FineFS').lower()
        default_fps = 25 if data_name == 'h36m' else 30
        self.fps = config['data'].get('fps', default_fps)
        self.text_encoder = TextEncoder(device=str(device))

    def evaluate(self, model, dataloader, nsample=1):
        """
        Evaluate the model on a given dataloader.

        Args:
            model: The PyTorch model to evaluate.
            dataloader: DataLoader providing the validation dataset.
            nsample (int): Number of samples to generate per input. (if > 1, the result is best-of-N)

        Returns:
            dict: A dictionary containing the computed metrics:
                - "A-MPJPE": Average Mean Per Joint Position Error over time.
                - "F-MPJPE": Final Frame Mean Per Joint Position Error.
                - "Div_Spatial": Spatial diversity (if relevant).
                - "Div_Vel": Velocity diversity (if relevant).
                - "Div_Acc": Acceleration diversity (if relevant).
                - "LDLJ": Log Dimensionless Jerk (smoothness metric).
                - "SPARC": Spectral Arc Length (smoothness metric).
        """
        model.eval()
        amp_total, fmp_total, samples_list, n_batches = 0, 0, [], 0
        ldlj_total, sparc_total = 0, 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                t_emb = self.text_encoder(batch.get("motion_name")) if self.text_encoder else None
                
                try:
                    samples, gt = model.evaluate(batch, nsample, t_emb)[:2] # (B, N, K, T)
                    if gt.dim() == 3: gt = gt.unsqueeze(1)
                    gt = gt.expand_as(samples)

                    # Only evaluate predicted frames (input_n:)
                    p_part, g_part = samples[..., self.input_n:], gt[..., self.input_n:]
                    
                    amp_total += ampjpe(p_part, g_part, self.target_dim).item()
                    fmp_total += fmpjpe(p_part, g_part, self.target_dim).item()
                    
                    # Compute smoothness metrics (LDLJ and SPARC)
                    # Take best sample (first) and convert to numpy: (B, K, T) -> (B, T, K)
                    pred_np = p_part[:, 0].transpose(-1, -2).cpu().numpy()  # (B, T, K)
                    for b in range(pred_np.shape[0]):
                        ldlj_total += compute_ldlj(pred_np[b], fps=self.fps)
                        sparc_total += compute_sparc(pred_np[b], fps=self.fps)
                    
                    # Store samples for diversity calculation
                    samples_list.append(p_part.cpu().numpy())
                    n_batches += 1
                except Exception: continue

        total_samples = sum(s.shape[0] for s in samples_list) if samples_list else 1
        m = {
            "A-MPJPE": amp_total/n_batches, 
            "F-MPJPE": fmp_total/n_batches,
            "LDLJ": ldlj_total/total_samples,
            "SPARC": sparc_total/total_samples
        }
        
        if samples_list:
            # Flatten all generations into a single sample pool for the dataset
            all_s = np.concatenate(samples_list, axis=0) # (Total_B, N, K, T_out)
            all_s = all_s.reshape(-1, self.target_dim, all_s.shape[-1]) # (Total_Samples, target_dim, T_out)
            
            div_times = min(300, len(all_s) // 2)
            
            # Spatial Diversity: Euclidean distance of joint features
            m["Div_Spatial"] = get_diversity(all_s.reshape(len(all_s), -1), div_times)
            
            # Temporal Diversity: Euclidean distance of velocities (diff) and accelerations (double diff)
            vel = np.diff(all_s, axis=-1).reshape(len(all_s), -1)
            acc = np.diff(all_s, axis=-1, n=2).reshape(len(all_s), -1)
            m["Div_Vel"], m["Div_Acc"] = get_diversity(vel, div_times), get_diversity(acc, div_times)
                 
        return m
