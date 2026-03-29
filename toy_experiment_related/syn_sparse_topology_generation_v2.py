import numpy as np
import os


class SparseTopologyGeneratorV2:
    """
    Synthetic dataset for the Sparse Channel Topology prior (v2).

    Graph: 8 channels on a chain:  0-1-2-3-4-5-6-7

    Key design: each edge (i, i+1) has a shared latent process.
    Each channel's signal = average of its edge-latents + noise.

        ch0 = edge_01                   + noise
        ch1 = (edge_01 + edge_12) / 2   + noise
        ch2 = (edge_12 + edge_23) / 2   + noise
        ...
        ch7 = edge_67                   + noise

    The edge-latents are smooth and predictable (sine + drift).
    The individual noise is moderate (SNR ~ 1 per single channel).

    To predict ch_c, the model benefits from looking at NEIGHBORS
    (who share those same edge-latents).  Non-neighbors share no
    edge-latents and are pure distractors.

    A model with topology-message prior passes information along known
    edges -> efficient denoising -> better prediction.
    """

    def __init__(self, num_samples=150, sample_len=192, seed=2025):
        self.num_samples = num_samples
        self.sample_len = sample_len
        self.num_channels = 8
        self.adjacency = [(i, i + 1) for i in range(self.num_channels - 1)]
        self.rng = np.random.RandomState(seed)

    def _generate_single_sample(self):
        L = self.sample_len
        C = self.num_channels
        t = np.arange(L, dtype=np.float64)

        edge_periods = [20, 28, 36, 24, 32, 18, 40]
        n_edges = C - 1

        edge_latents = np.zeros((L, n_edges))
        for e in range(n_edges):
            phase = self.rng.uniform(0, 2 * np.pi)
            drift = self.rng.uniform(-0.004, 0.004)
            edge_latents[:, e] = (
                np.sin(2 * np.pi * t / edge_periods[e] + phase)
                + drift * t
            )

        channels = np.zeros((L, C))
        for c in range(C):
            participating_edges = []
            if c > 0:
                participating_edges.append(c - 1)
            if c < C - 1:
                participating_edges.append(c)

            signal = np.mean(edge_latents[:, participating_edges], axis=1)
            noise = self.rng.normal(0, 1.0, L)
            channels[:, c] = signal + noise

        return channels.astype(np.float32)

    def generate(self, save_path='./dataset/syn_data/sparse_topo_v2_150.npy'):
        print(f"Generating {self.num_samples} sparse-topology-v2 samples "
              f"(chain graph, {self.num_channels} channels, "
              f"len={self.sample_len}) ...")
        all_samples = [self._generate_single_sample()
                       for _ in range(self.num_samples)]
        final_data = np.stack(all_samples, axis=0)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, final_data)
        print(f"Saved to {save_path}  shape={final_data.shape}")
        return final_data


if __name__ == "__main__":
    gen = SparseTopologyGeneratorV2(num_samples=150, sample_len=192, seed=2025)
    gen.generate(save_path='./dataset/syn_data/sparse_topo_v2_150.npy')
