import numpy as np
from .. import sspspace

from .ssp_agent import SSPAgent


# Made for use with NAS benchmarks
# https://github.com/google-research/nasbench
class SSPNASGraphAgent(SSPAgent):
    """SSP-based agent for Neural Architecture Search (NAS) graph encoding.

    Graphs are encoded by binding layer SPs with their connectivity and
    operation SPs. There is no traditional continuous decoder—graph
    decoding is performed via similarity lookup.

    Parameters (additional kwargs)
    --------------------------------
    max_conns : int
        Maximum number of edges in the graph.
    max_layers : int
        Maximum number of layers (nodes) in the graph.
    num_ops : int
        Number of distinct operations (excluding input/output).
    ssp_dim : int
        Dimensionality of the SP vectors.
    seed : int, optional
    """

    def __init__(self, init_xs, init_ys, **kwargs):
        super().__init__(init_xs, init_ys, None, **kwargs)

    def _set_ssp_space(self, max_conns=9, max_layers=7, num_ops=3,
                       ssp_dim=151, seed=None, **kwargs):
        self.max_conns = max_conns
        self.max_layers = max_layers
        self.num_ops = num_ops + 2  # include input and output
        self.x_dim = int(0.5 * (self.max_layers - 1) * self.max_layers)
        # REVIEW: threshold=0.3 is a magic number with no principled derivation.
        self.threshold = 0.3

        # REVIEW: SPSpace is constructed with seed=seed but SPSpace.__init__ uses
        # 'rng', not 'seed' — the seed is silently ignored via **kwargs.
        self.sp_space = sspspace.SPSpace(
            self.max_layers + self.num_ops + 3,
            dim=ssp_dim, rng=seed, make_quasi_ortho=False)
        self.layer_sps = self.sp_space.vectors[:self.max_layers]
        self.inverse_layer_sps = self.sp_space.inverse_vectors[:self.max_layers]
        self.ops_sps = self.sp_space.vectors[self.max_layers:self.max_layers + self.num_ops]
        self.inverse_ops_sps = self.sp_space.inverse_vectors[self.max_layers:self.max_layers + self.num_ops]
        self.op_slot_sp = self.sp_space.vectors[self.max_layers + self.num_ops].reshape(1, -1)
        self.inverse_op_slot_sp = self.sp_space.inverse_vectors[self.max_layers + self.num_ops].reshape(1, -1)
        self.target_slot_sp = self.sp_space.vectors[self.max_layers + self.num_ops + 1].reshape(1, -1)
        self.inverse_target_slot_sp = self.sp_space.inverse_vectors[self.max_layers + self.num_ops + 1].reshape(1, -1)
        self.other_sp = self.sp_space.vectors[self.max_layers + self.num_ops + 2].reshape(1, -1)
        self.ssp_dim = self.sp_space.dim
        self.identity = self.sp_space.identity()[None, :]

    def _set_decoder(self):
        pass

    def length_scale(self):
        return None

    def encode(self, G):
        """Encode a batch of NAS graphs into SP vectors.

        Parameters
        ----------
        G : np.ndarray, shape (n, x_dim + max_layers)
            Flattened adjacency bits followed by operation indices.

        Returns
        -------
        np.ndarray, shape (n, ssp_dim)

        Notes
        -----
        REVIEW: outer loop over samples is not vectorised; may be slow for large batches.
        """
        G = np.atleast_2d(G.copy())
        S = np.zeros((G.shape[0], self.ssp_dim))

        for n in range(G.shape[0]):
            S2 = self.identity.copy()
            for i in range(self.max_layers - 1):
                layer_i = G[n,
                           int((self.max_layers - 1 + 0.5 * (1 - i)) * i):
                           int((self.max_layers - 1 - 0.5 * i) * (i + 1))]
                op_i = 0 if i == 0 else int(G[n, self.x_dim + i])
                _S = self.sp_space.bind(self.op_slot_sp, self.ops_sps[op_i][None, :])
                target_bundle = self.identity.copy()
                if np.sum(layer_i) > 0:
                    target_bundle = np.sum(
                        self.layer_sps[1 + i + np.where(layer_i > 0)[0], :],
                        axis=0, keepdims=True).reshape(1, -1)
                    target_bundle = target_bundle / np.linalg.norm(target_bundle)
                    _S += self.sp_space.bind(self.target_slot_sp, target_bundle)
                _S = self.sp_space.bind(self.layer_sps[i][None, :], _S).flatten()
                S[n, :] = S[n, :] + _S
                S2 = self.sp_space.bind(S2, _S)
            S[n, :] += 0.1 * S2.flatten()
            S[n, :] = self.sp_space.normalize(S[n, :])

        return S

    def decode(self, ssp):
        """Decode SP vectors back to NAS graph representations.

        Parameters
        ----------
        ssp : np.ndarray, shape (n, ssp_dim)

        Returns
        -------
        np.ndarray, shape (n, x_dim + max_layers)
        """
        ssp = np.atleast_2d(ssp.copy())
        n_conns = 0
        decoded_graphs = np.zeros((ssp.shape[0], self.x_dim + self.max_layers))
        for n in range(ssp.shape[0]):
            decoded_graph = np.zeros((self.max_layers, self.max_layers))
            decoded_ops = np.zeros(self.max_layers)
            for i in range(self.max_layers - 1):
                query_i = self.sp_space.bind(
                    self.inverse_layer_sps[i][None, :], ssp[n, :].reshape(1, -1))
                op_query = self.sp_space.bind(self.inverse_op_slot_sp, query_i)
                op_sims = np.sum(self.ops_sps[1:-1] * op_query, axis=-1)
                op_i = 1 + np.argmax(op_sims)
                decoded_ops[i] = op_i
                target_query = self.sp_space.bind(self.inverse_target_slot_sp, query_i)
                sims = np.sum(self.layer_sps[i + 1:] * target_query, axis=-1)
                target_bundle = [np.zeros(self.ssp_dim)]
                for j, sim in enumerate(sims):
                    if sim >= self.threshold:
                        n_conns += 1
                        decoded_graph[i, 1 + i + j] = 1
                        target_bundle.append(self.layer_sps[i + 1 + j])
                    if n_conns > self.max_conns:
                        continue
                bound_term = (
                    self.sp_space.bind(self.op_slot_sp, self.ops_sps[op_i])
                    + self.sp_space.bind(
                        self.target_slot_sp,
                        np.sum(np.array(target_bundle), axis=0, keepdims=True))
                )
                ssp[n, :] = ssp[n, :] - self.sp_space.bind(
                    self.layer_sps[i][None, :], bound_term).flatten()
            decoded_ops[0] = 0
            decoded_ops[-1] = self.num_ops - 1
            decoded_graphs[n, :] = np.concatenate(
                [decoded_graph[i, i + 1:] for i in range(decoded_graph.shape[0] - 1)]
                + [decoded_ops])
        return decoded_graphs
