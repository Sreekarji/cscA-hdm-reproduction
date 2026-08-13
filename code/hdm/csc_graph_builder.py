import numpy as np
import torch
from torch_geometric.data import HeteroData

class CSCGraphBuilder:
    """
    Builds the Cognitive SemCom (CSC) graph Gt = (Vt, Et)
    as defined in Sun et al. 2026, Section V.B.

    Node types: csca, relay, message, base_station, init
    Edge types:
      (csca, comm_conn, base_station)
      (message, comm_req, csca)
      (message, semantic_conn, relay)
      (init, init_conn, csca)
      (init, init_conn, relay)
      (init, init_conn, base_station)
      (init, init_conn, message)

    Relay-message edges are content-dependent: a message of semantic type t
    connects to every relay whose fixed knowledge set covers t. Edge count
    therefore varies per episode (mean 39.9, range 29-53 at 20 messages /
    5 relays), which is what makes the HAN's attention over heterogeneous
    topology measurable (a static topology collapses HAN to a shared per-node
    MLP). This is the ONLY episode-varying edge type; comm_conn and comm_req
    are deterministic functions of node index.
    """

    N_SEMANTIC_TYPES = 3          # 0 = text, 1 = audio, 2 = image
    RELAY_KNOWLEDGE_P = 0.475     # per-(relay, type) inclusion probability.
                                  # MEASURED at n_messages=20, n_relays=5,
                                  # seed 42, 500 episodes: semantic_conn edges
                                  # mean 39.9, range 29-53 (std 3.6). An earlier
                                  # docstring claimed 45-50; that was never
                                  # measured and is wrong. Expected value is
                                  # n_messages * mean(relays per type) =
                                  # 20 * (1+3+2)/3 = 40. Do not retune this
                                  # without retraining: it changes topology.
    SEMANTIC_TYPE_FEATURE = [0.0, 0.5, 1.0]   # feature-column encoding per type

    # Must match sim_channel.QUAL_INT_MIN / QUAL_INT_SPAN / DELAY_URGENCY_DIVISOR.
    QUAL_INT_MIN          = 0.10
    QUAL_INT_SPAN         = 0.30
    DELAY_URGENCY_DIVISOR = 2.50

    def __init__(
        self,
        n_cscas: int = 5,
        n_relays: int = 5,
        n_messages: int = 5,
        n_base_stations: int = 5,
        csca_feat_dim: int = 3,
        relay_feat_dim: int = 3,
        message_feat_dim: int = 4,
        bs_feat_dim: int = 3,
    ):
        self.n_cscas = n_cscas
        self.n_relays = n_relays
        self.n_messages = n_messages
        self.bs_feat_dim = bs_feat_dim
        self.n_bs = n_base_stations
        self.csca_feat_dim = csca_feat_dim
        self.relay_feat_dim = relay_feat_dim
        self.message_feat_dim = message_feat_dim

        # Relay knowledge: which semantic types (0=text, 1=audio, 2=image) each
        # relay can recover. Fixed per relay for the lifetime of the deployment
        # (paper Sec IV-B: the LKB is provisioned to relays at deployment time),
        # so it must NOT be resampled per episode. Seeded independently of the
        # global RNG so topology is identical across runs and policies.
        _rng = np.random.default_rng(42)
        knowledge = _rng.random((n_relays, self.N_SEMANTIC_TYPES)) < self.RELAY_KNOWLEDGE_P
        for r in range(n_relays):
            if not knowledge[r].any():
                knowledge[r, _rng.integers(0, self.N_SEMANTIC_TYPES)] = True
        self.relay_knowledge = knowledge

    def build(self, system_state: dict = None,
              intent_vectors: list = None) -> HeteroData:
        """
        Build CSC graph.
        intent_vectors: list of [delay_urgency, quality_req] per task.
                       If provided, these override system_state delay/quality intents.
                       This ensures HAN sees real LAM-parsed intent, not random values.
        """
        data = HeteroData()
        n_c = self.n_cscas
        n_r = self.n_relays
        n_m = self.n_messages
        n_b = self.n_bs

        if system_state is not None:
            Rt = system_state.get("Rt", {})
            SCt = system_state.get("SCt", {})

            raw_csca = Rt.get("csca_features", torch.randn(n_c, self.csca_feat_dim).tolist())
            raw_relay = Rt.get("relay_features", torch.randn(n_r, self.relay_feat_dim).tolist())
            raw_bs = Rt.get("bs_features", torch.randn(n_b, self.bs_feat_dim).tolist())

            def _fit(feat_list, target_n, feat_dim):
                t = torch.tensor(feat_list, dtype=torch.float)
                if t.shape[0] >= target_n:
                    return t[:target_n]
                else:
                    pad = torch.randn(target_n - t.shape[0], feat_dim)
                    return torch.cat([t, pad], dim=0)

            csca_feats = _fit(raw_csca, n_c, self.csca_feat_dim)
            relay_feats = _fit(raw_relay, n_r, self.relay_feat_dim)
            bs_feats = _fit(raw_bs, n_b, self.bs_feat_dim)

            # Normalize data sizes with the same divisor sim_channel.py uses when
            # it writes SCt["message_features"] (6e5), so the HAN's size feature
            # matches the environment's own encoding.
            data_sizes = torch.tensor(
                SCt.get("data_sizes", [1e6] * n_m), dtype=torch.float
            )
            data_sizes_norm = torch.clamp(data_sizes / 6e5, 0.0, 1.0)

            # Build message features with REAL intent vectors if provided
            if intent_vectors is not None:
                intent_t = torch.tensor(intent_vectors[:n_m], dtype=torch.float)
                if intent_t.shape[0] < n_m:
                    pad = intent_t[-1:].expand(n_m - intent_t.shape[0], -1)
                    intent_t = torch.cat([intent_t, pad], dim=0)
            else:
                delay_intents = torch.tensor(
                    SCt.get("delay_intents", [1.0] * n_m), dtype=torch.float
                )
                quality_intents = torch.tensor(
                    SCt.get("quality_intents", [0.8] * n_m), dtype=torch.float
                )
                # Same fixed bounds as sim_channel.normalise_intents. Kept inline
                # (rather than importing) so the builder has no dependency on the
                # channel package; the constants below must track that helper.
                delay_norm = 1.0 - torch.clamp(
                    delay_intents / self.DELAY_URGENCY_DIVISOR, 0.0, 1.0)
                qual_norm = torch.clamp(
                    (quality_intents - self.QUAL_INT_MIN) / self.QUAL_INT_SPAN,
                    0.0, 1.0)
                intent_t = torch.stack([delay_norm, qual_norm], dim=1)

            # Message features: [data_size_norm, semantic_type, delay_urgency, quality_req]
            sem_types = SCt.get("semantic_types")
            if sem_types is None:
                sem_types = [i % self.N_SEMANTIC_TYPES for i in range(n_m)]
            sem_types = [int(t) % self.N_SEMANTIC_TYPES for t in sem_types[:n_m]]
            while len(sem_types) < n_m:
                sem_types.append(len(sem_types) % self.N_SEMANTIC_TYPES)
            semantic_type = torch.tensor(
                [self.SEMANTIC_TYPE_FEATURE[t] for t in sem_types],
                dtype=torch.float)
            message_feats = torch.cat([
                data_sizes_norm.unsqueeze(1),
                semantic_type.unsqueeze(1),
                intent_t,
            ], dim=1)  # shape: [n_m, 4]

        else:
            csca_feats = torch.randn(n_c, self.csca_feat_dim)
            relay_feats = torch.randn(n_r, self.relay_feat_dim)
            bs_feats = torch.randn(n_b, self.bs_feat_dim)
            sem_types = [i % self.N_SEMANTIC_TYPES for i in range(n_m)]
            # Column 1 is the semantic type channel; it must carry the same types
            # used to build the relay edges below, not an independent random draw.
            message_feats = torch.randn(n_m, self.message_feat_dim)
            message_feats[:, 1] = torch.tensor(
                [self.SEMANTIC_TYPE_FEATURE[t] for t in sem_types],
                dtype=torch.float)

        init_feat = torch.zeros(1, self.csca_feat_dim)

        data["csca"].x = csca_feats
        data["relay"].x = relay_feats
        data["message"].x = message_feats
        data["base_station"].x = bs_feats
        data["init"].x = init_feat

        # csca -> base_station: nearest BS by position if available, else random
        csca_idx = torch.arange(n_c)
        if (system_state is not None
                and "positions" in system_state.get("Rt", {})):
            pos = system_state["Rt"]["positions"]
            csca_pos_list = pos.get("cscas", [])
            bs_pos_list   = pos.get("bs",    [])
            bs_assign = []
            for cp in csca_pos_list[:n_c]:
                if bs_pos_list:
                    dists = [
                        ((cp[0] - bp[0])**2 + (cp[1] - bp[1])**2) ** 0.5
                        for bp in bs_pos_list[:n_b]
                    ]
                    bs_assign.append(int(np.argmin(dists)))
                else:
                    bs_assign.append(0)
            while len(bs_assign) < n_c:
                bs_assign.append(np.random.randint(0, n_b))
            bs_idx = torch.tensor(bs_assign, dtype=torch.long)
        else:
            bs_idx = torch.randint(0, n_b, (n_c,))
        data["csca", "comm_conn", "base_station"].edge_index = torch.stack(
            [csca_idx, bs_idx], dim=0
        )

        # message -> csca: round-robin, i.e. task i is issued by CSCA i % n_c.
        # This MUST match sim_channel.step(), which draws task i's path loss from
        # csca_positions[i % n_cscas]. A block assignment (i // tasks_per_csca)
        # would tell the HAN that tasks 0..tpc-1 contend for one CSCA's resources
        # while the simulator placed them at tpc different CSCAs.
        msg_idx = torch.arange(n_m)
        csca_assign = torch.tensor(
            [i % n_c for i in range(n_m)], dtype=torch.long
        )
        data["message", "comm_req", "csca"].edge_index = torch.stack(
            [msg_idx, csca_assign], dim=0
        )

        # message -> relay: semantic_type INTERSECT relay_knowledge. A relay can
        # only carry a message whose modality it holds knowledge for, so the
        # edge count varies with the episode's semantic type draw.
        rel_src, rel_dst = [], []
        for m in range(n_m):
            t = sem_types[m]
            for r in range(n_r):
                if self.relay_knowledge[r % self.relay_knowledge.shape[0], t]:
                    rel_src.append(m)
                    rel_dst.append(r)
        if rel_src:
            relay_edge_index = torch.stack([
                torch.tensor(rel_src, dtype=torch.long),
                torch.tensor(rel_dst, dtype=torch.long),
            ], dim=0)
        else:
            relay_edge_index = torch.empty((2, 0), dtype=torch.long)
        data["message", "semantic_conn", "relay"].edge_index = relay_edge_index

        # init -> all other node types
        for node_type, count in [
            ("csca", n_c), ("relay", n_r),
            ("base_station", n_b), ("message", n_m)
        ]:
            src = torch.zeros(count, dtype=torch.long)
            dst = torch.arange(count, dtype=torch.long)
            data["init", "init_conn", node_type].edge_index = torch.stack(
                [src, dst], dim=0
            )

        # Validate all edge indices are within bounds
        for store in data.edge_stores:
            edge_index = store.edge_index
            src_type = store._key[0]
            dst_type = store._key[2]

            n_src = data[src_type].x.shape[0]
            n_dst = data[dst_type].x.shape[0]

            if edge_index.numel() > 0:
                src_idx = edge_index[0]
                dst_idx = edge_index[1]

                if src_idx.min() < 0 or src_idx.max() >= n_src:
                    print(f"WARNING: {store._key} src index out of bounds: "
                          f"min={src_idx.min()}, max={src_idx.max()}, n_src={n_src}")
                    edge_index[0] = src_idx.clamp(0, n_src - 1)

                if dst_idx.min() < 0 or dst_idx.max() >= n_dst:
                    print(f"WARNING: {store._key} dst index out of bounds: "
                          f"min={dst_idx.min()}, max={dst_idx.max()}, n_dst={n_dst}")
                    edge_index[1] = dst_idx.clamp(0, n_dst - 1)

                store.edge_index = edge_index

        return data

    def get_metadata(self):
        node_types = ["csca", "relay", "message", "base_station", "init"]
        edge_types = [
            ("csca", "comm_conn", "base_station"),
            ("message", "comm_req", "csca"),
            ("message", "semantic_conn", "relay"),
            ("init", "init_conn", "csca"),
            ("init", "init_conn", "relay"),
            ("init", "init_conn", "base_station"),
            ("init", "init_conn", "message"),
        ]
        return node_types, edge_types
