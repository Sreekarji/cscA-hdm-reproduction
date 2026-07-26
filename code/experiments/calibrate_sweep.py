import os, sys, numpy as np, torch
_BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
for _sub in ("code/channel", "code/evaluation", "code/utils", "code/hdm"):
    sys.path.insert(0, os.path.join(_BASE, _sub))
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_isr

combos = [
    (0.50, 1.50, 0.10, 0.40),
    (0.50, 1.50, 0.15, 0.45),
    (0.50, 1.50, 0.20, 0.50),
    (0.50, 1.20, 0.10, 0.40),
    (0.60, 1.50, 0.10, 0.40),
    (0.50, 2.00, 0.10, 0.40),
]

for d_lo, d_hi, q_lo, q_hi in combos:
    results = []
    for tpc in (1, 2, 4, 10):
        env = MultiCSCAEnvironment(n_cscas=5, n_relays=5, difficulty="medium", tasks_per_csca=tpc)
        n = env.n_tasks
        n_mcs = 3
        sinr_l, dly_l, dist_l, dint_l, tint_l, isr_l = [], [], [], [], [], []
        bw = torch.ones(1, n, device="cpu") / n
        relay = torch.zeros(1, n, 5, device="cpu")
        mcs = torch.full((1, n, n_mcs), 1.0/n_mcs, device="cpu")
        static_a = {"bandwidth": bw, "relay": relay, "mcs": mcs}
        # Override medium difficulty ranges
        env._override_delay = (d_lo, d_hi)
        env._override_quality = (q_lo, q_hi)
        for _ in range(200):
            s = env.generate_state()
            r = env.step(static_a, s)
            for t in r["tasks"]:
                sinr_l.append(t.get("sinr_db", 0))
                dly_l.append(t.get("tau_S", 0))
                dist_l.append(t.get("vartheta_S", 1))
                dint_l.append(t.get("vartheta_S_int", 0.3))
                tint_l.append(t.get("tau_S_int", 2))
            isr_l.append(compute_isr(r["tasks"]))
        results.append((tpc, np.mean(isr_l), np.mean(dly_l), np.mean(dist_l)))
    
    row = f"d=({d_lo:.2f},{d_hi:.2f}) q=({q_lo:.2f},{q_hi:.2f})"
    for tpc, isr, dly, dist in results:
        row += f"  tpc{tpc}:ISR={isr:.3f}"
    print(row)
