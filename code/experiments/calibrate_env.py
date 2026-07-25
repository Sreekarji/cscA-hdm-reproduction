import os, sys
_BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
for _sub in ("code/channel", "code/evaluation", "code/utils", "code/hdm"):
    sys.path.insert(0, os.path.join(_BASE, _sub))
import numpy as np
import torch
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_isr

for tpc in (1, 2, 4, 10):
    env = MultiCSCAEnvironment(n_cscas=5, n_relays=5, difficulty="medium", tasks_per_csca=tpc)
    n = env.n_tasks
    n_mcs = 3
    sinr_l, dly_l, dist_l, dint_l, tint_l, isr_l = [], [], [], [], [], []
    bw = torch.ones(1, n, device="cpu") / n
    relay = torch.zeros(1, n, 5, device="cpu")
    mcs = torch.full((1, n, n_mcs), 1.0/n_mcs, device="cpu")
    static_a = {"bandwidth": bw, "relay": relay, "mcs": mcs}
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
    dist, dint = np.array(dist_l), np.array(dint_l)
    dly,  tint = np.array(dly_l),  np.array(tint_l)
    print(f"tpc={tpc:2d}  SINR={np.mean(sinr_l):6.2f}dB  "
          f"dist={dist.mean():.3f}(int={dint.mean():.3f})  "
          f"delay={dly.mean():.3f}s(int={tint.mean():.3f}s)  "
          f"P(qual)={np.mean(dist<=dint):.3f}  "
          f"P(delay)={np.mean(dly<=tint):.3f}  "
          f"ISR={np.mean(isr_l):.3f}")
