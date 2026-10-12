"""Two synthetic panel families used in all revision experiments.

RB-like: four outputs per group, 14 evaluators, M = T = 7 (RewardBench-2 shape).
MT-like: twelve outputs per group, 24 evaluators, M = 7, T = 17, 30% of
evaluators discretised (surface-overlap-like ties) (WMT23 shape).
"""
from panels import make_panel

PANELS = {
    "RB-like": dict(gen=dict(n_groups=1200, group_size=4, n_eval=14, shared=0.35, discrete_frac=0.0), M=7, T=7),
    "MT-like": dict(gen=dict(n_groups=450, group_size=12, n_eval=24, shared=0.5, discrete_frac=0.3), M=7, T=17),
}
PANEL_SEEDS = {"RB-like": 101, "MT-like": 202}
DEPTHS = (0.05, 0.10, 0.20, 0.30)
SPLITS = range(10)


def get_panel(name):
    return make_panel(seed=PANEL_SEEDS[name], **PANELS[name]["gen"])
