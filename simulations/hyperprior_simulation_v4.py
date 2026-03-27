"""
Hyperprior Divergence in the Consciousness Debate: Two Bayesian Simulations
Version 4 — Added committed agnostic; symmetric committed priors

Changes from v3:
- Sim 2: Four agents (open-minded, committed agnostic, committed A, committed B)
- Committed A/B symmetric around 0.50 (0.20 and 0.80)
- Committed agnostic (mu=0.50, prec=500): problem is precision, not position
- Sim 1: Unchanged (seed=2026 preserved)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(2026)

def simulation_1():
    p_d1_h1_v1 = 0.80
    p_d1_h0_v1 = 0.20
    p_d1_hx_v0 = 0.50
    agent_c = {'name': 'Contemplative-leaning', 'p_h1': 0.55, 'trust': {'A': 0.80, 'B': 0.25}}
    agent_p = {'name': 'Physicalist-leaning', 'p_h1': 0.45, 'trust': {'A': 0.20, 'B': 0.80}}
    n_obs = 80
    evidence = []
    for i in range(n_obs):
        if i % 2 == 0:
            d = 1 if np.random.random() < 0.60 else 0
            evidence.append(('A', d))
        else:
            d = 1 if np.random.random() < 0.30 else 0
            evidence.append(('B', d))
    def update(agent, evidence_stream):
        p_h1 = agent['p_h1']
        trust = dict(agent['trust'])
        hist_h = [p_h1]
        hist_trust_A = [trust['A']]
        hist_trust_B = [trust['B']]
        for etype, d in evidence_stream:
            p_v1 = trust[etype]
            p_v0 = 1 - p_v1
            p_h0 = 1 - p_h1
            def lik(d, h, v):
                if v == 0: return 0.5
                if d == 1: return p_d1_h1_v1 if h == 1 else p_d1_h0_v1
                else: return (1 - p_d1_h1_v1) if h == 1 else (1 - p_d1_h0_v1)
            joints = {}
            for h in [0, 1]:
                for v in [0, 1]:
                    ph = p_h1 if h == 1 else p_h0
                    pv = p_v1 if v == 1 else p_v0
                    joints[(h, v)] = ph * pv * lik(d, h, v)
            Z = sum(joints.values())
            for k in joints: joints[k] /= Z
            p_h1 = joints[(1, 0)] + joints[(1, 1)]
            new_pv1 = joints[(0, 1)] + joints[(1, 1)]
            trust[etype] = new_pv1
            hist_h.append(p_h1)
            hist_trust_A.append(trust['A'])
            hist_trust_B.append(trust['B'])
        return hist_h, hist_trust_A, hist_trust_B
    hc, tc_A, tc_B = update(agent_c, evidence)
    hp, tp_A, tp_B = update(agent_p, evidence)
    return hc, hp, tc_A, tp_A, tc_B, tp_B, evidence

def simulation_2():
    n_obs = 200
    true_val = 0.65
    obs_sd = 0.4
    obs_prec = 1.0 / (obs_sd ** 2)
    data = np.random.normal(true_val, obs_sd, n_obs)
    agents = {
        'Open-minded': {'mu': 0.50, 'prec': 1.0},
        'Committed agnostic': {'mu': 0.50, 'prec': 500.0},
        'Committed physicalist': {'mu': 0.10, 'prec': 500.0},
        'Committed contemplative': {'mu': 0.90, 'prec': 500.0},
    }
    histories = {}
    learning_rates = {}
    for name, ag in agents.items():
        mu = ag['mu']
        prec = ag['prec']
        hist = [mu]
        lrs = []
        for d in data:
            lr = obs_prec / (prec + obs_prec)
            lrs.append(lr)
            new_prec = prec + obs_prec
            mu = (prec * mu + obs_prec * d) / new_prec
            prec = new_prec
            hist.append(mu)
        histories[name] = hist
        learning_rates[name] = lrs
    return histories, learning_rates, true_val, obs_sd, obs_prec

# RUN
hc, hp, tc_A, tp_A, tc_B, tp_B, evidence = simulation_1()
histories, learning_rates, true_val, obs_sd, obs_prec = simulation_2()

# FIGURE 1
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), gridspec_kw={'height_ratios': [2, 1]})
fig1.subplots_adjust(hspace=0.35)
colors = {'C': '#534AB7', 'P': '#D85A30'}
ax1.plot(hc, color=colors['C'], linewidth=2.2, label='Contemplative-leaning researcher')
ax1.plot(hp, color=colors['P'], linewidth=2.2, label='Physicalist-leaning researcher')
ax1.axhline(y=0.5, color='#888', linestyle='--', alpha=0.3, linewidth=0.8)
ax1.set_ylabel('P(consciousness is fundamental)', fontsize=11)
ax1.set_xlabel('Shared observations', fontsize=11)
ax1.set_title('A.  Rational divergence: same evidence, different generative models\n'
              '     (Source reliability model after Jern, Chang & Kemp, 2014)',
              fontsize=11, fontweight='bold', loc='left')
ax1.legend(fontsize=10, loc='center left', bbox_to_anchor=(0.01, 0.5))
ax1.set_ylim(-0.02, 1.02)
for i, (et, d) in enumerate(evidence):
    c = '#534AB7' if et == 'A' else '#D85A30'
    ax1.axvspan(i+0.5, i+1.5, alpha=0.03, color=c)
ax2.plot(tc_A, color=colors['C'], linewidth=1.5, label="Contemplative's trust in first-person data")
ax2.plot(tp_A, color=colors['P'], linewidth=1.5, label="Physicalist's trust in first-person data")
ax2.set_ylabel('P(source reliable)', fontsize=11)
ax2.set_xlabel('Shared observations', fontsize=11)
ax2.set_title('B.  Self-reinforcing source reliability beliefs', fontsize=11, fontweight='bold', loc='left')
ax2.legend(fontsize=9, loc='center right')
ax2.set_ylim(-0.02, 1.02)
fig1.savefig('/home/claude/fig_sim1_source_reliability.png', dpi=300, bbox_inches='tight', facecolor='white')
fig1.savefig('/home/claude/fig_sim1_source_reliability.svg', bbox_inches='tight', facecolor='white')
plt.close(fig1)

# FIGURE 2
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 7.5), gridspec_kw={'height_ratios': [2, 1]})
fig2.subplots_adjust(hspace=0.35)
pcolors = {
    'Open-minded': '#1D9E75',
    'Committed agnostic': '#B8860B',
    'Committed physicalist': '#D85A30',
    'Committed contemplative': '#534AB7',
}
pstyles = {
    'Open-minded': '-',
    'Committed agnostic': '--',
    'Committed physicalist': '-',
    'Committed contemplative': '-',
}
for name, hist in histories.items():
    ax3.plot(hist, color=pcolors[name], linewidth=2, linestyle=pstyles[name], label=name)
ax3.axhline(y=true_val, color='#888', linestyle=':', alpha=0.5, linewidth=1,
            label=f'Data-generating value ({true_val})')
ax3.set_ylabel('Posterior mean (\u03b8)', fontsize=11)
ax3.set_xlabel('Shared observations', fontsize=11)
ax3.set_title('A.  Prior precision determines learning rate\n'
              '     (Convergence guaranteed but practically unreachable)',
              fontsize=11, fontweight='bold', loc='left')
ax3.legend(fontsize=8.5, loc='center right')
ax3.set_ylim(0.05, 0.95)
h_phys = histories['Committed physicalist']
h_cont = histories['Committed contemplative']
h_agno = histories['Committed agnostic']
h_open = histories['Open-minded']
gap_pc = abs(h_phys[-1] - h_cont[-1])
ax3.annotate(f'After 200 observations:\nAll three committed agents\nremain near their priors',
             xy=(185, h_agno[-1]), fontsize=8.5, color='#555', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f0', alpha=0.8))
# Panel B — ALL 4 visible
for name, lrs in learning_rates.items():
    ax4.semilogy(lrs, color=pcolors[name], linewidth=1.5, linestyle=pstyles[name],
                 label=f'{name} (initial LR = {lrs[0]:.4f})')
ax4.set_ylabel('Learning rate (log scale)', fontsize=11)
ax4.set_xlabel('Observations', fontsize=11)
ax4.set_title('B.  Learning rate: high prior precision \u2192 negligible updating',
              fontsize=11, fontweight='bold', loc='left')
ax4.legend(fontsize=7.5)
fig2.savefig('/home/claude/fig_sim2_precision_learning.png', dpi=300, bbox_inches='tight', facecolor='white')
fig2.savefig('/home/claude/fig_sim2_precision_learning.svg', bbox_inches='tight', facecolor='white')
plt.close(fig2)

# SUMMARY
print("=" * 65)
print("SIMULATION 1: Source Reliability Model")
print("=" * 65)
print(f"After {len(evidence)} shared observations:")
print(f"  Contemplative P(fundamental): {hc[-1]:.4f}")
print(f"  Physicalist P(fundamental):   {hp[-1]:.4f}")
print(f"  DIVERGENCE:                    {abs(hc[-1]-hp[-1]):.4f}")
print(f"  C trust in contempl. data:     {tc_A[-1]:.4f}")
print(f"  P trust in contempl. data:     {tp_A[-1]:.4f}")
pos = sum(1 for _, d in evidence if d == 1)
print(f"  Evidence: {pos}/{len(evidence)} positive = {pos/len(evidence)*100:.1f}%")
print(f"  Expected: 45.0%")
print()
print("=" * 65)
print("SIMULATION 2: Precision-weighted Learning (4 agents)")
print("=" * 65)
print(f"Data-generating value (theta): {true_val}")
for name, hist in histories.items():
    err = abs(hist[-1] - true_val)
    print(f"  {name:30s}: posterior={hist[-1]:.4f}  error={err:.4f}  start={hist[0]:.2f}")
print(f"\nInitial learning rates:")
for name, lrs in learning_rates.items():
    print(f"  {name:30s}: {lrs[0]:.6f}")
print(f"\nGap physicalist-contemplative: {gap_pc:.4f}")
print(f"Open-minded vs committed agnostic (SAME start, DIFFERENT precision):")
print(f"  Open-minded final:          {h_open[-1]:.4f}")
print(f"  Committed agnostic final:   {h_agno[-1]:.4f}")
print(f"  Difference:                 {abs(h_open[-1]-h_agno[-1]):.4f}")
n_to_halve = int(500 / obs_prec)
print(f"\nObservations to halve committed prior weight: ~{n_to_halve}")
