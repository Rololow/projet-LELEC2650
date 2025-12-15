#!/usr/bin/env python3
"""
Parse PVT_results.txt produced by TB_PVT.cir and plot metrics vs corners.
Usage: python plot_pvt.py
Outputs: plots/pvt_av0dB.png, plots/pvt_fT.png, plots/pvt_phase_margin.png, plots/pvt_power.png
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

repo_root = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(repo_root, 'PVT_results.txt')
out_dir = os.path.join(repo_root, 'plots')
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

if not os.path.exists(data_file):
    print(f"PVT results file not found: {data_file}")
    sys.exit(1)

# Expected corner order in TB_PVT.cir (16 corners)
corner_labels = [
    'FF_LOW_COLD','FF_LOW_HOT','FF_HIGH_COLD','FF_HIGH_HOT',
    'SS_LOW_COLD','SS_LOW_HOT','SS_HIGH_COLD','SS_HIGH_HOT',
    'SF_LOW_COLD','SF_LOW_HOT','SF_HIGH_COLD','SF_HIGH_HOT',
    'FS_LOW_COLD','FS_LOW_HOT','FS_HIGH_COLD','FS_HIGH_HOT'
]

# Read numeric data lines (skip empty/comment lines)
rows = []
with open(data_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Skip spice-like headers and end markers
        if line.startswith('.DATA') or line.startswith('.ENDDATA') or line.startswith('*') or line.lower().startswith('meas'):
            continue
        # Handle continuation lines that start with '+' (common in .DATA output)
        if line.startswith('+'):
            line = line[1:].strip()
            if not line:
                continue
        parts = line.split()
        # try to parse floats
        try:
            nums = [float(p) for p in parts]
            rows.append(nums)
        except ValueError:
            # ignore header or malformed lines
            continue

if len(rows) == 0:
    print('No numeric data found in', data_file)
    sys.exit(1)

rows = np.array(rows)
# Determine columns mapping from TB_PVT.cir .printfile: meas(Av0dB) meas(Av0) meas(fT) meas(Phase_Margin) meas(Power)
# So columns: Av0dB, Av0, fT, Phase_Margin, Power
if rows.shape[1] < 5:
    print('Unexpected data columns in', data_file, 'found', rows.shape[1])
    sys.exit(1)

av0dB = rows[:,0]
fT = rows[:,2]
phase = rows[:,3]
power = rows[:,4]

n = len(av0dB)
if n > len(corner_labels):
    # Extend with generic labels if file contains more corners than expected
    labels = corner_labels + [f'CORNER_{i+1}' for i in range(len(corner_labels), n)]
else:
    labels = corner_labels[:n]

x = np.arange(n)

plt.figure(figsize=(8,3))
plt.plot(x, av0dB, marker='o')
plt.xticks(x, labels, rotation=45, ha='right')
plt.ylabel('Av0 (dB)')
plt.title('PVT: Av0 (dB)')
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(out_dir, 'pvt_av0dB.png'))
print('Saved', os.path.join(out_dir, 'pvt_av0dB.png'))

plt.figure(figsize=(8,3))
plt.plot(x, fT/1e6, marker='o')
plt.xticks(x, labels, rotation=45, ha='right')
plt.ylabel('fT (MHz)')
plt.title('PVT: fT')
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(out_dir, 'pvt_fT.png'))
print('Saved', os.path.join(out_dir, 'pvt_fT.png'))

plt.figure(figsize=(8,3))
plt.plot(x, phase, marker='o')
plt.xticks(x, labels, rotation=45, ha='right')
plt.ylabel('Phase Margin (deg)')
plt.title('PVT: Phase Margin')
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(out_dir, 'pvt_phase_margin.png'))
print('Saved', os.path.join(out_dir, 'pvt_phase_margin.png'))

plt.figure(figsize=(8,3))
plt.plot(x, power*1e6, marker='o')
plt.xticks(x, labels, rotation=45, ha='right')
plt.ylabel('Power (uW)')
plt.title('PVT: Power')
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(out_dir, 'pvt_power.png'))
print('Saved', os.path.join(out_dir, 'pvt_power.png'))

print('Done.')
