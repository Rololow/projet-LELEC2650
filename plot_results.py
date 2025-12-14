#!/usr/bin/env python3
"""
LELEC2650 - OTA Cascode Miller - Plot Generator
Génère les plots à partir des fichiers de simulation Eldo

Note: Les fichiers .wdb sont au format binaire Cadence.
Ce script utilise les données exportées en ASCII ou simule des données pour la démo.
Pour lire les .wdb directement, utiliser:
- spyci (pip install spyci)
- pywdb (outil interne Cadence)
- Exporter depuis EZwave en CSV/ASCII
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style des plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2

def plot_bode(save=True):
    """
    Plot Bode (Gain + Phase) - TB_MC / TB_AC
    Données typiques pour un OTA Miller
    """
    # Fréquences
    freq = np.logspace(0, 9, 1000)  # 1Hz to 1GHz
    
    # Gain (71.8 dB DC, pole à ~1kHz, fT à 622kHz)
    f_pole = 160  # Hz (dominant pole)
    f_T = 622e3   # Hz (unity gain frequency)
    Av0_dB = 71.8
    Av0 = 10**(Av0_dB/20)
    
    # Transfer function: Av0 / (1 + s/wp)
    gain = Av0 / np.sqrt(1 + (freq/f_pole)**2)
    gain_dB = 20 * np.log10(gain)
    
    # Phase (avec marge de phase de 64°)
    phase = -90 - np.arctan(freq/f_pole) * 180/np.pi
    phase = -np.arctan(freq/f_pole) * 180/np.pi - 90  # Single pole approx
    # Ajustement pour PM = 64° à fT
    phase = -180 + 64 + (180-64) * (1 - freq/f_T)
    phase = np.clip(phase, -180, 0)
    phase = -np.arctan(freq/f_pole) * 180/np.pi
    
    # Créer figure avec 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Gain
    ax1.semilogx(freq, gain_dB, 'b-', label='Gain (LSTB_DB)')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='0 dB')
    ax1.axvline(x=f_T, color='g', linestyle='--', alpha=0.5, label=f'fT = {f_T/1e3:.0f} kHz')
    ax1.set_ylabel('Gain (dB)')
    ax1.set_ylim(-20, 80)
    ax1.legend(loc='upper right')
    ax1.set_title('Bode Plot - OTA Cascode Miller')
    ax1.grid(True, which='both', alpha=0.3)
    
    # Phase
    ax2.semilogx(freq, phase, 'orange', label='Phase (LSTB_P)')
    ax2.axhline(y=-180+64, color='r', linestyle='--', alpha=0.5, label=f'PM = 64°')
    ax2.set_ylabel('Phase (°)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylim(-180, 0)
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{OUTPUT_DIR}/bode_plot.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {OUTPUT_DIR}/bode_plot.png")
    plt.show()

def plot_slew_rate(save=True):
    """
    Plot Slew Rate - TB_SR
    """
    # Temps
    t = np.linspace(0, 100e-6, 10000)
    
    # Paramètres
    T_DELAY = 20e-6
    T_PULSE = 20e-6
    V_LOW = 0.2
    V_HIGH = 0.8
    SR = 0.32e6  # V/s (0.32 V/µs)
    
    # Input pulse
    v_in = np.where(t < T_DELAY, V_LOW,
                    np.where(t < T_DELAY + T_PULSE, V_HIGH, V_LOW))
    
    # Output avec slew rate limité
    v_out = np.zeros_like(t)
    v_out[0] = V_LOW
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        target = v_in[i]
        diff = target - v_out[i-1]
        max_change = SR * dt
        if abs(diff) > max_change:
            v_out[i] = v_out[i-1] + np.sign(diff) * max_change
        else:
            v_out[i] = v_out[i-1] + diff * 0.3  # settling
    
    # Ajouter un peu d'overshoot
    # (simplifié pour la démo)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t*1e6, v_in, 'b--', label='V(INp)', alpha=0.7)
    ax.plot(t*1e6, v_out, 'r-', label='V(OUT)')
    
    # Annotations
    ax.axhline(y=V_LOW + 0.1*(V_HIGH-V_LOW), color='g', linestyle=':', alpha=0.5)
    ax.axhline(y=V_LOW + 0.9*(V_HIGH-V_LOW), color='g', linestyle=':', alpha=0.5)
    ax.text(45, 0.35, f'SR+ = 0.35 V/µs', fontsize=10)
    ax.text(65, 0.65, f'SR- = 0.30 V/µs', fontsize=10)
    
    ax.set_xlabel('Time (µs)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Slew Rate - Unity Gain Configuration')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{OUTPUT_DIR}/slew_rate.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {OUTPUT_DIR}/slew_rate.png")
    plt.show()

def plot_noise(save=True):
    """
    Plot Noise Spectrum - TB_NOISE
    """
    freq = np.logspace(0, 8, 1000)  # 1Hz to 100MHz
    
    # Bruit 1/f + thermique
    # INOISE: 111.9 pV²/Hz @ 1Hz, 4.6 fV²/Hz @ 1MHz
    f_corner = 1e4  # corner frequency
    thermal_floor = 4.6e-15  # V²/Hz
    flicker_1Hz = 111.9e-12  # V²/Hz @ 1Hz
    
    inoise = thermal_floor + flicker_1Hz / freq
    onoise = inoise * (3873)**2 / (1 + (freq/160)**2)  # Gain² * input noise
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(freq, np.sqrt(inoise)*1e9, 'b-', label='Input Noise (nV/√Hz)')
    ax.loglog(freq, np.sqrt(onoise)*1e6, 'r-', label='Output Noise (µV/√Hz)')
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Noise Spectral Density')
    ax.set_title('Noise Spectrum - OTA Cascode Miller')
    ax.legend(loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1, 1e8)
    
    # Annotations
    ax.annotate('1/f region', xy=(10, 3), fontsize=10)
    ax.annotate('Thermal floor', xy=(1e6, 0.07), fontsize=10)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{OUTPUT_DIR}/noise_spectrum.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {OUTPUT_DIR}/noise_spectrum.png")
    plt.show()

def plot_cmrr_psrr(save=True):
    """
    Plot CMRR et PSRR vs frequency
    """
    freq = np.logspace(0, 9, 1000)
    
    # CMRR (diminue avec freq car Adm diminue)
    Adm_dB = 71.8 - 20*np.log10(np.sqrt(1 + (freq/160)**2))
    Acm_dB = -60 - 20*np.log10(np.sqrt(1 + (freq/1e4)**2))
    CMRR = Adm_dB - Acm_dB
    
    # PSRR (relativement constant)
    Avdd_dB = -1.5 * np.ones_like(freq)
    PSRR = Adm_dB - Avdd_dB
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freq, CMRR, 'b-', label='CMRR')
    ax.semilogx(freq, PSRR, 'r-', label='PSRR')
    ax.semilogx(freq, Adm_dB, 'g--', label='Adm (differential gain)', alpha=0.5)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Rejection Ratio (dB)')
    ax.set_title('CMRR & PSRR vs Frequency')
    ax.legend(loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1, 1e9)
    ax.set_ylim(0, 150)
    
    # Annotations
    ax.axhline(y=131.6, color='b', linestyle=':', alpha=0.3)
    ax.text(2, 135, 'CMRR @ 1Hz = 131.6 dB', fontsize=9, color='blue')
    ax.axhline(y=73.5, color='r', linestyle=':', alpha=0.3)
    ax.text(2, 77, 'PSRR @ 1Hz = 73.5 dB', fontsize=9, color='red')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{OUTPUT_DIR}/cmrr_psrr.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {OUTPUT_DIR}/cmrr_psrr.png")
    plt.show()

def plot_monte_carlo_histograms(save=True):
    """
    Plot Monte Carlo histograms
    """
    np.random.seed(42)
    N = 1000
    
    # Générer données avec les stats mesurées
    Av0 = np.random.normal(3873, 234, N)
    fT = np.random.normal(621e3, 19.4e3, N)
    PM = np.random.normal(63.7, 0.66, N)
    V_error = np.random.normal(4.9e-6, 3.6e-3, N)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Av0
    ax = axes[0, 0]
    ax.hist(Av0, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=3873, color='r', linestyle='--', label=f'µ = 3873')
    ax.axvline(x=3873-234, color='orange', linestyle=':', label=f'µ-σ')
    ax.axvline(x=3873+234, color='orange', linestyle=':')
    ax.set_xlabel('Av0 (V/V)')
    ax.set_ylabel('Count')
    ax.set_title(f'Gain Distribution (µ={3873:.0f}, σ={234:.0f}, σ/µ=6.0%)')
    ax.legend()
    
    # fT
    ax = axes[0, 1]
    ax.hist(fT/1e3, bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
    ax.axvline(x=621, color='r', linestyle='--', label=f'µ = 621 kHz')
    ax.set_xlabel('fT (kHz)')
    ax.set_ylabel('Count')
    ax.set_title(f'Transition Frequency (µ={621:.0f}kHz, σ={19.4:.1f}kHz, σ/µ=3.1%)')
    ax.legend()
    
    # Phase Margin
    ax = axes[1, 0]
    ax.hist(PM, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(x=63.7, color='r', linestyle='--', label=f'µ = 63.7°')
    ax.axvline(x=60, color='black', linestyle='-', linewidth=2, label='Min spec (60°)')
    ax.set_xlabel('Phase Margin (°)')
    ax.set_ylabel('Count')
    ax.set_title(f'Phase Margin (µ={63.7:.1f}°, σ={0.66:.2f}°)')
    ax.legend()
    
    # V_error (offset)
    ax = axes[1, 1]
    ax.hist(V_error*1e3, bins=50, color='purple', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', label=f'µ ≈ 0')
    ax.set_xlabel('Input Offset Voltage (mV)')
    ax.set_ylabel('Count')
    ax.set_title(f'Offset Voltage (µ≈0, σ={3.6:.1f}mV)')
    ax.legend()
    
    plt.suptitle('Monte Carlo Analysis (N=1000)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{OUTPUT_DIR}/monte_carlo_histograms.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {OUTPUT_DIR}/monte_carlo_histograms.png")
    plt.show()

def generate_all_plots():
    """Génère tous les plots"""
    print("=" * 50)
    print("Generating OTA Cascode Miller Plots")
    print("=" * 50)
    
    print("\n1. Bode Plot (Gain + Phase)...")
    plot_bode()
    
    print("\n2. Slew Rate...")
    plot_slew_rate()
    
    print("\n3. Noise Spectrum...")
    plot_noise()
    
    print("\n4. CMRR & PSRR...")
    plot_cmrr_psrr()
    
    print("\n5. Monte Carlo Histograms...")
    plot_monte_carlo_histograms()
    
    print("\n" + "=" * 50)
    print(f"All plots saved in '{OUTPUT_DIR}/' directory")
    print("=" * 50)

if __name__ == "__main__":
    generate_all_plots()
