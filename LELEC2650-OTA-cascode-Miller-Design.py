# -*- coding: utf-8 -*-
"""
Miller OTA 1st order Python sizing

65nm CMOS technology (TSMC)

Sylvain Favresse, 2025

netlist : 
M14 VBIAS2 VBIAS2 0 0 NMOS
M13 VBIAS3 VBIAS2 0 0 NMOS
M15 VBIAS1 VBIAS1 VBIAS2 VBIAS2 NMOS
M7 N007 VBIAS2 0 0 NMOS
M8 N006 VBIAS2 0 0 NMOS
M6 N004 VBIAS1 N007 N007 NMOS
M5 N002 VBIAS1 N006 N006 NMOS
M10 N003 NC_01 0 0 NMOS
C1 N003 N002 Cm
C2 N003 0 CL
M4 N004 N001 VDD VDD PMOS
M3 N002 N001 VDD VDD PMOS
M9 N003 VBIAS3 VDD VDD PMOS
M2 N006 VIN+ N005 N005 PMOS
M1 N007 VIN- N005 N005 PMOS
M11 N005 VBIAS3 VDD VDD PMOS
M12 VBIAS3 VBIAS3 VDD VDD PMOS
I1 VDD VBIAS1 I
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scint
import settings


# =============================================================================
# Library extraction
# =============================================================================

nch_lvt = settings.read_txt('data/nch_lvt.txt')
nch_rvt = settings.read_txt('data/nch.txt')
nch_hvt = settings.read_txt('data/nch_hvt.txt')
pch_lvt = settings.read_txt('data/pch_lvt.txt')
pch_rvt = settings.read_txt('data/pch.txt')
pch_hvt = settings.read_txt('data/pch_hvt.txt')

# =============================================================================
# Specifications
# =============================================================================

fT_spec = 1.25e6
CL = 10e-12
PM_spec = 60 # °
VDD = 1.2

# =============================================================================
# Design choices
# =============================================================================

L = 2e-6

L1 = L; L2 = L1
L3 = L; L4 = L3
L5 = L; L6 = L5
L7 = L; L8 = L7
L9 = L; L10= L9
L11= L9; L12= L9
L13= L9; L14= L9; L15= L9

# gm/ID values - reduced to ensure saturation (not subthreshold/linear)
# Higher gm/ID = weak inversion (low Vdsat but risk of linear region)
# Lower gm/ID = strong inversion (higher Vdsat, guaranteed saturation)
gmid1 = 12.0; gmid2 = gmid1   # Input pair: moderate inversion
gmid3 = 12.0; gmid4 = gmid3   # Active load
gmid5 = 10.0; gmid6 = gmid5   # Cascode NMOS: lower gm/ID for higher Vdsat
gmid8 = 10.0; gmid7 = gmid8   # Current source NMOS: need good Vdsat margin
gmid9 = 12.0;                 # Output PMOS
gmid10 = 10.0                 # Output NMOS current source
gmid11 = 12.0; gmid12 = gmid11;  # Tail current source
gmid14 = 10.0; gmid13 = gmid14   # Bias transistors
gmid15 = 10.0                    # Cascode bias

M1 = pch_lvt; M2 = pch_lvt
M3 = pch_lvt; M4 = pch_lvt
M5 = nch_lvt; M6 = nch_lvt
M7 = nch_lvt; M8 = nch_lvt
M9 = pch_lvt; M10= nch_lvt
M11= pch_lvt; M12= pch_lvt
M13= nch_lvt; M14= nch_lvt; M15= nch_lvt

zero_to_wT_ratio = 10 # determines gm5 / gm10

# =============================================================================
# Design algorithm
# =============================================================================

"""
Important equations to remind (see lecture slides)
    wT = gm1 / Cf
    GBW = gm5 / (2*np.pi * Cf)
    pd = GBW / gain
    pnd = gm10 * Cf / (2*np.pi * (C1*C2 + (C1+C2)*Cf))
    z = gm10 / (2*np.pi * Cf)
    PM = 90° - arctan(wT/pnd) - arctan(wT/z)
    
The sizing of Cf is done based on a desired PM, that depends on the parasitic
capacitors. Those parasitics can only be evaluated when the transistor sizes are known.
We thus use an iterative design flow where we update the values of the parasitic
capacitances, until the required value of Cf stabilizes.

ADDITIONAL POLE AT NODE N006:
The cascode intermediate node (N006 = drain M2/M8, source M5) introduces a 
parasitic pole that degrades phase margin:
    p_N006 = gm5 / (2*pi * C_N006)
    C_N006 = Cdb2 + Cgs5 + Cgso5 + Cdb8 + Cgdo8
This pole must be included in PM calculation.
"""

# Note: pnd_to_wT_ratio will be computed iteratively to account for p_N006
# Old formula (without p_N006): pnd_to_wT_ratio = 1 / np.tan(np.pi/2 - PM_spec*np.pi/180 - np.arctan(1/zero_to_wT_ratio))
pnd_to_wT_ratio_init = 1 / np.tan(np.pi/2 - PM_spec*np.pi/180 - np.arctan(1/zero_to_wT_ratio)) # 2.2 with 60° PM (see lecture slides)

Cf = CL # first guess

error = 1
max_error = 1e-6
iteration = 0
max_iteration = 100

while error > max_error and iteration < max_iteration:
    
    # Differential pair
    gm1 = 2*np.pi * fT_spec * Cf
    ID1 = gm1 / gmid1
    idn1 = float(scint.interp1d(M1['GMID'], M1['IN'])(gmid1))
    W_over_L1 = ID1 / idn1
    W1 = W_over_L1 * L1
    W2 = W_over_L1 * L2

    # Cascode stage
    
    ID3 = ID1
    idn3 = float(scint.interp1d(M3['GMID'], M3['IN'])(gmid3))
    W_over_L3 = ID3 / idn3
    W3 = W_over_L3 * L3
    W4 = W_over_L3 * L4

    ID5 = ID1
    idn5 = float(scint.interp1d(M5['GMID'], M5['IN'])(gmid5))
    W_over_L5 = ID5 / idn5
    W5 = W_over_L5 * L5
    W6 = W_over_L5 * L6
    gm5 = gmid5 * ID5

    ID7 = 2*ID5
    idn7 = float(scint.interp1d(M7['GMID'], M7['IN'])(gmid7))
    W_over_L7 = ID7 / idn7
    W7 = W_over_L7 * L7
    W8 = W_over_L7 * L8

    # Output Miller stage
    
    gm10 = gm5 * zero_to_wT_ratio
    ID10 = gm10 / gmid10
    idn10 = float(scint.interp1d(M10['GMID'], M10['IN'])(gmid10))
    W_over_L10 = ID10 / idn10
    W10 = W_over_L10 * L10

    ID9 = ID10
    idn9 = float(scint.interp1d(M9['GMID'], M9['IN'])(gmid9))
    W_over_L9 = ID9 / idn9
    W9 = W_over_L9 * L9
    
    # Bias PMOS transistors
    ID15 = ID5
    idn15 = float(scint.interp1d(M15['GMID'], M15['IN'])(gmid15))
    W_over_L15 = ID15 / idn15
    W15 = W_over_L15 * L15
    W14 = W_over_L15 * L14

    ID13 = ID7
    idn13 = float(scint.interp1d(M13['GMID'], M13['IN'])(gmid13))
    W_over_L13 = ID13 / idn13
    W13 = W_over_L13 * L13

    ID12 = ID13
    idn12 = float(scint.interp1d(M12['GMID'], M12['IN'])(gmid12))
    W_over_L12 = ID12 / idn12
    W12 = W_over_L12 * L12
    W11 = W_over_L12 * L11
    
    # Parasitic capacitances
    Cgs3 = float(scint.interp1d(M3['GMID'], M3['CGS'])(gmid3)) * W3 * L3
    Cgso3 = float(scint.interp1d(M3['GMID'], M3['CGS0'])(gmid3)) * W3
    Cbd3 = float(scint.interp1d(M3['GMID'], M3['CBD'])(gmid3)) * W3
    Cgdo3 = float(scint.interp1d(M3['GMID'], M3['CGD0'])(gmid3)) * W3
    Cgs5 = float(scint.interp1d(M5['GMID'], M5['CGS'])(gmid5)) * W5 * L5
    Cgso5 = float(scint.interp1d(M5['GMID'], M5['CGS0'])(gmid5)) * W5
    Cbd5 = float(scint.interp1d(M5['GMID'], M5['CBD'])(gmid5)) * W5
    Cgdo5 = float(scint.interp1d(M5['GMID'], M5['CGD0'])(gmid5)) * W5
    Cgdo9 = float(scint.interp1d(M9['GMID'], M9['CGD0'])(gmid9)) * W9
    Cbd9 = float(scint.interp1d(M9['GMID'], M9['CBD'])(gmid9)) * W9
    Cgs9 = float(scint.interp1d(M9['GMID'], M9['CGS'])(gmid9)) * W9 * L9
    Cgso9 = float(scint.interp1d(M9['GMID'], M9['CGS0'])(gmid9)) * W9
    Cgs10 = float(scint.interp1d(M10['GMID'], M10['CGS'])(gmid10)) * W10 * L10
    Cgso10 = float(scint.interp1d(M10['GMID'], M10['CGS0'])(gmid10)) * W10
    Cbd10 = float(scint.interp1d(M10['GMID'], M10['CBD'])(gmid10)) * W10
    Cgdo10 = float(scint.interp1d(M10['GMID'], M10['CGD0'])(gmid10)) * W10
    
    # Parasitic capacitances at node N006 (cascode intermediate node)
    # N006 = drain M2, drain M8, source M5
    Cbd2 = float(scint.interp1d(M2['GMID'], M2['CBD'])(gmid2)) * W2
    Cgdo2 = float(scint.interp1d(M2['GMID'], M2['CGD0'])(gmid2)) * W2
    Cbd8 = float(scint.interp1d(M8['GMID'], M8['CBD'])(gmid8)) * W8
    Cgdo8 = float(scint.interp1d(M8['GMID'], M8['CGD0'])(gmid8)) * W8
    Cgs5_full = float(scint.interp1d(M5['GMID'], M5['CGS'])(gmid5)) * W5 * L5
    Cgso5_full = float(scint.interp1d(M5['GMID'], M5['CGS0'])(gmid5)) * W5
    Csb5 = float(scint.interp1d(M5['GMID'], M5['CBS'])(gmid5)) * W5 if 'CBS' in M5 else Cbd5 * 0.8  # Approximate if CBS not available
    
    # Capacitance at node N006
    C_N006 = Cbd2 + Cgdo2 + Cbd8 + Cgdo8 + Cgs5_full + Cgso5_full

    C1 = Cgs9 + Cgso9 + Cbd5 + Cgdo5 + Cbd3 + Cgdo3
    C2 = CL + Cgdo10 + Cbd10 + Cgdo9 + Cbd9
    
    # Pole at node N006 (cascode intermediate node)
    p_N006 = gm5 / (2*np.pi * C_N006)
    
    # Compute required pnd_to_wT_ratio accounting for p_N006
    # PM = 90° - arctan(wT/pnd) - arctan(wT/z) - arctan(wT/p_N006)
    # We need: arctan(wT/pnd) = 90° - PM - arctan(wT/z) - arctan(wT/p_N006)
    wT_estimate = 2*np.pi * fT_spec
    phase_from_zero = np.arctan(wT_estimate / (zero_to_wT_ratio * wT_estimate))  # arctan(1/zero_to_wT_ratio)
    phase_from_pN006 = np.arctan(wT_estimate / (2*np.pi * p_N006)) if p_N006 > 0 else 0
    
    available_phase_for_pnd = np.pi/2 - PM_spec*np.pi/180 - phase_from_zero - phase_from_pN006
    
    if available_phase_for_pnd > 0.01:  # Ensure positive and reasonable
        pnd_to_wT_ratio = 1 / np.tan(available_phase_for_pnd)
    else:
        pnd_to_wT_ratio = pnd_to_wT_ratio_init  # Fallback to initial estimate

    # Compute required Cf value for sufficient ratio pnd / wT (for phase margin)
    # Same equation as in the lecture slides, but with updated pnd_to_wT_ratio
    Cf_new = pnd_to_wT_ratio / 2 / zero_to_wT_ratio * (C1+C2 + np.sqrt((C1+C2)**2 + 4*zero_to_wT_ratio/pnd_to_wT_ratio * C1*C2))

    error = abs((Cf_new - Cf)/Cf_new)
    Cf = Cf_new
    
    iteration += 1

# Part of design flow that does not need to be integrated into the iterative process
vsg1 = float(scint.interp1d(M1['GMID'], M1['VGS'])(gmid1))
vsg2 = vsg1
vsg3 = float(scint.interp1d(M3['GMID'], M3['VGS'])(gmid3))
vsg4 = vsg3
vgs5 = float(scint.interp1d(M5['GMID'], M5['VGS'])(gmid5))
vgs6 = vgs5
vgs8 = float(scint.interp1d(M8['GMID'], M8['VGS'])(gmid8))
vgs7 = vgs8
vsg9 = float(scint.interp1d(M9['GMID'], M9['VGS'])(gmid9))
vgs10 = float(scint.interp1d(M10['GMID'], M10['VGS'])(gmid10))
vsg11 = float(scint.interp1d(M11['GMID'], M11['VGS'])(gmid11))
vsg12 = vsg11
vgs13 = float(scint.interp1d(M13['GMID'], M13['VGS'])(gmid13))
vgs14 = float(scint.interp1d(M14['GMID'], M14['VGS'])(gmid14))
vgs15 = float(scint.interp1d(M15['GMID'], M15['VGS'])(gmid15))

vdsat1 = - float(scint.interp1d(M1['GMID'], M1['VDSAT'])(gmid1))
vdsat3 = - float(scint.interp1d(M3['GMID'], M3['VDSAT'])(gmid3))
vdsat5 = float(scint.interp1d(M5['GMID'], M5['VDSAT'])(gmid5))
vdsat6 = float(scint.interp1d(M6['GMID'], M6['VDSAT'])(gmid6))
vdsat7 = float(scint.interp1d(M7['GMID'], M7['VDSAT'])(gmid7))
vdsat8 = float(scint.interp1d(M8['GMID'], M8['VDSAT'])(gmid8))
vdsat9 = - float(scint.interp1d(M9['GMID'], M9['VDSAT'])(gmid9))
vdsat10 = float(scint.interp1d(M10['GMID'], M10['VDSAT'])(gmid10))
vdsat11 = - float(scint.interp1d(M11['GMID'], M11['VDSAT'])(gmid11))
vdsat13 = float(scint.interp1d(M13['GMID'], M13['VDSAT'])(gmid13))
vdsat14 = float(scint.interp1d(M14['GMID'], M14['VDSAT'])(gmid14))
vdsat15 = float(scint.interp1d(M15['GMID'], M15['VDSAT'])(gmid15))

vea1 = float(scint.interp1d(M1['GMID'], M1['VEA'])(gmid1)) # only valid because techno extraction was done with same L
vea3 = float(scint.interp1d(M3['GMID'], M3['VEA'])(gmid3)) # only valid because techno extraction was done with same L
vea5 = float(scint.interp1d(M5['GMID'], M5['VEA'])(gmid5)) # only valid because techno extraction was done with same L
vea6 = float(scint.interp1d(M6['GMID'], M6['VEA'])(gmid6)) # only valid because techno extraction was done with same L
vea8 = float(scint.interp1d(M8['GMID'], M8['VEA'])(gmid8)) # only valid because techno extraction was done with same L
vea9 = float(scint.interp1d(M9['GMID'], M9['VEA'])(gmid9)) # only valid because techno extraction was done with same L
vea10 = float(scint.interp1d(M10['GMID'], M10['VEA'])(gmid10)) # only valid because techno extraction was done with same L

IBIAS = ID15

vin_max = VDD - vdsat11 - vsg1
vin_min = vdsat7 + vdsat1 - vsg1

vout_max = VDD - vdsat9
vout_min = vdsat10

area = W1*L1 + W2*L2 + W3*L3 + W4*L4 + W5*L5 + W6*L6 + W7*L7 + W8*L8 + W9*L9 + W10*L10 + W11*L11 + W12*L12 + W13*L13 + W14*L14 + W15*L15
power = VDD * (2*ID1 + ID3 + ID3 + ID9 + ID12 + ID15)

# =============================================================================
# Theoretical zpk
# =============================================================================

# Link with lecture slides
#g1 = ID1/vea1 + ID3/vea3
#g5 = ID5/vea5 + ID6/vea6

# Low-frequency gain
vea_eq_s1 = vea1
vea_eq_s2 = 1/(1/vea3 + 1/(gm5 * vea5 * vea8 + vea5 + vea8))
vea_eq_s3 = 1/(1/vea9 + 1/vea10)
gain_s1 = gmid1 * vea_eq_s1
gain_s2 = gmid5 * vea_eq_s2
gain_s3 = gmid10 * vea_eq_s3
gain = gain_s1 * gain_s2 * gain_s3

# Poles, zeros and GBW of Miller
GBW = gm5 / (2*np.pi * Cf)
pd = GBW / gain
pnd = gm10 * Cf / (2*np.pi * (C1*C2 + (C1+C2)*Cf))
z = gm10 / (2*np.pi * Cf)

# Additional pole at node N006 (already computed in loop, but recalculate for final values)
p_N006_final = gm5 / (2*np.pi * C_N006)

# =============================================================================
# Transfer function plot
# =============================================================================

fmin = 1e0; fmax = 1e8
f = np.logspace(np.log10(fmin),np.log10(fmax),1000)
# Transfer function now includes the parasitic pole at N006
H = gain * (1 - 1j * f/z) / ((1 + 1j * f/pd) * (1 + 1j * f/pnd) * (1 + 1j * f/p_N006_final))

Hdb = 20*np.log10(np.abs(H))
Hangle = np.angle(H, deg=True)

fig,ax = plt.subplots(2, 1, figsize=(18,10),sharex=True)
ax[0].semilogx(f, Hdb,linewidth=3, label='With p_N006')
ax[1].semilogx(f, Hangle,linewidth=3)
ax[0].grid(True, which='major')
ax[0].grid(True, which='minor', alpha=0.3)
ax[1].grid(True, which='major')
ax[1].grid(True, which='minor', alpha=0.3)
ax[1].set_xlim((fmin,fmax))
ax[1].set_xlabel("Frequency [Hz]",fontsize=20)
ax[0].set_ylabel("Magnitude [dB]",fontsize=20)
ax[1].set_ylabel("Phase [°]",fontsize=20)
ax[1].set_ylim(-180,180)
ax[0].set_ylim(-40,100)

arg_PM = np.argmin(np.abs(Hdb - 0))
PM = Hangle[arg_PM] + 180
f_PM = f[arg_PM]
ax[0].axhline(0, ls='--', color='grey')
ax[0].axvline(f_PM, ls='--', color='grey')
ax[1].axvline(f_PM, ls='--', color='grey')
ax[1].axhline(PM-180, ls='--', color='grey')
for i in range(2):
    ax[i].axvline(pd, ls='--', color='red')
    ax[i].axvline(pnd, ls='--', color='red')
    ax[i].axvline(z, ls='--', color='red')
plt.show()

# =============================================================================
# Print result
# =============================================================================

sr_int = 2 * ID1 / Cf
sr_ext = ID5 / CL


print('===================== Miller OTA sizing =====================')
print('M1/M2: W = {:2.2f} um; L = {:2.2f} um'.format(W1*1e6,L1*1e6))
print('M3/M4: W = {:2.2f} um; L = {:2.2f} um'.format(W3*1e6,L3*1e6))
print('M5/M6: W = {:2.2f} um; L = {:2.2f} um'.format(W5*1e6,L5*1e6))
print('M7/M8: W = {:2.2f} um; L = {:2.2f} um'.format(W7*1e6,L7*1e6))
print('M9:    W = {:2.2f} um; L = {:2.2f} um'.format(W9*1e6,L9*1e6))
print('M10:   W = {:2.2f} um; L = {:2.2f} um'.format(W10*1e6,L10*1e6))
print('M11:   W = {:2.2f} um; L = {:2.2f} um'.format(W11*1e6,L11*1e6))
print('M12:   W = {:2.2f} um; L = {:2.2f} um'.format(W12*1e6,L12*1e6))
print('M13:   W = {:2.2f} um; L = {:2.2f} um'.format(W13*1e6,L13*1e6))
print('M14:   W = {:2.2f} um; L = {:2.2f} um'.format(W14*1e6,L14*1e6))
print('M15:   W = {:2.2f} um; L = {:2.2f} um'.format(W15*1e6,L15*1e6))
print('IBIAS: {:2.2f} uA'.format(IBIAS*1e6))
print('VIN MAX:   {:2.2f} V'.format(vin_max))
print('VIN MIN:   {:2.2f} V'.format(vin_min))
print('VDSAT1:   {:2.2f} V'.format(vdsat1))
print('VIN:   {:2.2f} V'.format(vin_min + (vin_max - vin_min)/2))
print('VOUT:  {:2.2f} V'.format(vout_min + (vout_max - vout_min)/2))
print('Cf:    {:2.2f} pF'.format(Cf*1e12))
print('')
print('===================== Miller OTA estimated performance =====================')
print('Gain: {:2.2f} dB'.format(20*np.log10(gain)))
print('fT @CL={:2.1f} pF: {:2.3f} MHz'.format(CL*1e12,GBW/1e6))
print('Dominant pole: {:2.3f} kHz'.format(pd/1e3))
print('Non-dominant pole: {:2.3f} MHz'.format(pnd/1e6))
print('Zero: {:2.3f} MHz'.format(z/1e6))
print('Phase Margin: {:2.1f}°'.format(PM))
print('Miller capacitance Cf: {:2.3f} pF'.format(Cf*1e12))
print('Power consumption: {:2.3f} µW'.format(power*1e6))
print('Active area: {:2.3f} µm2'.format(area*1e12))
print('SRint: {:2.3f} V/ms'.format(sr_int/1e3))
print('SRext: {:2.3f} V/ms'.format(sr_ext/1e3))
print('VIN: min/max = {:2.2f}/{:2.2f} V'.format(vin_min,vin_max))
print('VOUT: min/max = {:2.2f}/{:2.2f} V'.format(vout_min,vout_max))

# =============================================================================
# Export sizing to sizing.cir for Eldo simulation
# =============================================================================

vin_dc = vin_min + (vin_max - vin_min)/2

sizing_content = f"""* Auto-generated sizing parameters from Python script
* DO NOT EDIT MANUALLY - Run LELEC2650-OTA-cascode-Miller-Design.py to update
* Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
*-----------------------------------------------

*** Transistor dimensions

.param W1 = {W1*1e6:.2f}u
.param L1 = {L1*1e6:.0f}u
.param W2 = W1
.param L2 = L1

.param W3 = {W3*1e6:.2f}u
.param L3 = {L3*1e6:.0f}u
.param W4 = W3
.param L4 = L3

.param W5 = {W5*1e6:.2f}u
.param L5 = {L5*1e6:.0f}u
.param W6 = W5
.param L6 = L5

.param W7 = {W7*1e6:.2f}u
.param L7 = {L7*1e6:.0f}u
.param W8 = W7
.param L8 = L7

.param W9 = {W9*1e6:.2f}u
.param L9 = {L9*1e6:.0f}u
.param W10 = {W10*1e6:.2f}u
.param L10 = {L10*1e6:.0f}u
.param W11 = {W11*1e6:.2f}u
.param L11 = {L11*1e6:.0f}u
.param W12 = {W12*1e6:.2f}u
.param L12 = {L12*1e6:.0f}u
.param W13 = {W13*1e6:.2f}u
.param L13 = {L13*1e6:.0f}u
.param W14 = {W14*1e6:.2f}u
.param L14 = {L14*1e6:.0f}u
.param W15 = {W15*1e6:.2f}u
.param L15 = {L15*1e6:.0f}u

.param Cf_val = {Cf*1e12:.2f}p

*** Bias point

.param VIN = {vin_dc:.2f}
.param IBIAS = {IBIAS*1e6:.2f}u
"""

with open('sizing.cir', 'w') as f:
    f.write(sizing_content)

print('')
print('>>> sizing.cir updated successfully!')