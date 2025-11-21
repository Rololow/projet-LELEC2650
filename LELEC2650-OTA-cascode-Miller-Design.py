# -*- coding: utf-8 -*-
"""
Miller OTA 1st order Python sizing

65nm CMOS technology (TSMC)

Sylvain Favresse, 2025

netlist : 
pair diff :
M1 N003 VIN- N005 N005 PMOS
M2 N004 VIN+ N003 N003 PMOS
M3 N002 N001 VDD VDD PMOS
M4 N001 N001 VDD VDD PMOS
M5 N004 VBIAS1 N002 N002 NMOS
M6 N005 VBIAS1 N001 N001 NMOS
M7 0 VBIAS2 N005 N005 NMOS
M8 0 VBIAS2 N004 N004 NMOS
M9 OUT VBIAS3 VDD VDD PMOS
M10 0 N002 OUT OUT NMOS
C1 OUT N002 Cm
C2 OUT 0 CL
M11 N003 VBIAS3 VDD VDD PMOS
M12 VBIAS3 VBIAS3 VDD VDD PMOS
M13 0 VBIAS2 VBIAS3 VBIAS3 NMOS
M14 0 VBIAS2 VBIAS2 VBIAS2 NMOS
M15 VBIAS2 VBIAS1 VBIAS1 VBIAS1 NMOS
IB VDD VBIAS1 I
.model NMOS NMOS
.model PMOS PMOS
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

fT_spec = 1e6
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

gmid1 = 18; gmid2 = gmid1
gmid3 = 12; gmid4 = gmid3; gmid5 = gmid3; gmid6 = gmid3
gmid8 = 8; gmid7 = gmid8
gmid9 = 10; gmid10 = gmid9
gmid11 = 10; gmid12 = gmid11; gmid13 = gmid11; gmid14 = gmid11; gmid15 = gmid11

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
    GBW = gm1 / (2*pi*Cf)
    wT = gm1 / Cf
    Av0 = (gm1*gm5) / (g1 * g5)
    pd = - (g1*g5)/(gm5*Cf)
    pnd = - (gm5 * Cf) / (C1*C2+(C1+C2)*Cf)
    z = gm5 / Cf
    PM = 90° - arctan(wT/pnd) - arctan(wT/z)
    
The sizing of Cf is done based on a desired PM, that depends on the parasitic
capacitors. Those parasitics can only be evaluated when the transistor sizes are known.
We thus use an iterative design flow where we update the values of the parasitic
capacitances, until the required value of Cf stabilizes.
"""

pnd_to_wT_ratio = 1 / np.tan(np.pi/2 - PM_spec*np.pi/180 - np.arctan(1/zero_to_wT_ratio)) # 2.2 with 60° PM (see lecture slides)

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
    Cgs10 = float(scint.interp1d(M10['GMID'], M10['CGS'])(gmid10)) * W10 * L10
    Cgso10 = float(scint.interp1d(M10['GMID'], M10['CGS0'])(gmid10)) * W10
    Cbd10 = float(scint.interp1d(M10['GMID'], M10['CBD'])(gmid10)) * W10
    Cgdo10 = float(scint.interp1d(M10['GMID'], M10['CGD0'])(gmid10)) * W10

    C1 = Cgs10 + Cgso10 + Cbd5 + Cgdo5 + Cbd3 + Cgdo3
    C2 = CL + Cgdo10 + Cbd10 + Cgdo9 + Cbd9

    # Compute required Cf value for sufficient ratio pnd / wT (for phase margin)
    # Same equation as in the lecture slides
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
vdsat3 = float(scint.interp1d(M3['GMID'], M3['VDSAT'])(gmid3))
vdsat5 = float(scint.interp1d(M5['GMID'], M5['VDSAT'])(gmid5))
vdsat6 = - float(scint.interp1d(M6['GMID'], M6['VDSAT'])(gmid6))
vdsat7 = - float(scint.interp1d(M7['GMID'], M7['VDSAT'])(gmid7))
vea1 = float(scint.interp1d(M1['GMID'], M1['VEA'])(gmid1)) # only valid because techno extraction was done with same L
vea3 = float(scint.interp1d(M3['GMID'], M3['VEA'])(gmid3)) # only valid because techno extraction was done with same L
vea5 = float(scint.interp1d(M5['GMID'], M5['VEA'])(gmid5)) # only valid because techno extraction was done with same L
vea6 = float(scint.interp1d(M6['GMID'], M6['VEA'])(gmid6)) # only valid because techno extraction was done with same L
vea8 = float(scint.interp1d(M8['GMID'], M8['VEA'])(gmid8)) # only valid because techno extraction was done with same L
vea9 = float(scint.interp1d(M9['GMID'], M9['VEA'])(gmid9)) # only valid because techno extraction was done with same L
vea10 = float(scint.interp1d(M10['GMID'], M10['VEA'])(gmid10)) # only valid because techno extraction was done with same L

IBIAS = ID15

#vin_max = VDD - vdsat7 - vsg1
#vin_min = vgs3 + vdsat1 - vsg1

#vout_max = VDD - vdsat6
#vout_min = vdsat5

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

# =============================================================================
# Transfer function plot
# =============================================================================

fmin = 1e0; fmax = 1e8
f = np.logspace(np.log10(fmin),np.log10(fmax),1000)
H = gain * (1 - 1j * f/z) / ((1 + 1j * f/pd) * (1 + 1j * f/pnd))

Hdb = 20*np.log10(np.abs(H))
Hangle = np.angle(H, deg=True)

fig,ax = plt.subplots(2, 1, figsize=(18,10),sharex=True)
ax[0].semilogx(f, Hdb,linewidth=3)
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
print('M5:    W = {:2.2f} um; L = {:2.2f} um'.format(W5*1e6,L5*1e6))
print('M6:    W = {:2.2f} um; L = {:2.2f} um'.format(W6*1e6,L6*1e6))
print('M7:    W = {:2.2f} um; L = {:2.2f} um'.format(W7*1e6,L7*1e6))
print('M8:    W = {:2.2f} um; L = {:2.2f} um'.format(W8*1e6,L8*1e6))
print('IBIAS: {:2.2f} uA'.format(IBIAS*1e6))
#print('VIN:   {:2.2f} V'.format(VDD-vsg7-vsg1))
#print('VOUT:  {:2.2f} V'.format(VDD-vsg6))
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
#print('VIN: min/max = {:2.2f}/{:2.2f} V'.format(vin_min,vin_max))
#print('VOUT: min/max = {:2.2f}/{:2.2f} V'.format(vout_min,vout_max))