"""
DC Characteristics and Analyses


65 nm CMOS technology (TSMC)
nch_lvt_mac with W = 1 µm and L = 1 µm

ELEC2650 - TP 2 - Part 1 "Analyses of the DC characteristics"

Settings

Sylvain Favresse, 2022
From Leopold Van Brandt's work in MATLAB (2021)

"""
import numpy as np

global peta,tera,giga,mega,kilo,milli,micro,nano,pico,femto,ato
peta = 1e15
tera = 1e12
giga = 1e9
mega = 1e6
kilo = 1e3
milli = 1e-3
micro = 1e-6
nano = 1e-9
pico = 1e-12
femto = 1e-15
ato = 1e-18

global VDD
VDD = 1.2

row_headings = ['VGS','GMID','IN','CGS','CGS0','CGD0','CBD','VEA','VDSAT','GD']

def read_txt(filename,headerlines=2):
    try: 
        file = open(filename, 'r')
    except OSError:
        print(f"Could not open file: {filename}")
        return
    with file:
        rl = file.readlines()[2:]
        l = len(rl)
        
        ncols = len(rl[0].split())
        if ncols != len(row_headings):
            print(f"Inconsistent number of columns in {filename}")
            return
        
        struct = {}
        for col in range(ncols):
            struct[row_headings[col]] = np.zeros(l)
            
        for i in range(l):
            line = rl[i].split()
            
            for col in range(ncols):
                struct[row_headings[col]][i] = float(line[col])
        
        # For interpolation, a monotonic gm/ID function is necessary
        # First points that make the function non-monotonic are suppressed
        struct = remove_non_monotonic(struct, struct['GMID'])
        
        return struct
    
def remove_non_monotonic(struct, mono_array):
    idx = np.argmax(mono_array)
    
    for col in range(len(row_headings)):
        struct[row_headings[col]] = struct[row_headings[col]][idx:] 
        
    return struct