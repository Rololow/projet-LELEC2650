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



def load_two_column(path):
    """Load two-column ASCII file produced by Eldo .printfile format=data.
    Returns (x, y) arrays or (None, None) if file missing or no numeric rows.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    rows = []
    single_col = []
    with open(path, 'r') as fh:
        print(f"[debug] load_two_column: opening {path}")
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            # skip DATA header/footer tokens
            if line.startswith('.DATA') or line.startswith('.ENDDATA'):
                continue
            # lines may start with '+' continuation
            if line.startswith('+'):
                line = line[1:].strip()
            parts = line.split()
            if not parts:
                continue
            # try parse two columns first
            if len(parts) >= 2:
                try:
                    a = float(parts[0])
                    b = float(parts[1])
                    rows.append((a, b))
                    continue
                except ValueError:
                    pass
            # try parse single numeric value (common for some .printfile TRAN exports)
            try:
                v = float(parts[-1])
                single_col.append(v)
            except ValueError:
                # non-numeric header lines are ignored in strict mode
                continue

    if rows:
        data = np.array(rows)
        print(f"[debug] load_two_column: parsed {data.shape[0]} rows (2-col) from {path}")
        return data[:, 0], data[:, 1]
    if rows:
        data = np.array(rows)
        print(f"[debug] load_two_column: parsed {data.shape[0]} rows (2-col) from {path}")
        return data[:, 0], data[:, 1]
    if single_col:
        y = np.array(single_col)
        lower = path.lower()
        # If this is a noise export, construct a frequency vector from TB_NOISE.cir
        if 'noise' in lower or 'onoise' in lower or 'inoise' in lower:
            fmin, fmax = guess_freq_range_from_tb('TB_NOISE.cir')
            if fmin is None or fmax is None:
                fmin, fmax = 1.0, 1e9
            x = np.logspace(np.log10(fmin), np.log10(fmax), y.size)
            print(f"[debug] load_two_column: parsed {y.size} single-column noise values from {path}")
            return x, y
        # otherwise treat as time series and try to guess time step from TB_SR.cir
        dt = guess_time_step_from_tb('TB_SR.cir')
        if dt is None:
            # last resort: unit index
            x = np.arange(y.size)
        else:
            x = np.arange(y.size) * dt
        print(f"[debug] load_two_column: parsed {y.size} single-column time values from {path}, dt={dt}")
        return x, y
    # Strict mode: no valid data found
    raise ValueError(f"No two-column numeric rows found in {path}")


def parse_time_string(s):
    """Parse a time string like '10n' or '1u' and return seconds as float."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    # strip surrounding quotes if present
    if (s[0] == s[-1]) and s[0] in ('"', "'"):
        s = s[1:-1]
    # suffix multipliers
    suf = s[-1].lower()
    mult = 1.0
    if suf in 'pnuµu':
        if suf == 'p':
            mult = 1e-12
        elif suf == 'n':
            mult = 1e-9
        elif suf in ('u', 'µ'):
            mult = 1e-6
    elif suf == 'm':
        mult = 1e-3
    elif suf == 'k':
        mult = 1e3
    elif suf == 'g':
        mult = 1e9
    # if last char is a letter, separate
    try:
        if suf.isalpha():
            val = float(s[:-1]) * mult
        else:
            val = float(s)
        return float(val)
    except Exception:
        return None


def guess_time_step_from_tb(tb_path):
    """Look for `param t_step =` or `.TRAN` in the TB netlist to infer timestep (seconds)."""
    if not os.path.exists(tb_path):
        return None
    try:
        with open(tb_path, 'r') as fh:
            for line in fh:
                ln = line.strip()
                # param t_step = 10n
                if ln.lower().startswith('.param') and 't_step' in ln:
                    parts = ln.split()
                    for i, p in enumerate(parts):
                        if 't_step' in p:
                            # next token may be '=' or the value
                            if '=' in p:
                                _, val = p.split('=', 1)
                                dt = parse_time_string(val)
                                if dt:
                                    return dt
                            elif i+1 < len(parts) and parts[i+1] == '=' and i+2 < len(parts):
                                dt = parse_time_string(parts[i+2])
                                if dt:
                                    return dt
                # .TRAN t_step t_stop
                if ln.lower().startswith('.tran'):
                    parts = ln.split()
                    if len(parts) >= 2:
                        # parts[1] might be timestep
                        dt = parse_time_string(parts[1])
                        if dt:
                            return dt
    except Exception:
        return None
    return None


def find_file_ci(filename):
    """Find a file in the current directory case-insensitively.
    Returns the actual filename if found, else None.
    """
    if os.path.exists(filename):
        print(f"[debug] find_file_ci: exact match found: {filename}")
        return filename
    target = filename.lower()
    for f in os.listdir('.'):
        if f.lower() == target:
            print(f"[debug] find_file_ci: case-insensitive match: {f}")
            return f
    return None


def guess_freq_range_from_tb(tb_path):
    """Look for `.param fmin =` / `.param fmax =` or `.AC` line in the TB to infer frequency range."""
    if not os.path.exists(tb_path):
        return None, None
    fmin = None
    fmax = None
    try:
        with open(tb_path, 'r') as fh:
            for line in fh:
                ln = line.strip()
                low = ln.lower()
                if low.startswith('.param') and 'fmin' in low:
                    parts = ln.split()
                    for i, p in enumerate(parts):
                        if 'fmin' in p:
                            if '=' in p:
                                _, val = p.split('=', 1)
                                fmin = parse_time_string(val)
                            elif i+2 < len(parts) and parts[i+1] == '=':
                                fmin = parse_time_string(parts[i+2])
                if low.startswith('.param') and 'fmax' in low:
                    parts = ln.split()
                    for i, p in enumerate(parts):
                        if 'fmax' in p:
                            if '=' in p:
                                _, val = p.split('=', 1)
                                fmax = parse_time_string(val)
                            elif i+2 < len(parts) and parts[i+1] == '=':
                                fmax = parse_time_string(parts[i+2])
                if low.startswith('.ac'):
                    parts = ln.split()
                    # .AC dec 100 fmin fmax
                    if len(parts) >= 4:
                        try:
                            fmin = float(parts[-2])
                            fmax = float(parts[-1])
                        except Exception:
                            pass
        return fmin, fmax
    except Exception:
        return None, None


def load_complex_printfile(path, fmin=1.0, fmax=1e9):
    """Load Eldo .printfile that exported complex waveform columns (WR, WI).
    Reconstruct frequency vector (log spaced) from number of samples and fmin/fmax.
    Returns (freq, mag_dB) or (None, None) if not applicable.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    rows = []
    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # data lines may start with '+'
            if line.startswith('+'):
                line = line[1:].strip()
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                a = float(parts[0])
                b = float(parts[1])
            except ValueError:
                continue
            rows.append((a, b))
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    data = np.array(rows)
    col0 = data[:, 0]
    col1 = data[:, 1]
    # If header indicates WR/WI (real/imag), compute magnitude and phase and reconstruct freq
    # Look at first line of file to detect WR/WI header
    with open(path, 'r') as fh:
        header = None
        for line in fh:
            line = line.strip()
            if not line:
                continue
            header = line
            break
    if header and ('WR(' in header.upper() and 'WI(' in header.upper()):
        # columns are real and imaginary parts of complex gain
        wr = col0
        wi = col1
        # Need frequency axis: read from TB_AC.cir params
        fmin_tb, fmax_tb = guess_freq_range_from_tb('TB_AC.cir')
        # If TB_AC.cir uses param tokens (ac_dec fmin fmax), try to read params from the included netlist
        if fmin_tb is None or fmax_tb is None:
            fmin_tb, fmax_tb = guess_freq_range_from_tb('CascodeMillerOTA.cir')
        if fmin_tb is None or fmax_tb is None:
            raise ValueError(f"Cannot reconstruct frequency axis for {path}: missing fmin/fmax in TB_AC.cir or CascodeMillerOTA.cir")
        n = len(wr)
        freq = np.logspace(np.log10(fmin_tb), np.log10(fmax_tb), n)
        mag = np.sqrt(wr**2 + wi**2)
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-300))
        phase_deg = np.angle(wr + 1j*wi, deg=True)
        return freq, mag_db, phase_deg

    # Otherwise assume file already contains (frequency, magnitude)
    # Strict: require first column to be a positive, strictly increasing frequency axis
    if not (np.all(np.diff(col0) > 0) and col0[0] > 0):
        raise ValueError(f"File {path} does not contain a valid frequency column")
    return col0, col1, None


def load_ac_csv(path):
    """Load AC.csv produced by some exporters: columns like Index,X Axis,W(AV)
    where W(AV) is complex string 'real-imagj'. Returns (freq, mag_db, phase_deg).
    Strict: file must exist and contain numeric rows.
    """
    import csv
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    freqs = []
    mags = []
    phases = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        # skip header if present
        header = next(reader, None)
        # if header contains non-numeric in second col, treat it as header
        def is_number(s):
            try:
                float(s)
                return True
            except Exception:
                return False
        # If header looks like data row, process it; otherwise continue with reader
        if header is None:
            raise ValueError(f"Empty AC CSV: {path}")
        # determine whether header is header or data by checking second field
        if len(header) >= 2 and is_number(header[1]):
            # header is actually data, process it
            row0 = header
            rows = [row0] + list(reader)
        else:
            rows = list(reader)
        for r in rows:
            if not r:
                continue
            # common formats: Index, X Axis, W(AV)
            if len(r) < 3:
                continue
            try:
                f = float(r[1])
                # parse complex from third column; ensure 'j' present
                s = r[2].replace('i', 'j')
                # remove spaces
                s = s.replace(' ', '')
                c = complex(s)
            except Exception:
                continue
            mag = abs(c)
            phase = np.angle(c, deg=True)
            freqs.append(f)
            mags.append(20.0 * np.log10(max(mag, 1e-300)))
            phases.append(phase)
    if not freqs:
        raise ValueError(f"No numeric rows found in {path}")
    return np.array(freqs), np.array(mags), np.array(phases)

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
    Données mesurées pour OTA Miller
    """
    # Require Eldo exported differential gain (diff_gain.txt) or complex WR/WI export
    diff_f, diff_dB = None, None
    # Require `AC.csv` export (complex W(AV)). TXT files are deprecated and no longer used.
    ac_fn = find_file_ci('AC.csv') or find_file_ci('ac.csv')
    if not ac_fn:
        raise FileNotFoundError('Required CSV export not found: AC.csv (TXT files are deprecated)')
    diff_f, diff_dB, diff_phase = load_ac_csv(ac_fn)
    freq = np.array(diff_f)
    gain_dB = np.array(diff_dB)
    Av0_dB = np.max(gain_dB)
    
    # Estimate fT by locating where gain crosses 0 dB (interpolate if crossing exists)
    def compute_fT_from_gain(freq, gain_dB):
        signs = np.sign(gain_dB)
        crossings = np.where(signs[:-1] != signs[1:])[0]
        if crossings.size == 0:
            raise ValueError('gain curve does not cross 0 dB; cannot compute fT')
        i = crossings[0]
        f1, f2 = freq[i], freq[i+1]
        g1, g2 = gain_dB[i], gain_dB[i+1]
        if g2 != g1:
            fT_est = f1 + (0 - g1) * (f2 - f1) / (g2 - g1)
        else:
            fT_est = freq[i]
        return fT_est

    f_T = compute_fT_from_gain(freq, gain_dB)

    # Phase: try to load phase export if available (phase file name: diff_phase.txt)
    phase = diff_phase
    
    # If phase available, plot magnitude + phase; otherwise plot magnitude only
    if phase is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax2.semilogx(freq, phase, 'orange', label='Phase (deg)')
        ax2.set_ylabel('Phase (°)')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.grid(True, which='both', alpha=0.3)
        ax2.set_ylim(-360, 360)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Gain (magnitude)
    ax1.semilogx(freq, gain_dB, 'b-', label='Gain (dB)')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='0 dB')
    ax1.axvline(x=f_T, color='g', linestyle='--', alpha=0.5, label=f'fT = {f_T/1e3:.0f} kHz')
    ax1.set_ylabel('Gain (dB)')
    ax1.set_ylim(np.min(gain_dB)-5, np.max(gain_dB)+5)
    ax1.legend(loc='upper right')
    ax1.set_title(f'Bode Plot - OTA Cascode Miller (Av0={Av0_dB:.1f} dB, fT={f_T/1e3:.0f} kHz)')
    ax1.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{OUTPUT_DIR}/bode_plot.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {OUTPUT_DIR}/bode_plot.png")
    plt.show()

def plot_slew_rate(save=True):
    """
    Plot Slew Rate - TB_SR
    """
    # Prefer CSV time-domain exports: `sr_up.csv` / `sr_down.csv`.
    def load_time_csv(path):
        """Return (time, vout, vin)
        If `vin` not present, return None for vin. If only values present, returns (None, values, None).
        """
        if not os.path.exists(path):
            return None, None, None
        import csv
        try:
            with open(path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None)
                if header:
                    hdr = [h.strip().lower() for h in header]
                    # find time column (X Axis / Time)
                    time_idx = None
                    for cand in ('x axis', 'xaxis', 'time', 't'):
                        if cand in hdr:
                            time_idx = hdr.index(cand)
                            break
                    # find Vout column
                    vout_idx = None
                    for cand in ('v(out)', 'vout', 'v(out)', 'v(out)'):
                        if cand in hdr:
                            vout_idx = hdr.index(cand)
                            break
                    # find Vin column (optional)
                    vin_idx = None
                    for cand in ('v(inp)', 'v(inp)', 'vin', 'v(in)'):
                        if cand in hdr:
                            vin_idx = hdr.index(cand)
                            break
                    if time_idx is not None and vout_idx is not None:
                        xs = []
                        vouts = []
                        vins = []
                        for r in reader:
                            if not r or len(r) <= max(time_idx, vout_idx):
                                continue
                            try:
                                xs.append(float(r[time_idx]))
                                vouts.append(float(r[vout_idx]))
                                if vin_idx is not None and len(r) > vin_idx:
                                    try:
                                        vins.append(float(r[vin_idx]))
                                    except Exception:
                                        vins.append(np.nan)
                                else:
                                    vins.append(np.nan)
                            except Exception:
                                continue
                        if xs:
                            vin_arr = np.array(vins) if any(not np.isnan(x) for x in vins) else None
                            return np.array(xs), np.array(vouts), vin_arr
                    # fallback: attempt numeric read via numpy
            data = np.genfromtxt(path, delimiter=',')
            if data.ndim == 2 and data.shape[1] >= 2:
                # assume columns: time, vout [, vin]
                t = data[:, 0]
                vout = data[:, 1]
                vin = data[:, 2] if data.shape[1] >= 3 else None
                return t, vout, vin
            if data.ndim == 1 and data.size >= 2:
                # single-column CSV values
                return None, data, None
        except Exception:
            return None, None, None
        return None, None, None

    # Accept combined `SR.csv` or per-direction `sr_up.csv` / `sr_down.csv`
    up_csv = find_file_ci('sr_up.csv') or find_file_ci('SR_UP.csv') or find_file_ci('SR.csv')
    down_csv = find_file_ci('sr_down.csv') or find_file_ci('SR_DOWN.csv')
    up_t, up_v, up_inp = (None, None, None)
    down_t, down_v, down_inp = (None, None, None)
    if up_csv:
        up_t, up_v, up_inp = load_time_csv(up_csv)
    if down_csv:
        down_t, down_v, down_inp = load_time_csv(down_csv)
    # If combined SR.csv provided as up_csv, use it
    if up_t is None and up_v is None and down_t is None and down_v is None:
        print('No SR CSV exports found (sr_up.csv / sr_down.csv / SR.csv) — skipping Slew Rate plot')
        return

    # Prefer combined trace if available
    if up_t is not None and up_v is not None:
        t = np.array(up_t)
        vout = np.array(up_v)
        vin = np.array(up_inp) if up_inp is not None else None
    elif down_t is not None and down_v is not None:
        t = np.array(down_t)
        vout = np.array(down_v)
        vin = np.array(down_inp) if down_inp is not None else None
    else:
        print('No SR traces available — skipping Slew Rate plot')
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t*1e6, vout, 'b-', label='Vout')
    if vin is not None:
        # secondary axis for input
        ax.plot(t*1e6, vin, 'k--', label='Vin')

    ax.set_xlabel('Time (µs)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Slew Rate - Measured from Eldo exports')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # Compute derivative dv/dt and extract SR+ (max) and SR- (min)
    try:
        dv_dt = np.gradient(vout, t)
        sr_plus = np.max(dv_dt)
        sr_minus = np.min(dv_dt)
        # annotate on plot
        ax.axvline(x=t[np.argmax(dv_dt)]*1e6, color='green', linestyle='--', alpha=0.6)
        ax.axvline(x=t[np.argmin(dv_dt)]*1e6, color='orange', linestyle='--', alpha=0.6)
        ax.text(0.02, 0.95, f'SR+ = {sr_plus*1e-6:.3f} V/µs\nSR- = {sr_minus*1e-6:.3f} V/µs', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        print(f"SR+ = {sr_plus:.6e} V/s ({sr_plus*1e-6:.3f} V/µs)")
        print(f"SR- = {sr_minus:.6e} V/s ({sr_minus*1e-6:.3f} V/µs)")
    except Exception as e:
        print(f"Warning: failed to compute slew rate from waveform: {e}")

    if save:
        plt.savefig(f'{OUTPUT_DIR}/slew_rate.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {OUTPUT_DIR}/slew_rate.png")
    plt.show()

def plot_noise(save=True):
    """
    Plot Noise Spectrum - TB_NOISE
    """
    # Prefer CSV noise exports: noise_out.csv / noise_in.csv
    def load_noise_csv(path):
        if not os.path.exists(path):
            return None, None
        try:
            data = np.genfromtxt(path, delimiter=',')
            if data.ndim == 2 and data.shape[1] >= 2:
                return data[:, 0], data[:, 1]
        except Exception:
            return None, None
        return None, None

    # try combined NOISE.csv first (contains ONOISE, INOISE)
    noise_combined = find_file_ci('NOISE.csv') or find_file_ci('noise.csv') or find_file_ci('NOISE.CSV')
    out_csv = find_file_ci('noise_out.csv') or find_file_ci('noise_out.CSV')
    in_csv = find_file_ci('noise_in.csv') or find_file_ci('noise_in.CSV')
    freq_out, out_noise = (None, None)
    freq_in, in_noise = (None, None)
    if noise_combined:
        # parse header-aware CSV
        import csv
        try:
            with open(noise_combined, newline='') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None)
                if header:
                    hdr = [h.strip().lower() for h in header]
                    # find columns
                    try:
                        f_idx = hdr.index('x axis') if 'x axis' in hdr else hdr.index('index') if 'index' in hdr else 0
                    except Exception:
                        f_idx = 1 if len(hdr) > 1 else 0
                    onoise_idx = None
                    inoise_idx = None
                    for name in ('onoise', 'o(noise)', 'onoise'):
                        if name in hdr:
                            onoise_idx = hdr.index(name)
                            break
                    for name in ('inoise', 'i(noise)', 'inoise'):
                        if name in hdr:
                            inoise_idx = hdr.index(name)
                            break
                    freqs = []
                    ons = []
                    ins = []
                    for r in reader:
                        if not r or len(r) <= f_idx:
                            continue
                        try:
                            freqs.append(float(r[f_idx]))
                        except Exception:
                            continue
                        if onoise_idx is not None and len(r) > onoise_idx:
                            try:
                                ons.append(float(r[onoise_idx]))
                            except Exception:
                                ons.append(np.nan)
                        else:
                            ons.append(np.nan)
                        if inoise_idx is not None and len(r) > inoise_idx:
                            try:
                                ins.append(float(r[inoise_idx]))
                            except Exception:
                                ins.append(np.nan)
                        else:
                            ins.append(np.nan)
                    if freqs:
                        freq_out = np.array(freqs); out_noise = np.array(ons)
                        freq_in = np.array(freqs); in_noise = np.array(ins)
        except Exception:
            pass
    # individual files fallback
    if out_csv and (freq_out is None):
        freq_out, out_noise = load_noise_csv(out_csv)
    if in_csv and (freq_in is None):
        freq_in, in_noise = load_noise_csv(in_csv)
    if freq_out is None and freq_in is None:
        print('No noise CSV exports found (NOISE.csv or noise_out.csv / noise_in.csv) — skipping Noise Spectrum plot')
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    if freq_out is not None:
        ax.loglog(freq_out, out_noise, 'r-', label='Output Noise (ONOISE)')
    if freq_in is not None:
        ax.loglog(freq_in, in_noise, 'b-', label='Input Noise (INOISE)')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Noise')
    ax.set_title('Noise Spectrum - Measured from Eldo export')
    ax.legend(loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(f'{OUTPUT_DIR}/noise_spectrum.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {OUTPUT_DIR}/noise_spectrum.png")
    plt.show()

def plot_cmrr_psrr(save=True):
    """
    Plot CMRR et PSRR vs frequency
    """
    # Strict CSV-only mode: require AC.csv (differential), CMRR.csv (common-mode).
    diff_csv = find_file_ci('AC.csv') or find_file_ci('ac.csv')
    cm_csv = find_file_ci('CMRR.csv') or find_file_ci('cmrr.csv') or find_file_ci('CM.csv') or find_file_ci('cm.csv')
    psrr_csv = find_file_ci('PSRR.csv') or find_file_ci('psrr.csv')
    if not diff_csv:
        raise FileNotFoundError('Required CSV export not found: AC.csv (diff). TXT files are deprecated')
    if not cm_csv:
        raise FileNotFoundError('Required CSV export not found: CMRR.csv (common-mode). TXT files are deprecated')

    diff_f, diff_mag, diff_phase = load_ac_csv(diff_csv)
    cm_f, cm_mag, cm_phase = load_ac_csv(cm_csv)

    # common freq grid
    freq = np.logspace(np.log10(min(np.min(diff_f), np.min(cm_f))), np.log10(max(np.max(diff_f), np.max(cm_f))), 1000)
    diff_interp = np.interp(freq, diff_f, diff_mag)
    cm_interp = np.interp(freq, cm_f, cm_mag)
    CMRR = diff_interp - cm_interp

    # PSRR is optional; if present, load it (CSV only) and compute PSRR but don't plot
    if psrr_csv:
        vdd_f, vdd_mag, vdd_phase = load_ac_csv(psrr_csv)
        # keep PSRR available if needed
        vdd_interp = np.interp(freq, vdd_f, vdd_mag)
        PSRR = diff_interp - vdd_interp

    # Combined plot: CMRR, PSRR (if present) and Adm (differential gain in dB) on a single dB axis
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogx(freq, CMRR, 'b-', label='CMRR (dB)')
        if psrr_csv:
            ax.semilogx(freq, PSRR, 'g-', label='PSRR (dB)')
        # Adm is the differential gain (diff_interp) already in dB
        ax.semilogx(freq, diff_interp, 'r--', label='Adm (dB)')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude / Rejection (dB)')
        ax.set_title('CMRR, PSRR and Adm (dB) vs Frequency')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_xlim(np.min(freq), np.max(freq))
        # auto y-limits with small margin
        y_all = [CMRR]
        if psrr_csv:
            y_all.append(PSRR)
        y_all.append(diff_interp)
        yconcat = np.hstack(y_all)
        ymin, ymax = np.nanmin(yconcat), np.nanmax(yconcat)
        yrange = max(1.0, ymax - ymin)
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)

        plt.tight_layout()
        if save:
            plt.savefig(f'{OUTPUT_DIR}/cmrr_psrr_adm.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {OUTPUT_DIR}/cmrr_psrr_adm.png")
        plt.show()
    except Exception as e:
        print(f"Warning: failed to create combined CMRR/PSRR/Adm plot: {e}")

def plot_monte_carlo_histograms(save=True):
    """
    Plot Monte Carlo histograms
    """
    # Read Monte Carlo exported statistics (expecting files: mc_Av0.txt, mc_fT.txt, mc_PM.txt, mc_offset.txt)
    # First check for a direct per-run raw file `mc_raw.txt` (columns: Av0, fT, Phase_Margin, V_error)
    raw_fn = find_file_ci('mc_raw.txt')
    raw_present = False
    if raw_fn:
        rows = []
        with open(raw_fn, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('.DATA') or line.startswith('.ENDDATA'):
                    continue
                if line.startswith('+'):
                    line = line[1:].strip()
                parts = line.split()
                nums = []
                for p in parts:
                    try:
                        nums.append(float(p))
                    except Exception:
                        pass
                if nums:
                    rows.append(nums)
        if rows:
            arr = np.array(rows)
            # Expect 4 columns: Av0, fT, Phase_Margin, V_error
            if arr.shape[1] >= 4:
                Av0_dB = arr[:, 0]
                fT = arr[:, 1]
                PM = arr[:, 2]
                V_error = arr[:, 3]
                raw_present = True
                print(f"[debug] plot_mc: found mc_raw.txt with shape {arr.shape}")

    # Support case-insensitive file names (some exports are MC_*.txt)
    def load_mc_file(base_name):
        fn = find_file_ci(base_name)
        if not fn:
            return None, None
        x, y = load_two_column(fn)
        # If loader returned (None, array) for single-column or multi-col summary,
        # try to parse numeric matrix from the file directly.
        if (x is None and isinstance(y, np.ndarray)):
            # y is a 1D array of values; treat as per-run samples
            return None, y
        if x is not None:
            return x, y
        # fallback: attempt to parse matrix rows (for summary files with multiple MEAS columns)
        try:
            rows = []
            with open(fn, 'r') as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith('.DATA') or line.startswith('.ENDDATA'):
                        continue
                    if line.startswith('+'):
                        line = line[1:].strip()
                    parts = line.split()
                    nums = []
                    for p in parts:
                        try:
                            nums.append(float(p))
                        except Exception:
                            pass
                    if nums:
                        rows.append(nums)
            if not rows:
                return None, None
            arr = np.array(rows)
            # If summary stats (single row with mean,std,min,max), return those as dict via y
            if arr.shape[0] >= 1 and arr.shape[1] >= 4:
                # return array of shape (4,) containing mean,std,min,max
                return None, arr[1] if arr.shape[0] > 1 else arr[0]
            # else flatten to 1D
            return None, arr.flatten()
        except Exception:
            return None, None

    if not raw_present:
        # Strict: require all three Monte Carlo summary files to exist
        av_fn = find_file_ci('mc_Av0.txt')
        ft_fn = find_file_ci('mc_fT.txt')
        pm_fn = find_file_ci('mc_PM.txt')
        voff_fn = find_file_ci('mc_offset.txt')
        if not av_fn or not ft_fn or not pm_fn:
            raise FileNotFoundError('Required Monte Carlo files missing: mc_Av0.txt / mc_fT.txt / mc_PM.txt')
        _, av_vals = load_mc_file(av_fn)
        _, ft_vals = load_mc_file(ft_fn)
        _, pm_vals = load_mc_file(pm_fn)
        _, voff_vals = load_mc_file(voff_fn) if voff_fn else (None, None)

    # If the files contain a single row of summary stats (mean,std,min,max), handle separately
    def is_summary(arr):
        return isinstance(arr, np.ndarray) and arr.size >= 4 and arr.ndim == 1 and arr.shape[0] <= 10

    # Prepare arrays for plotting
    if raw_present:
        # plotting using per-run arrays from mc_raw (Av0_dB, fT, PM, V_error already set)
        # detect if Av0 is linear magnitude and convert to dB
        try:
            if np.nanmean(np.abs(Av0_dB)) > 50:
                print('[debug] plot_mc: converting per-run Av0 linear->dB')
                Av0_dB = 20.0 * np.log10(np.abs(Av0_dB))
        except Exception:
            pass
        pass
    elif is_summary(av_vals) and is_summary(ft_vals) and is_summary(pm_vals):
        # av_vals contains [mean, std, min, max] in linear units (likely)
        Av0_mean_lin, Av0_std_lin, Av0_min_lin, Av0_max_lin = av_vals[0], av_vals[1], av_vals[2], av_vals[3]
        # convert to dB
        if Av0_mean_lin > 0:
            Av0_mean = 20.0 * np.log10(Av0_mean_lin)
            # approximate std in dB by converting (mean+std) difference
            try:
                Av0_std = 20.0 * np.log10(Av0_mean_lin + Av0_std_lin) - Av0_mean
            except Exception:
                Av0_std = 0.0
            Av0_min = 20.0 * np.log10(max(Av0_min_lin, 1e-30))
            Av0_max = 20.0 * np.log10(max(Av0_max_lin, 1e-30))
        else:
            Av0_mean, Av0_std, Av0_min, Av0_max = Av0_mean_lin, Av0_std_lin, Av0_min_lin, Av0_max_lin

        fT_mean_lin, fT_std_lin, fT_min_lin, fT_max_lin = ft_vals[0], ft_vals[1], ft_vals[2], ft_vals[3]
        fT_mean, fT_std, fT_min, fT_max = fT_mean_lin, fT_std_lin, fT_min_lin, fT_max_lin
        PM_mean_lin, PM_std_lin, PM_min_lin, PM_max_lin = pm_vals[0], pm_vals[1], pm_vals[2], pm_vals[3]
        PM_mean, PM_std, PM_min, PM_max = PM_mean_lin, PM_std_lin, PM_min_lin, PM_max_lin
        V_error = voff_vals if voff_vals is not None else np.array([0.0])
        # Create representative plots summarizing mean±std and min/max
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # Av0 bar
        ax = axes[0, 0]
        ax.bar([0], [Av0_mean], yerr=[Av0_std], color='steelblue', capsize=8)
        ax.set_xticks([0]); ax.set_xticklabels(['Av0'])
        ax.set_ylabel('Av0 (dB)')
        ax.set_title(f'Av0 (µ={Av0_mean:.2f} dB, σ={Av0_std:.2f} dB)')
        # fT bar
        ax = axes[0, 1]
        ax.bar([0], [fT_mean/1e3], yerr=[fT_std/1e3], color='forestgreen', capsize=8)
        ax.set_xticks([0]); ax.set_xticklabels(['fT'])
        ax.set_ylabel('fT (kHz)')
        ax.set_title(f'fT (µ={fT_mean/1e3:.0f} kHz, σ={fT_std/1e3:.0f} kHz)')
        # PM bar
        ax = axes[1, 0]
        ax.bar([0], [PM_mean], yerr=[PM_std], color='coral', capsize=8)
        ax.set_xticks([0]); ax.set_xticklabels(['PM'])
        ax.set_ylabel('Phase Margin (°)')
        ax.set_title(f'Phase Margin (µ={PM_mean:.2f}°, σ={PM_std:.2f}°)')
        # V_error if present
        ax = axes[1, 1]
        verr = V_error if isinstance(V_error, np.ndarray) else np.array([V_error])
        ax.bar([0], [np.mean(verr)*1e3], yerr=[np.std(verr)*1e3], color='purple', capsize=8)
        ax.set_xticks([0]); ax.set_xticklabels(['V_err'])
        ax.set_ylabel('Input Offset (mV)')
        ax.set_title(f'Offset (µ={np.mean(verr)*1e3:.2f} mV, σ={np.std(verr)*1e3:.2f} mV)')
        plt.suptitle('Monte Carlo Summary (mean ± std, min/max summarized)')
        plt.tight_layout()
        if save:
            plt.savefig(f'{OUTPUT_DIR}/monte_carlo_histograms.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {OUTPUT_DIR}/monte_carlo_histograms.png")
        plt.show()
        return

    # Otherwise assume arrays of per-run samples and plot histograms
    if not raw_present:
        Av0_dB = np.array(av_vals)
        fT = np.array(ft_vals)
        PM = np.array(pm_vals)
        V_error = np.array(voff_vals) if voff_vals is not None else np.zeros_like(Av0_dB)
    # if raw_present, Av0_dB, fT, PM, V_error were set above from mc_raw.txt

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Av0 in dB
    ax = axes[0, 0]
    ax.hist(Av0_dB, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(Av0_dB), color='r', linestyle='--', label=f'µ = {np.mean(Av0_dB):.2f} dB')
    ax.axvline(x=np.mean(Av0_dB)-np.std(Av0_dB), color='orange', linestyle=':', label=f'µ±σ')
    ax.axvline(x=np.mean(Av0_dB)+np.std(Av0_dB), color='orange', linestyle=':')
    ax.set_xlabel('Av0 (dB)')
    ax.set_ylabel('Count')
    ax.set_title(f'Gain Distribution (µ={np.mean(Av0_dB):.2f} dB, σ={np.std(Av0_dB):.2f} dB)')
    ax.legend()

    # fT
    ax = axes[0, 1]
    ax.hist(fT/1e3, bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
    ft_mu = np.mean(fT)/1e3
    ft_std = np.std(fT)/1e3
    ax.axvline(x=ft_mu, color='r', linestyle='--', label=f'µ = {ft_mu:.0f} kHz')
    ax.axvline(x=ft_mu - ft_std, color='orange', linestyle=':', label='µ±σ')
    ax.axvline(x=ft_mu + ft_std, color='orange', linestyle=':')
    ax.set_xlabel('fT (kHz)')
    ax.set_ylabel('Count')
    ax.set_title(f'Transition Frequency (µ={ft_mu:.0f}kHz, σ={ft_std:.0f}kHz)')
    ax.legend()

    # Phase Margin
    ax = axes[1, 0]
    ax.hist(PM, bins=50, color='coral', edgecolor='black', alpha=0.7)
    pm_mu = np.mean(PM)
    pm_std = np.std(PM)
    ax.axvline(x=pm_mu, color='r', linestyle='--', label=f'µ = {pm_mu:.2f}°')
    ax.axvline(x=60, color='black', linestyle='-', linewidth=2, label='Min spec (60°)')
    ax.axvline(x=pm_mu - pm_std, color='orange', linestyle=':', label='µ±σ')
    ax.axvline(x=pm_mu + pm_std, color='orange', linestyle=':')
    ax.set_xlabel('Phase Margin (°)')
    ax.set_ylabel('Count')
    ax.set_title(f'Phase Margin (µ={pm_mu:.2f}°, σ={pm_std:.2f}°)')
    ax.legend()

    # V_error (offset)
    ax = axes[1, 1]
    ax.hist(V_error*1e3, bins=50, color='purple', edgecolor='black', alpha=0.7)
    verr_mu = np.mean(V_error)*1e3
    verr_std = np.std(V_error)*1e3
    ax.axvline(x=verr_mu, color='r', linestyle='--', label=f'µ = {verr_mu:.2f} mV')
    ax.axvline(x=verr_mu - verr_std, color='orange', linestyle=':', label='µ±σ')
    ax.axvline(x=verr_mu + verr_std, color='orange', linestyle=':')
    ax.set_xlabel('Input Offset Voltage (mV)')
    ax.set_ylabel('Count')
    ax.set_title(f'Offset Voltage (µ={verr_mu:.2f} mV, σ={verr_std:.2f} mV)')
    ax.legend()

    plt.suptitle('Monte Carlo Analysis', fontsize=14, fontweight='bold')
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
