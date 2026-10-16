"""Fit a beat grid and inspect coverage for the complete, explicit input."""
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import torch
from beat_this.inference import File2Beats
from muscriptor.utils.beats import infer_beats_per_bar
from context import WORK, CONFIG

torch.set_num_threads(1)
model = File2Beats(
    checkpoint_path=CONFIG.get('beat_checkpoint') or 'final0',
    device='cpu', dbn=False)
beats, downbeats = model(str(WORK/'source.wav'))
beats, downbeats = np.asarray(beats), np.asarray(downbeats)
np.savez(WORK/'beats.npz', beats=beats, downbeats=downbeats)
assert len(beats) >= 4 and np.isfinite(beats).all()
delta = np.diff(beats)
assert np.all(delta > 0)
period = float(np.median(delta))
index = np.r_[0, np.cumsum(np.maximum(1, np.rint(delta/period)).astype(int))]
slope, phase = np.polyfit(index, beats, 1)
residual = beats-phase-slope*index
simple_slope, simple_phase = np.polyfit(np.arange(len(beats)), beats, 1)
simple_residual = beats-simple_phase-simple_slope*np.arange(len(beats))
signal, sr = sf.read(WORK/'source.wav', dtype='float32', always_2d=True)
duration = len(signal)/sr
regions = []
for start in range(0, int(np.ceil(duration)), 20):
    mask = (beats >= start) & (beats < start+20)
    selected = beats[mask]
    if len(selected) >= 3:
        regions.append({'start': start, 'end': min(start+20, duration),
                        'median_bpm': float(60/np.median(np.diff(selected))),
                        'fitted_bpm': float(60/np.polyfit(index[mask], selected, 1)[0])})
tail = []
for start in range(max(0, int(duration)-9), int(np.ceil(duration))):
    part = signal[start*sr:min((start+1)*sr, len(signal))]
    if part.size:
        tail.append({'start': start, 'rms': float(np.sqrt(np.mean(part**2)))})
report = {
    'beat_count': len(beats), 'downbeat_count': len(downbeats),
    'median_interval_seconds': period,
    'interval_quantiles': np.percentile(delta, [5, 25, 50, 75, 95]).tolist(),
    'fitted_bpm_with_skipped_beats': 60/float(slope),
    'fitted_phase_seconds': float(phase),
    'residual_rms_seconds': float(np.sqrt(np.mean(residual**2))),
    'ordinary_fit_bpm': 60/float(simple_slope),
    'ordinary_fit_residual_seconds': float(np.std(simple_residual)),
    'conservative_beats_per_bar': infer_beats_per_bar(beats, downbeats),
    'first_downbeat_seconds': float(downbeats[0]) if len(downbeats) else None,
    'regions': regions, 'source_duration_seconds': duration,
    'tail_rms_per_second': tail,
    'method': 'Beat This final0, local weights; integer beat index allows skipped detections.'}
(WORK/'tempo_analysis.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
