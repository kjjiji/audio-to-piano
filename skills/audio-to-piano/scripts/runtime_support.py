"""Shared device, precision and local-model checks. Imports stay lazy for CLI help."""
import contextlib
import importlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


def select_device(value='auto', torch_module=None):
    if not re.fullmatch(r'auto|cpu|cuda(?::\d+)?', value):
        raise ValueError('Device must be auto, cpu, cuda or cuda:N')
    if value == 'cpu':
        return value
    if torch_module is None:
        import torch as torch_module
    cuda = torch_module.cuda
    if value == 'auto':
        value = 'cuda' if cuda.is_available() else 'cpu'
        if value == 'cpu':
            return value
    if not torch_module.version.cuda:
        raise ValueError('This PyTorch build has no CUDA runtime; use a CUDA wheel or --device cpu')
    if not cuda.is_available():
        raise ValueError('CUDA requested but unavailable; check NVIDIA driver, GPU visibility and PyTorch build')
    index = int(value.split(':')[1]) if ':' in value else cuda.current_device()
    if index >= cuda.device_count():
        raise ValueError(f'CUDA device index {index} is unavailable ({cuda.device_count()} visible devices)')
    return f'cuda:{index}'


def select_dtype(value, device, torch_module=None):
    if value not in ('auto', 'float32', 'float16', 'bfloat16'):
        raise ValueError('Dtype must be auto, float32, float16 or bfloat16')
    if device == 'cpu':
        if value == 'float16':
            raise ValueError('CPU float16 is unsupported; use bfloat16 or float32')
        return 'bfloat16' if value == 'auto' else value
    if torch_module is None:
        import torch as torch_module
    # Query the selected GPU, not whichever GPU happens to be current.
    with torch_module.cuda.device(device):
        bf16 = (torch_module.cuda.get_device_capability(device)[0] >= 8
                and torch_module.cuda.is_bf16_supported())
    if value == 'bfloat16' and not bf16:
        raise ValueError(f'{device} does not support native bfloat16; use float16 or float32')
    return ('bfloat16' if bf16 else 'float16') if value == 'auto' else value


def kernel_check(torch_module, device, dtype):
    """Exercise allocation, GEMM and attention; availability alone is insufficient."""
    with torch_module.inference_mode():
        x = torch_module.ones((1, 2, 16, 32), device=device, dtype=getattr(torch_module, dtype))
        y = x @ x.transpose(-2, -1)
        z = torch_module.nn.functional.scaled_dot_product_attention(x, x, x)
        if not bool(torch_module.isfinite(y).all()) or not bool(torch_module.isfinite(z).all()):
            raise ValueError('Non-finite output in device kernel check')
        if device.startswith('cuda'):
            torch_module.cuda.synchronize(device)
    return {'status': 'passed', 'operations': ['allocation', 'matmul', 'scaled_dot_product_attention'],
            'device': device, 'dtype': dtype}


def torch_diagnostics(requested='auto', dtype='auto'):
    report = {'requested_device': requested, 'requested_dtype': dtype, 'ready': False,
              'devices': [], 'kernel_test': {'status': 'not_run'}}
    try:
        import torch
        report.update(torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(), compiled_arches=torch.cuda.get_arch_list())
        import torchaudio
        report['torchaudio_version'] = torchaudio.__version__
        if torch.__version__.split('+')[0] != torchaudio.__version__.split('+')[0]:
            raise ValueError('torch and torchaudio release versions do not match')
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            report['devices'].append({'index': index, 'name': props.name,
                                      'compute_capability': list(torch.cuda.get_device_capability(index)),
                                      'total_memory_bytes': props.total_memory,
                                      'free_memory_bytes': torch.cuda.mem_get_info(index)[0]})
        device = select_device(requested, torch)
        precision = select_dtype(dtype, device, torch)
        report.update(selected_device=device, selected_dtype=precision)
        report['kernel_test'] = kernel_check(torch, device, precision)
        report['ready'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    # Driver and toolkit are separate observations; neither proves PyTorch works.
    for key, command in [('nvidia_smi', ['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total', '--format=csv,noheader']),
                         ('nvcc', ['nvcc', '--version'])]:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            report[key] = {'returncode': result.returncode, 'output': result.stdout.strip(),
                           'error': result.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            report[key] = {'error': str(error)}
    return report


def model_diagnostics(path):
    report = {'path': str(path) if path else None, 'ready': False,
              'weights_loaded': False}
    try:
        if not path:
            raise ValueError('No local MuScriptor weights found; pass --model-path')
        weights = Path(path)
        if not weights.is_file() or not weights.stat().st_size:
            raise ValueError('Model weights are missing or empty')
        config_path = weights.with_name('config.json')
        report['config_path'] = str(config_path)
        config = json.loads(config_path.read_text())
        for name in ('dim', 'num_heads', 'num_layers', 'card'):
            if type(config.get(name)) is not int or config[name] <= 0:
                raise ValueError(f'Model config requires a positive integer {name}')
        if config['dim'] % config['num_heads']:
            raise ValueError('Model dim must be divisible by num_heads')
        report.update(config=config, weights_bytes=weights.stat().st_size, ready=True)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    return report


def load_muscriptor(weights, device, dtype):
    """Use the public loader and align MuScriptor 0.3's CUDA autocast with dtype."""
    import torch
    from muscriptor import TranscriptionModel
    model = TranscriptionModel.load_model(weights, device=device, dtype=dtype)
    if device.startswith('cuda') and dtype != 'float16':
        # 0.3.0 otherwise forces fp16 compute even with bf16/fp32 weights.
        # Fail clearly if a reused version changes this internal interface.
        core = getattr(model, '_model', None)
        if core is None or not hasattr(core, 'autocast'):
            raise ValueError('MuScriptor autocast interface changed; use --dtype float16 or MuScriptor 0.3.0')
        core.autocast = (torch.autocast('cuda', dtype=getattr(torch, dtype))
                         if dtype != 'float32' else contextlib.nullcontext())
        if dtype == 'float32':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    return model


def model_smoke_test(weights, device, dtype, audio=None):
    """Consume a real multi-instrument inference stream and parse its MIDI."""
    import mido
    import numpy as np
    import soundfile as sf
    import torch
    from muscriptor.events import ProgressEvent
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='piano-doctor-') as directory:
        source = Path(directory) / 'smoke.wav'
        if audio:
            # Limit the diagnostic to five seconds; user audio stays local.
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(audio), '-t', '5',
                            '-ar', '16000', '-ac', '1', str(source)], check=True)
        else:
            t = np.arange(5 * 16000) / 16000
            signal = sum(.07 * np.sin(2 * np.pi * f * t) for f in (130.81, 261.63, 329.63))
            sf.write(source, signal, 16000)
        completed = 0
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            model = load_muscriptor(weights, device, dtype)
            core = getattr(model, '_model', None)
            if core is None:
                raise ValueError('Cannot verify the loaded MuScriptor model device; use MuScriptor 0.3.0')
            model_devices = sorted({str(p.device) for p in core.parameters()})
            if model_devices != [str(torch.device(device))]:
                raise ValueError(f'Loaded model devices {model_devices} differ from requested {device}')
            parameter_dtypes = sorted({str(p.dtype) for p in core.parameters()})
            def events():
                nonlocal completed
                for event in model.transcribe(source, instruments=['acoustic_piano', 'acoustic_bass'],
                                              prelude_forcing=True, use_sampling=False):
                    if isinstance(event, ProgressEvent):
                        completed = max(completed, event.completed)
                    yield event
            data = model.events_to_midi_bytes(events())
            if device.startswith('cuda'):
                torch.cuda.synchronize(device)
        if completed < 1:
            raise ValueError('MuScriptor did not complete any inference chunks')
        midi = mido.MidiFile(file=io.BytesIO(data))
        notes = sum(msg.type == 'note_on' and msg.velocity > 0 for track in midi.tracks for msg in track)
    return {'status': 'passed', 'device': device, 'dtype': dtype, 'completed_chunks': completed,
            'midi_bytes': len(data), 'note_count': notes, 'seconds': round(time.monotonic() - started, 3),
            'model_devices': model_devices, 'parameter_dtypes': parameter_dtypes,
            'musical_accuracy_evaluated': False}


def import_diagnostics():
    modules = ['numpy', 'scipy', 'librosa', 'soundfile', 'pretty_midi', 'mido', 'fluidsynth',
               'torch', 'torchaudio', 'demucs.separate', 'muscriptor', 'beat_this.inference',
               'onnxruntime', 'basic_pitch.inference', 'essentia.standard']
    results = {}
    for name in modules:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                importlib.import_module(name)
            results[name] = {'ready': True}
        except Exception as error:
            results[name] = {'ready': False, 'error': f'{type(error).__name__}: {error}'}
    versions = {}
    for name in ('numpy', 'scipy', 'librosa', 'soundfile', 'torch', 'torchaudio', 'muscriptor',
                 'demucs', 'beat_this', 'basic-pitch', 'onnxruntime', 'essentia'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return results, versions


def onnx_diagnostics():
    report = {'ready': False, 'provider': 'CPUExecutionProvider'}
    try:
        import basic_pitch
        import onnxruntime as ort
        model = Path(basic_pitch.__file__).parent / 'saved_models/icassp_2022/nmp.onnx'
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        report.update(ready=True, model_path=str(model), providers=session.get_providers())
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    return report


def package_conflicts():
    """Basic Pitch uses ONNX here; its upstream TensorFlow default is intentional."""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True,
                                text=True, timeout=30)
        if result.returncode == 0:
            return {'ready': True, 'conflicts': [], 'expected_exclusions': []}
        lines = (result.stdout + result.stderr).strip().splitlines()
        expected = [line for line in lines if re.fullmatch(
            r'basic-pitch 0\.4\.0 requires tensorflow, which is not installed\.', line)]
        conflicts = [line for line in lines if line not in expected]
        return {'ready': not conflicts, 'conflicts': conflicts, 'expected_exclusions': expected}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'ready': False, 'error': str(error)}
