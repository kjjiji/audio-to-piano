"""Isolated Linux setup: CPU (default), CUDA 12.4, or preserve an existing runtime."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parent
SNAPSHOT = """import importlib.metadata as m,json,sys
print(json.dumps({'version':list(sys.version_info[:2]),'platform':sys.platform,
 'prefix':sys.prefix,'base_prefix':sys.base_prefix,
 'packages':{d.metadata['Name']:d.version for d in m.distributions() if d.metadata['Name']}}))
"""


def snapshot(python):
    return json.loads(subprocess.check_output([str(python), '-c', SNAPSHOT], text=True))


def normalized(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def plan(venv, python, profile='cpu'):
    target = Path(venv).expanduser().resolve()
    executable = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    exists = executable.is_file()
    if profile not in ('cpu', 'cuda', 'reuse'):
        raise ValueError('Unknown installation profile')
    if profile == 'reuse' and not exists:
        raise ValueError('reuse requires an existing environment with bin/python')
    info = snapshot(executable if exists else python)
    if info['platform'] != 'linux' or info['version'] not in ([3, 11], [3, 12]):
        raise ValueError('Supported installer targets are Linux Python 3.11 or 3.12')
    if exists and info['prefix'] == info['base_prefix'] and not (target / 'conda-meta').is_dir():
        raise ValueError('Refusing to install into system Python; use a venv or conda environment')
    if exists and profile != 'reuse':
        raise ValueError('Target environment already exists; use --profile reuse or a new --venv path')
    commands = []
    files = {}
    protected = {}
    pip = [str(executable), '-m', 'pip']
    if profile == 'reuse':
        protected = info['packages']
        installed = {normalized(name): version for name, version in protected.items()}
        for name in ('torch', 'torchaudio', 'muscriptor'):
            if name not in installed:
                raise ValueError('reuse requires an installed ' + name + '; use a new CUDA environment instead')
        constraints = target / '.audio-to-piano-preserved.txt'
        requirements = target / '.audio-to-piano-missing.txt'
        missing = []
        for line in (SCRIPTS / 'requirements.txt').read_text().splitlines():
            match = re.match(r'^([A-Za-z0-9_.-]+)', line)
            if match and normalized(match[1]) not in installed:
                if normalized(match[1])=='onnxruntime' and 'onnxruntime-gpu' in installed:continue
                missing.append(line)
        files[str(constraints)] = ''.join(f'{name}=={version}\n' for name, version in sorted(protected.items()))
        files[str(requirements)] = '\n'.join(missing) + '\n'
        if missing:
            install = [*pip, 'install', '-r', str(requirements), '-c', str(constraints)]
            commands.extend([[*install, '--dry-run'], install])
        if 'basic-pitch' not in installed:
            commands.append([*pip, 'install', 'basic-pitch==0.4.0', '--no-deps'])
    else:
        commands.append([python, '-m', 'venv', str(target)])
        flavor = 'cpu' if profile == 'cpu' else 'cu124'
        constraints = SCRIPTS / ('torch-constraints.txt' if profile == 'cpu' else 'torch-constraints-cuda.txt')
        commands.extend([
            [*pip, 'install', '--upgrade', 'pip'],
            [*pip, 'install', f'torch==2.5.1+{flavor}', f'torchaudio==2.5.1+{flavor}',
             '--index-url', f'https://download.pytorch.org/whl/{flavor}'],
            [*pip, 'install', '-r', str(SCRIPTS / 'requirements.txt'), '-c', str(constraints)],
            [*pip, 'install', 'basic-pitch==0.4.0', '--no-deps'],
        ])
    commands.append([str(executable), str(SCRIPTS / 'piano.py'), 'doctor',
                     '--device', 'cuda' if profile == 'cuda' else 'auto'])
    return {'target_venv': str(target), 'profile': profile, 'python_version': info['version'],
            'commands': commands, 'files_to_write': files, 'preserved_packages': protected,
            'models_and_soundfont_downloaded': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', required=True)
    parser.add_argument('--python', default=sys.executable, help='Interpreter used to create a new environment')
    parser.add_argument('--profile', choices=['cpu', 'cuda', 'reuse'], default='cpu')
    parser.add_argument('--plan', action='store_true', help='Inspect the target and print a plan without writing files')
    args = parser.parse_args()
    try:
        details = plan(args.venv, args.python, args.profile)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(details), flush=True)
    if args.plan:
        return
    for path, data in details['files_to_write'].items():
        Path(path).write_text(data)
    for command in details['commands']:
        subprocess.run(command, check=True)
    if details['preserved_packages']:
        after = snapshot(Path(details['target_venv']) / 'bin/python')['packages']
        changed = [name for name, version in details['preserved_packages'].items() if after.get(name) != version]
        if changed:
            raise RuntimeError('Previously installed distributions changed: ' + ', '.join(changed))


if __name__ == '__main__':
    main()
