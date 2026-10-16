"""Isolated Linux/CPU dependency setup; no models, audio or soundfonts bundled."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def plan(venv,python):
    target=Path(venv).expanduser().resolve()
    executable=target/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    constraints=Path(__file__).with_name('torch-constraints.txt')
    commands=[]
    if not executable.is_file():commands.append([python,'-m','venv',str(target)])
    commands.extend([
        [str(executable),'-m','pip','install','--upgrade','pip'],
        [str(executable),'-m','pip','install','torch==2.5.1','torchaudio==2.5.1',
         '--index-url','https://download.pytorch.org/whl/cpu'],
        [str(executable),'-m','pip','install','-r',str(Path(__file__).with_name('requirements.txt')),
         '-c',str(constraints)],
        [str(executable),'-m','pip','install','basic-pitch==0.4.0','--no-deps'],
        [str(executable),str(Path(__file__).with_name('piano.py')),'doctor'],
    ])
    return target,commands


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv',required=True)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args()
    if not args.plan and (sys.platform!='linux' or sys.version_info[:2]!=(3,11)):
        parser.error('Validated installer requires Linux and Python 3.11. Use --plan to inspect, or prepare a compatible runtime manually.')
    target,commands=plan(args.venv,args.python)
    if target==Path(sys.prefix).resolve() and sys.prefix==sys.base_prefix:
        parser.error('Refusing to use the system Python prefix as the target environment.')
    print(json.dumps({'target_venv':str(target),'commands':commands,'models_and_soundfont_downloaded':False}),flush=True)
    if args.plan:return
    for command in commands:subprocess.run(command,check=True)


if __name__=='__main__':main()
