"""Runtime paths are supplied by the CLI, never by a song-specific module."""
from pathlib import Path
import json
import os

PIPELINE_VERSION='0.1.0'
WORK=Path(os.environ.get('PIANO_WORK_DIR','.')).expanduser().resolve()
CONFIG_PATH=WORK/'config.json'
CONFIG=json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
DEST=Path(CONFIG.get('output_dir',str(WORK/'output'))).expanduser().resolve()
STEM=CONFIG.get('basename','Piano-Arrangement')
TITLE=CONFIG.get('title','Piano arrangement')
STEMS=WORK/'separated/htdemucs_6s/source'

def emit(stage,**values):
    print(json.dumps({'stage':stage,**values},ensure_ascii=False),flush=True)

def save_json(path,value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)))
    temporary.replace(path)
