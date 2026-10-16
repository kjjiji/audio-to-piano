"""Validate a portable skills-only plugin and create its contained ZIP."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import stat
import xml.etree.ElementTree as ET
import zipfile


def package(root,destination,schema=None):
    root=Path(root).resolve();destination=Path(destination).resolve()
    if destination.is_relative_to(root):raise ValueError('Archive must be outside the plugin directory')
    manifest=json.loads((root/'plugin.json').read_text())
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',root.name) or len(root.name)>64:
        raise ValueError('Invalid root directory name')
    if manifest['name']!=root.name:raise ValueError('Manifest name must match directory')
    if manifest.get('$schema')!='https://agent-plugins.org/schemas/1.0.0/plugin.schema.json':
        raise ValueError('Unexpected schema version')
    if any(key in manifest for key in ['skills','mcpServers','apps','interface']):raise ValueError('Nonportable root keys')
    if schema:
        import jsonschema
        jsonschema.validate(manifest,json.loads(Path(schema).read_text()))
    interface=manifest['extensions']['com.openai']['interface']
    if not interface['displayName'] or len(interface['shortDescription'])>30:raise ValueError('Invalid presentation')
    for key in ['logo','composerIcon','logoDark','composerIconDark']:
        if key not in interface:continue
        value=interface[key]
        path=(root/value).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Asset escapes or is missing')
        if path.stat().st_size>5*1024*1024:raise ValueError('Oversized icon')
        if path.suffix=='.svg':
            svg=ET.fromstring(path.read_text());width=float(svg.attrib['width']);height=float(svg.attrib['height'])
            if width!=height or width<48:raise ValueError('Icon must be square and >=48px')
            if any(el.tag.rsplit('}',1)[-1] in ['script','foreignObject'] for el in svg.iter()):
                raise ValueError('Active SVG content')
            if any(k.rsplit('}',1)[-1]=='href' and not str(v).startswith('#') for el in svg.iter() for k,v in el.attrib.items()):
                raise ValueError('External SVG content')
    skills=list((root/'skills').glob('*/SKILL.md'))
    if not skills:raise ValueError('No skills')
    for path in skills:
        text=path.read_text()
        front=re.match(r'^---\n(.*?)\n---\n',text,re.S)
        if not front:raise ValueError('Missing frontmatter')
        name=re.search(r'^name:\s*(\S+)\s*$',front.group(1),re.M)
        if not name or name.group(1)!=path.parent.name:raise ValueError('Skill name must match directory')
        if not re.search(r'^description:\s*\S',front.group(1),re.M):raise ValueError('Missing trigger description')
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if '://' not in link and not (path.parent/link).is_file():raise ValueError('Missing skill reference: '+link)
    files=[]
    denied={'.mp3','.wav','.flac','.mid','.sf2','.safetensors','.ckpt','.pyc'}
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('Symlink in plugin')
        relative=path.relative_to(root)
        if relative.parts[0]=='dist' or any(part in ['.git','__pycache__'] for part in relative.parts):continue
        if not path.is_file():continue
        if any(part in ['.venv','node_modules'] or part.startswith('.env') for part in relative.parts):
            raise ValueError('Runtime, secrets or repository data in plugin')
        if path.suffix.lower() in denied:raise ValueError('Runtime media/model artifact in plugin')
        if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
        files.append(path)
    for required in ['README.md','LICENSE','THIRD_PARTY_NOTICES.md','VALIDATION.md']:
        if not (root/required).is_file():raise ValueError('Missing '+required)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):archive.write(path,str(Path(root.name)/path.relative_to(root)))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:raise ValueError('Invalid archive')
        if {Path(info.filename).parts[0] for info in archive.infolist()}!={root.name}:raise ValueError('Multiple archive roots')
        for info in archive.infolist():
            if info.filename.startswith('/') or '..' in Path(info.filename).parts:raise ValueError('Unsafe archive path')
            if stat.S_ISLNK(info.external_attr>>16):raise ValueError('Archived symlink')
            original=root/Path(info.filename).relative_to(root.name)
            if archive.read(info)!=original.read_bytes():raise ValueError('Archive byte mismatch')
    return {'archive':str(destination),'bytes':destination.stat().st_size,
        'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'files':len(files),
        'skill_names':[path.parent.name for path in skills],'display_name':interface['displayName'],
        'official_schema_validated':bool(schema)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root');parser.add_argument('archive');parser.add_argument('--schema')
    args=parser.parse_args()
    print(json.dumps(package(args.root,args.archive,args.schema),ensure_ascii=False,indent=2))
