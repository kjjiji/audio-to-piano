"""Check bank 0/program 0 key and velocity coverage without reading PCM."""
from pathlib import Path
import struct
import numpy as np

def tables(path):
    result={}
    with Path(path).open('rb') as stream:
        header=stream.read(12)
        if len(header)!=12 or header[:4]!=b'RIFF' or header[8:]!=b'sfbk':raise ValueError('Not a SoundFont RIFF file')
        while True:
            chunk=stream.read(8)
            if not chunk:break
            if len(chunk)!=8:raise ValueError('Truncated SoundFont chunk')
            tag,size=struct.unpack('<4sI',chunk);end=stream.tell()+size+(size%2)
            if tag==b'LIST' and size>=4:
                kind=stream.read(4)
                if kind==b'pdta':
                    while stream.tell()<end-(size%2):
                        label,length=struct.unpack('<4sI',stream.read(8))
                        if length>32*1024*1024:raise ValueError('Oversized SoundFont metadata table')
                        data=stream.read(length)
                        if len(data)!=length:raise ValueError('Truncated SoundFont metadata')
                        result[label.decode('ascii')]=data
                        if length%2:stream.read(1)
            stream.seek(end)
    for name in ['phdr','pbag','pgen','inst','ibag','igen']:
        if name not in result:raise ValueError('Missing SoundFont metadata: '+name)
    return result

def piano_coverage(path):
    t=tables(path)
    presets=list(struct.iter_unpack('<20sHHHIII',t['phdr']))
    instruments=list(struct.iter_unpack('<20sH',t['inst']))
    pbag=list(struct.iter_unpack('<HH',t['pbag']));ibag=list(struct.iter_unpack('<HH',t['ibag']))
    pgen=list(struct.iter_unpack('<HH',t['pgen']));igen=list(struct.iter_unpack('<HH',t['igen']))
    masks=np.zeros((128,128),bool)
    def values(bags,gens,index):return dict(gens[bags[index][0]:bags[index+1][0]])
    def bounds(zone,parent):
        key=zone.get(43,0x7f00);vel=zone.get(44,0x7f00)
        return (max(parent[0],key&255),min(parent[1],key>>8),max(parent[2],vel&255),min(parent[3],vel>>8))
    chosen=next((k for k,p in enumerate(presets[:-1]) if p[1]==0 and p[2]==0),None)
    if chosen is None:raise ValueError('SoundFont has no bank 0/program 0 piano')
    pglobal=(0,127,0,127)
    for bag in range(presets[chosen][3],presets[chosen+1][3]):
        zone=values(pbag,pgen,bag)
        if 41 not in zone:pglobal=bounds(zone,pglobal);continue
        prange=bounds(zone,pglobal);index=zone[41]
        if index>=len(instruments)-1:raise ValueError('Invalid SoundFont instrument index')
        iglobal=(0,127,0,127)
        for ib in range(instruments[index][1],instruments[index+1][1]):
            izone=values(ibag,igen,ib)
            if 53 not in izone:iglobal=bounds(izone,iglobal);continue
            krange=bounds(izone,iglobal)
            low=max(prange[0],krange[0]);high=min(prange[1],krange[1])
            vlow=max(prange[2],krange[2]);vhigh=min(prange[3],krange[3])
            if low<=high and vlow<=vhigh:masks[low:high+1,vlow:vhigh+1]=True
    return masks

def validate_notes(path,midi):
    mask=piano_coverage(path)
    needed={(n.pitch,n.velocity) for i in midi.instruments for n in i.notes}
    missing=sorted((pitch,velocity) for pitch,velocity in needed if not mask[pitch,velocity])
    if missing:raise ValueError('SoundFont piano has silent key/velocity gaps for '+repr(missing[:12])+'. Select a font with continuous velocity coverage.')
    return {'required_key_velocity_pairs':len(needed),'uncovered_pairs':0,
        'covers_all_88_keys_and_velocities':bool(mask[21:109,1:128].all())}
