"""Command line access to the released model and latent experiments."""
import argparse
import json
import os
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description="MolAlign3D molecular inference")
    parser.add_argument('--experiment',choices=['reconstruct','encode','wander','atom-edit','interpolate','swap'],default='reconstruct')
    parser.add_argument('--smiles',default='CC(=O)Oc1ccccc1C(=O)O')
    parser.add_argument('--smiles-b',default='CCOc1ccccc1')
    parser.add_argument('--sdf',type=Path)
    parser.add_argument('--sdf-b',type=Path)
    parser.add_argument('--steps',type=int,default=100,choices=range(10,201),metavar='10..200')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--frames',type=int,default=5,choices=range(3,12),metavar='3..11')
    parser.add_argument('--step-size',type=float,default=.1)
    parser.add_argument('--space',choices=['zx','zh','both'],default='both')
    parser.add_argument('--radius',type=int,default=2,choices=range(0,5))
    parser.add_argument('--no-align',action='store_true')
    parser.add_argument('--device',choices=['cpu','auto'],default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--output',type=Path,default=Path('outputs/result.json'))
    args=parser.parse_args()
    if not 0<=args.step_size<=1:parser.error('--step-size must be between 0 and 1')
    if args.threads<1:parser.error('--threads must be positive')
    if not 0<=args.seed<=2147483647:parser.error('--seed must be in 0..2147483647')
    os.environ['MOLECULAR_DEVICE']=args.device
    os.environ['MOLECULAR_CPU_THREADS']=str(args.threads)
    from .engine import Engine
    engine=Engine()
    a={'molblock':args.sdf.read_text(),'smiles':''} if args.sdf else {'smiles':args.smiles}
    b={'molblock':args.sdf_b.read_text(),'smiles':''} if args.sdf_b else {'smiles':args.smiles_b}
    common={'key':'molalign3d','seed':args.seed}
    if args.experiment=='encode':result=engine.encode(**common,**a)
    elif args.experiment=='reconstruct':result=engine.reconstruct(**common,**a,steps=args.steps)
    elif args.experiment=='wander':result=engine.wander(**common,**a,steps=args.steps,frames=args.frames,step_size=args.step_size,space=args.space)
    elif args.experiment=='atom-edit':result=engine.atom_edit(**common,**a,steps=args.steps,radius=args.radius)
    else:result=getattr(engine,args.experiment)(**common,a=a,b=b,steps=args.steps,**({'frames':args.frames} if args.experiment=='interpolate' else {}),align=not args.no_align)
    output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    molecules=result.get('frames',[result])
    blocks=[]
    for frame in molecules:
        if frame.get('decoded',{}).get('molblock'):
            blocks.append(frame['decoded']['molblock']+'\n> <experiment_frame>\n'+frame.get('label','reconstruction')+'\n\n$$$$\n')
    if blocks:output.with_suffix('.sdf').write_text(''.join(blocks))
    print(json.dumps({'model':'MolAlign3D','experiment':args.experiment,'steps':args.steps,'output':str(output)}))

if __name__=='__main__':main()
