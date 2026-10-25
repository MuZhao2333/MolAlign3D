"""Inference for the released molecular encoder and decoder checkpoint."""
from __future__ import annotations

import contextlib
import io
import logging
import os
import random
import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[1]
DEFAULT_WEIGHTS = CHECKOUT / "weights" / "molalign3d" if (CHECKOUT / "weights" / "molalign3d").is_dir() else Path.home() / ".cache" / "molalign3d" / "weights"
ROOT = Path(os.environ.get("MOLALIGN3D_WEIGHTS_DIR", str(DEFAULT_WEIGHTS))).expanduser().resolve()
RUNTIME = Path(os.environ.get("MOLECULAR_RUNTIME", str(Path.home() / ".cache" / "molalign3d" / "runtime"))).expanduser().resolve()
ATOM_TO_INDEX = {6:0, 7:1, 8:2, 9:3, 15:4, 16:5, 17:6, 35:7, 53:8}
INDEX_TO_ATOM = {value:key for key,value in ATOM_TO_INDEX.items()}
MODEL_INFO = {'molalign3d': {'name': 'MolAlign3D', 'global_nodes': 8, 'structural_dim': 16, 'decoder_dim': 80, 'checkpoint': 'model.ckpt', 'config': 'configs/model.yaml', 'source_revision': 'a47090b56f7f8db07b6d002640891807f8c96512', 'conditioning': '512-dimensional UniMol representation split into 8 × 64 channels'}}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_molecule(smiles="", molblock="", seed=42):
    if bool(smiles.strip()) == bool(molblock.strip()):
        raise ValueError("Provide either one SMILES string or one SDF/MOL file.")
    if len(smiles)>1000 or len(molblock)>200000:
        raise ValueError("Input exceeds the supported size limit.")
    if molblock:
        mol=Chem.MolFromMolBlock(molblock.split("$$$$")[0], removeHs=True, sanitize=True)
        if mol is None:
            raise ValueError("The SDF/MOL file could not be parsed.")
        if mol.GetNumConformers()==0 or not mol.GetConformer().Is3D():
            raise ValueError("Upload a file with a 3D conformer, or use SMILES to generate one.")
        source="Uploaded 3D coordinates; hydrogens removed"
    else:
        mol=Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            raise ValueError("Invalid SMILES string.")
        source="RDKit ETKDGv3 conformer; hydrogens removed after generation"
    if len(Chem.GetMolFrags(mol))!=1:
        raise ValueError("Please provide one connected molecule, without disconnected salts or fragments.")
    unsupported=sorted({atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() not in ATOM_TO_INDEX and atom.GetAtomicNum()!=1})
    if unsupported:
        raise ValueError("This checkpoint does not support: "+", ".join(unsupported)+". Supported heavy atoms: C, N, O, F, P, S, Cl, Br, I.")
    if not 2<=mol.GetNumHeavyAtoms()<=80:
        raise ValueError("This model accepts 2–80 heavy atoms per molecule.")
    if not molblock:
        mol=Chem.AddHs(mol)
        params=AllChem.ETKDGv3()
        params.randomSeed=int(seed)
        params.maxIterations=1000
        if AllChem.EmbedMolecule(mol, params)!=0:
            raise ValueError("Could not generate a 3D conformer. Try uploading an SDF file.")
        if AllChem.UFFHasAllMoleculeParams(mol):
            AllChem.UFFOptimizeMolecule(mol,maxIters=200)
    mol=Chem.RemoveHs(mol)
    coords=np.asarray(mol.GetConformer().GetPositions(),dtype=np.float32)
    if not np.isfinite(coords).all() or np.max(np.abs(coords))>10000:
        raise ValueError("The input contains invalid or out-of-range coordinates.")
    coords-=coords.mean(axis=0,keepdims=True)
    for i,point in enumerate(coords):
        mol.GetConformer().SetAtomPosition(i,point.astype(float))
    return mol,coords,source


def molecule_payload(mol, source=""):
    valid=True
    try:
        Chem.SanitizeMol(Chem.Mol(mol))
        smiles=Chem.MolToSmiles(mol)
    except Exception:
        valid=False
        smiles=None
    try:
        block=Chem.MolToMolBlock(mol,kekulize=False)
    except Exception:
        block=""
    return {"molblock":block,"smiles":smiles,"rdkit_valid":valid,"source":source,
            "atomic_numbers":[a.GetAtomicNum() for a in mol.GetAtoms()],
            "coordinates":np.asarray(mol.GetConformer().GetPositions()).round(6).tolist(),
            "atom_count":mol.GetNumAtoms(),"fragments":len(Chem.GetMolFrags(mol))}


class Engine:
    def __init__(self):
        RUNTIME.mkdir(parents=True,exist_ok=True)
        requested=os.environ.get("MOLECULAR_DEVICE","auto")
        self.device=torch.device("cpu" if requested=="cpu" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        torch.set_num_threads(int(os.environ.get("MOLECULAR_CPU_THREADS","4")))
        self.models={}
        self.encodings=OrderedDict()
        from molalign3d.utils.build_mol import MoleculeBuilder
        self.builder=MoleculeBuilder()

    def load(self,key):
        if key in self.models:
            return self.models[key]
        if key not in MODEL_INFO:
            raise ValueError("Unknown model.")
        info=MODEL_INFO[key]
        cfg_path=HERE/info["config"]
        cfg=yaml.safe_load(cfg_path.read_text())
        cfg["evaluation"]["save_dir"]=str(RUNTIME/key)
        os.environ["UNIMOL_WEIGHT_DIR"]=str(ROOT)
        from molalign3d.models.autoencoder import TrainLoop
        cfg["unimol"]["ckpt_path"]=str(ROOT/"mol_pre_no_h_220816.pt")
        if (cfg["encoder_config"]["global_node_num"],cfg["optimal_layer_config"]["latent_dim"],cfg["decoder_config"]["protein_atom_feature_dim"]) != (info["global_nodes"],info["structural_dim"],info["decoder_dim"]):
            raise RuntimeError("Model metadata disagrees with the original training configuration.")
        with contextlib.redirect_stdout(io.StringIO()):
            model=TrainLoop(cfg)
        checkpoint_path=ROOT/info["checkpoint"]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Place the verified checkpoint at {checkpoint_path}; see README.md.")
        checkpoint=torch.load(str(checkpoint_path),map_location="cpu",weights_only=False)
        model.load_state_dict(checkpoint["state_dict"],strict=True)
        model.to(self.device)
        model.eval()
        model.decoder.device=self.device
        for attr in ("sigma1_coord","sigma1_v","beta1"):
            value=getattr(model.decoder,attr,None)
            if isinstance(value,torch.Tensor):
                setattr(model.decoder,attr,value.to(self.device))
        self.models[key]=model
        logging.info("Loaded %s strictly on %s",info["name"],self.device)
        return model

    def _encode(self,key,mol,coords):
        model=self.load(key)
        indices=torch.tensor([ATOM_TO_INDEX[a.GetAtomicNum()] for a in mol.GetAtoms()],dtype=torch.long,device=self.device)
        x=torch.as_tensor(coords,dtype=torch.float32,device=self.device)
        batch=torch.zeros_like(indices)
        with torch.inference_mode():
            zh,zx,global_batch,*_=model.encode(F.one_hot(indices,9).float(),x,batch,deterministic=True)
            condition=model._attach_unimol_to_Zh(zh,global_batch,indices,x,batch) if key=="molalign3d" else zh
        for tensor in (zh,zx,condition):
            if not torch.isfinite(tensor).all():
                raise RuntimeError("Non-finite encoder output.")
        return {"key":key,"mol":mol,"zh":zh.detach().cpu(),"zx":zx.detach().cpu(),"condition":condition.detach().cpu(),"created":time.time()}

    @staticmethod
    def latent_payload(encoded):
        return {"positions":encoded["zx"].numpy().round(6).tolist(),
                "features":encoded["zh"].numpy().round(6).tolist(),
                "decoder_features":encoded["condition"].numpy().round(6).tolist(),
                "feature_shape":list(encoded["zh"].shape),"decoder_shape":list(encoded["condition"].shape),
                "position_shape":list(encoded["zx"].shape)}

    def encode(self,key,smiles="",molblock="",seed=42,progress=lambda *args:None):
        started=time.monotonic()
        progress("Preparing input conformer",0.1)
        mol,coords,source=prepare_molecule(smiles,molblock,seed)
        set_seed(seed)
        progress("Running the encoder",0.3)
        encoded=self._encode(key,mol,coords)
        token=uuid.uuid4().hex
        self.encodings[token]=encoded
        while len(self.encodings)>32:
            self.encodings.popitem(last=False)
        return {"model":key,"model_name":MODEL_INFO[key]["name"],"encoding_id":token,
                "input":molecule_payload(mol,source),"latent":self.latent_payload(encoded),
                "encode_seconds":round(time.monotonic()-started,3),"seed":seed,"provenance":self.provenance(key)}

    @staticmethod
    def provenance(key):
        info=MODEL_INFO[key]
        return {k:info[k] for k in ("checkpoint","source_revision","global_nodes","structural_dim","decoder_dim","conditioning") if k in info}

    def _decode(self,encoded,steps,seed,atom_count=None):
        model=self.load(encoded["key"])
        set_seed(seed)
        count=atom_count or encoded["mol"].GetNumAtoms()
        if not 2<=count<=80:
            raise ValueError("Decoded heavy-atom count must be between 2 and 80.")
        zx=encoded["zx"].to(self.device)
        condition=encoded["condition"].to(self.device)
        with torch.inference_mode():
            _,chain,_=model.decoder.sample(protein_pos=zx,protein_v=condition,
                batch_protein=torch.zeros(zx.shape[0],dtype=torch.long,device=self.device),
                batch_ligand=torch.zeros(count,dtype=torch.long,device=self.device),
                n_nodes=1,sample_steps=steps,desc="decode")
        coords=chain[-1][0].detach().cpu().numpy()
        numbers=[INDEX_TO_ATOM[i] for i in chain[-1][1].argmax(-1).detach().cpu().tolist()]
        if not np.isfinite(coords).all():
            raise RuntimeError("Non-finite decoder output.")
        perceived=True
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                generated=self.builder.build_mol(coords,numbers,require_H=False)
            if generated is None:
                raise ValueError("Bond perception failed.")
        except Exception:
            perceived=False
            editable=Chem.RWMol()
            conformer=Chem.Conformer(len(numbers))
            conformer.Set3D(True)
            for i,(number,point) in enumerate(zip(numbers,coords)):
                editable.AddAtom(Chem.Atom(int(number)))
                conformer.SetAtomPosition(i,point.astype(float))
            generated=editable.GetMol()
            generated.AddConformer(conformer)
        result=molecule_payload(generated,"Model-predicted atom types and coordinates; bonds inferred by the original postprocessor" if perceived else "Raw predicted atoms; bond perception unsuccessful")
        result["bond_perception_succeeded"]=perceived
        result["predicted_atom_count"]=count
        result["steps"]=steps
        return result,generated

    def decode(self,encoding_id,steps=100,seed=42,progress=lambda *args:None):
        if encoding_id not in self.encodings:
            raise ValueError("This encoding expired. Please encode the molecule again.")
        encoded=self.encodings[encoding_id]
        if time.time()-encoded["created"]>3600:
            del self.encodings[encoding_id]
            raise ValueError("This encoding expired. Please encode the molecule again.")
        started=time.monotonic()
        progress("Decoding cached latent",0.2)
        output,generated=self._decode(encoded,steps,seed)
        output["aligned_rmsd"]=None
        if output["rdkit_valid"] and output["fragments"]==1:
            try:
                # GetBestRMS determines a valid graph mapping; it never assumes decoded atom order.
                aligned=Chem.Mol(generated)
                rmsd=rdMolAlign.GetBestRMS(aligned,encoded["mol"],maxMatches=1000)
                output["aligned_rmsd"]=round(float(rmsd),4)
                output["aligned_molblock"]=Chem.MolToMolBlock(aligned,kekulize=False)
            except Exception:
                pass
        return {"model":encoded["key"],"encoding_id":encoding_id,"decoded":output,
                "decode_seconds":round(time.monotonic()-started,3),"seed":seed,"provenance":self.provenance(encoded["key"])}

    def reconstruct(self,progress=lambda *args:None,**args):
        steps=args.pop("steps",100)
        encoded=self.encode(progress=progress,**args)
        decoded=self.decode(encoded["encoding_id"],steps,args.get("seed",42),progress)
        encoded.update(decoded)
        return encoded

    def _pair(self,key,a,b,seed,align):
        ma,xa,sa=prepare_molecule(seed=seed,**a)
        mb,xb,sb=prepare_molecule(seed=seed,**b)
        set_seed(seed)
        ea=self._encode(key,ma,xa)
        eb=self._encode(key,mb,xb)
        alignment="Independent centered coordinate frames"
        if align:
            # Same trained global-node slots provide the point correspondence.
            pa=ea["zx"].numpy();pb=eb["zx"].numpy()
            ca=pa.mean(0);cb=pb.mean(0)
            u,_,vt=np.linalg.svd((pb-cb).T@(pa-ca))
            sign=np.diag([1.,1.,np.linalg.det(u@vt)])
            rotation=u@sign@vt  # SO(3), never a reflection that can invert chirality.
            eb["zx"]=torch.tensor((pb-cb)@rotation+ca,dtype=torch.float32)
            transformed=(xb-cb)@rotation+ca
            for i,point in enumerate(transformed):
                mb.GetConformer().SetAtomPosition(i,point.astype(float))
            alignment="Endpoint B aligned to A by a proper rotation of corresponding latent global-node slots"
        return ma,sa,mb,sb,ea,eb,alignment

    def interpolate(self,key,a,b,frames=11,steps=100,seed=42,align=True,progress=lambda *args:None):
        started=time.monotonic()
        ma,sa,mb,sb,ea,eb,alignment=self._pair(key,a,b,seed,align)
        results=[]
        for i,alpha in enumerate(np.linspace(0.,1.,frames)):
            progress("Decoding interpolation frame %d / %d"%(i+1,frames),0.15+0.8*i/frames)
            mixed={"key":key,"mol":ma,
                   "zx":torch.lerp(ea["zx"],eb["zx"],float(alpha)),
                   "zh":torch.lerp(ea["zh"],eb["zh"],float(alpha)),
                   "condition":torch.lerp(ea["condition"],eb["condition"],float(alpha))}
            count=int(np.rint((1-alpha)*ma.GetNumAtoms()+alpha*mb.GetNumAtoms()))
            output,_=self._decode(mixed,steps,seed,count)
            results.append({"alpha":round(float(alpha),4),"label":"t = %.2f"%alpha,"decoded":output,"latent":self.latent_payload(mixed)})
        return {"model":key,"model_name":MODEL_INFO[key]["name"],
                "a":molecule_payload(ma,sa),"b":molecule_payload(mb,sb),"frames":results,
                "alignment":alignment,"steps":steps,"seed":seed,
                "sampling":"Same decoder seed per frame; heavy-atom counts are linearly interpolated and rounded",
                "runtime_seconds":round(time.monotonic()-started,3),"kind":"interpolation","provenance":self.provenance(key)}

    def _replace_structural(self,encoded,zh):
        """Keep the seed semantics and apply the original structural normalization."""
        if encoded["key"]=="molflae":
            return zh
        semantic=encoded["condition"][:,MODEL_INFO["molalign3d"]["structural_dim"]:]
        model=self.load(encoded["key"])
        global_batch=torch.zeros(zh.shape[0],dtype=torch.long)
        normalized=model._match_Zh_to_unimol(zh,semantic.reshape(1,-1),global_batch,
                                             method=model.unimol_cfg.get("outnorm_method","scalar"))
        return torch.cat([normalized,semantic],dim=-1)

    def wander(self,key,smiles="",molblock="",frames=7,steps=100,seed=42,step_size=0.1,space="both",progress=lambda *args:None):
        started=time.monotonic()
        mol,coords,source=prepare_molecule(smiles,molblock,seed)
        original=self._encode(key,mol,coords)
        rng=torch.Generator(device="cpu").manual_seed(seed)
        walk_x=torch.zeros_like(original["zx"]);walk_h=torch.zeros_like(original["zh"])
        x_scale=(original["zx"]-original["zx"].mean(0)).square().mean().sqrt().clamp_min(1e-3)
        h_scale=original["zh"].square().mean().sqrt().clamp_min(1e-3)
        results=[]
        for i in range(frames):
            if i:
                if space in ("zx","both"):
                    noise=torch.randn(walk_x.shape,generator=rng)
                    walk_x+=noise-noise.mean(0)
                if space in ("zh","both"):
                    walk_h+=torch.randn(walk_h.shape,generator=rng)
            mixed=dict(original)
            mixed["zx"]=original["zx"]+step_size*x_scale*walk_x
            mixed["zh"]=original["zh"]+step_size*h_scale*walk_h
            mixed["condition"]=original["condition"] if i==0 or space=="zx" else self._replace_structural(original,mixed["zh"])
            progress("Decoding random-walk step %d / %d"%(i+1,frames),0.15+0.8*i/frames)
            output,_=self._decode(mixed,steps,seed)
            results.append({"walk_step":i,"label":"Walk step %d"%i,"decoded":output,"latent":self.latent_payload(mixed)})
        return {"model":key,"model_name":MODEL_INFO[key]["name"],"input":molecule_payload(mol,source),
                "frames":results,"kind":"wander","steps":steps,"seed":seed,"space":space,"step_size":step_size,
                "sampling":"Anchored Gaussian random walk; step size relative to seed RMS; seed UniMol features and atom count fixed; same decoder seed per frame",
                "runtime_seconds":round(time.monotonic()-started,3),"provenance":self.provenance(key)}

    def atom_edit(self,key,smiles="",molblock="",radius=2,steps=100,seed=42,progress=lambda *args:None):
        started=time.monotonic()
        mol,coords,source=prepare_molecule(smiles,molblock,seed)
        encoded=self._encode(key,mol,coords)
        counts=[(delta,mol.GetNumAtoms()+delta) for delta in range(-radius,radius+1) if 2<=mol.GetNumAtoms()+delta<=80]
        results=[]
        for i,(delta,count) in enumerate(counts):
            progress("Decoding %d heavy atoms (%+d)"%(count,delta),0.15+0.8*i/len(counts))
            output,_=self._decode(encoded,steps,seed,count)
            results.append({"delta":delta,"label":"%d heavy atoms · %+d"%(count,delta),"decoded":output,"latent":self.latent_payload(encoded)})
        return {"model":key,"model_name":MODEL_INFO[key]["name"],"input":molecule_payload(mol,source),
                "frames":results,"kind":"atom-edit","steps":steps,"seed":seed,
                "sampling":"All latent coordinates and complete decoder features fixed; only requested heavy-atom count changes; same decoder seed",
                "runtime_seconds":round(time.monotonic()-started,3),"provenance":self.provenance(key)}

    def swap(self,key,a,b,steps=100,seed=42,align=True,progress=lambda *args:None):
        started=time.monotonic()
        ma,sa,mb,sb,ea,eb,alignment=self._pair(key,a,b,seed,align)
        results=[]
        combinations=[("A","A",ea,ea),("B","A",eb,ea),("A","B",ea,eb),("B","B",eb,eb)]
        for i,(geometry_name,feature_name,geometry,features) in enumerate(combinations):
            mixed={"key":key,"mol":geometry["mol"],"zx":geometry["zx"],"zh":features["zh"],"condition":features["condition"]}
            progress("Decoding Zx from %s / features from %s"%(geometry_name,feature_name),0.15+0.8*i/len(combinations))
            output,_=self._decode(mixed,steps,seed,geometry["mol"].GetNumAtoms())
            results.append({"geometry_source":geometry_name,"feature_source":feature_name,
                            "label":"Zx %s + Zh %s"%(geometry_name,feature_name),"decoded":output,"latent":self.latent_payload(mixed)})
        return {"model":key,"model_name":MODEL_INFO[key]["name"],"a":molecule_payload(ma,sa),"b":molecule_payload(mb,sb),
                "frames":results,"kind":"swap","alignment":alignment,"steps":steps,"seed":seed,
                "sampling":"Swap complete feature conditioning (structural Zh plus seed UniMol semantics); atom count follows the geometry donor; same decoder seed",
                "runtime_seconds":round(time.monotonic()-started,3),"provenance":self.provenance(key)}
