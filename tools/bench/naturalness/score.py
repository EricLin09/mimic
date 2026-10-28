"""Offline CPU naturalness proxies. No implicit downloads or torch.hub execution.
UTMOS22/SpeechMOS is a naturalness proxy; DNSMOS P.835 is a noise-quality proxy.
"""
from __future__ import annotations
import argparse,ast,hashlib,importlib.util,json,math,os,pathlib,sys,time
sys.dont_write_bytecode=True
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
COLUMNS=('utmos_mean','utmos_se','dnsmos_sig','dnsmos_bak','dnsmos_ovrl','mos_n')
def mean_se(values):
 a=np.asarray(values,dtype=float)
 return {'mean':float(a.mean()) if len(a) else None,'se':float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else 0.,'n':len(a)}
def load16(path):
 import soundfile as sf,torch,torchaudio
 a,sr=sf.read(str(path),dtype='float32',always_2d=True);a=a.mean(axis=1)
 if not len(a) or not np.isfinite(a).all():raise ValueError('Empty/nonfinite waveform')
 if sr!=16000:a=torchaudio.functional.resample(torch.from_numpy(a),sr,16000).numpy()
 return a.astype(np.float32)
def content_lag(reference,candidate):
 # Candidate[t+lag] matches reference[t]; 10 ms energy, +-400 ms search.
 def env(x):
  n=len(x)//160
  e=np.sqrt(np.mean(x[:n*160].reshape(n,160)**2,axis=1)+1e-12)
  return np.maximum(20*np.log10(e),-80)
 a,b=env(reference),env(candidate);best=(-2.,0)
 for lag in range(-40,41):
  lo=max(0,-lag);hi=min(len(a),len(b)-lag)
  if hi-lo<20:continue
  aa,bb=a[lo:hi],b[lo+lag:hi+lag]
  if aa.std()<1e-6 or bb.std()<1e-6:continue
  best=max(best,(float(np.corrcoef(aa,bb)[0,1]),lag))
 return best[1]*160,best[0]
class NaturalnessScorer:
 def __init__(self,models=('utmos','dnsmos'),weights=None,threads=2):
  import torch
  self.torch=torch;self.models=tuple(models);self.weights=pathlib.Path(weights) if weights else ROOT/'weights'
  if not self.models or set(self.models)-{'utmos','dnsmos'}:raise ValueError('models must contain utmos and/or dnsmos')
  if threads<1:raise ValueError('threads must be positive')
  expected_weights={'utmos':('utmos22_strong_step7459_v1.pt','38aa51ab79e2a4e09a1449758a4b37e9cbb2e8235a49662a732d33a9ba1e9bff'),'dnsmos':('sig_bak_ovr.onnx','269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd')}
  fingerprints=[]
  for name in self.models:
   file,expected=expected_weights[name];actual=hashlib.sha256((self.weights/file).read_bytes()).hexdigest()
   if actual!=expected:raise RuntimeError(f'Weight hash mismatch: {file}')
   fingerprints.append(actual)
  self.cache_signature=hashlib.sha256(pathlib.Path(__file__).read_bytes()+''.join(fingerprints).encode()).hexdigest()
  torch.set_num_threads(threads);self.load_ms={}
  if 'utmos' in self.models:
   start=time.perf_counter()
   # Fully reviewed fixed local upstream source; never import its hub loader.
   path=ROOT/'vendor/fairseq_alt.py'
   expected='a554d47bebef0eb764a048d58dbb7e1e1fca316d627c571add54faa42fe5f45b'
   if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('Reviewed fairseq_alt source changed')
   spec=importlib.util.spec_from_file_location('_mimic_utmos_fairseq',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   path=ROOT/'vendor/utmos_model.py'
   if hashlib.sha256(path.read_bytes()).hexdigest()!='bffb8eba14e74e4f228684280cbd295ecffbfcd0414a5a460ca55c94f2f0a0e4':raise RuntimeError('Reviewed UTMOS source changed')
   import torchaudio
   tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UTMOS22Strong']
   ns={'torch':torch,'nn':torch.nn,'Tensor':torch.Tensor,'torchaudio':torchaudio,'Wav2Vec2Model':module.Wav2Vec2Model}
   exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
   self.utmos=ns['UTMOS22Strong']().cpu().eval()
   state=torch.load(self.weights/'utmos22_strong_step7459_v1.pt',map_location='cpu',weights_only=True)
   self.utmos.load_state_dict(state,strict=True);self.load_ms['utmos']=(time.perf_counter()-start)*1000
  if 'dnsmos' in self.models:
   import onnxruntime as ort
   so=ort.SessionOptions();so.intra_op_num_threads=threads;so.inter_op_num_threads=1;so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
   so.add_session_config_entry('session.intra_op.allow_spinning','0')
   self.dns=ort.InferenceSession(str(self.weights/'sig_bak_ovr.onnx'),so,providers=['CPUExecutionProvider'])
 def score_segment(self,segment):
  x=np.asarray(segment,dtype=np.float32);out={}
  if len(x)<16000:raise ValueError('At least 1s of input is required')
  with self.torch.inference_mode():
   if 'utmos' in self.models:
    t=time.perf_counter();out['utmos']=float(self.utmos(self.torch.from_numpy(x[None,:]),16000).item());out['utmos_ms']=(time.perf_counter()-t)*1000
  if 'dnsmos' in self.models:
   # Upstream repeats short audio until >=9.01 s. This is a proxy on a repeated
   # 2.5 s segment, not an independently validated native short-window metric.
   z=np.tile(x,int(math.ceil(144160/len(x))))[:144160][None,:]
   t=time.perf_counter();raw=self.dns.run(None,{self.dns.get_inputs()[0].name:z})[0][0]
   for key,val,poly in zip(['dnsmos_sig','dnsmos_bak','dnsmos_ovrl'],raw,[[-.08397278,1.22083953,.0052439],[-.13166888,1.60915514,-.39604546],[-.06766283,1.11546468,.04602535]]):out[key]=float(np.polyval(poly,val))
   out['dnsmos_ms']=(time.perf_counter()-t)*1000
  return out
 def score_file(self,path,reference=None,segment_s=2.5):
  if not math.isfinite(segment_s) or segment_s<1:raise ValueError('segment_s must be finite and >=1 second')
  y=load16(path);r=load16(reference) if reference else y;lag,corr=content_lag(r,y) if reference else (0,1.)
  n=int(round(segment_s*16000));rows=[]
  for start in range(0,len(r)-n+1,n):
   ref=r[start:start+n]
   if reference and 20*np.log10(np.sqrt(np.mean(ref*ref)+1e-12)) < -45:continue
   a=start+lag
   if a<0 or a+n>len(y):continue
   rows.append({'start_s':start/16000,'lag_s':lag/16000,**self.score_segment(y[a:a+n])})
  metric_keys=['utmos','dnsmos_sig','dnsmos_bak','dnsmos_ovrl']
  stats={k:mean_se([z[k] for z in rows]) for k in metric_keys if rows and k in rows[0]}
  return {'path':str(path),'reference':str(reference) if reference else None,'segment_s':segment_s,'lag_s':lag/16000,'alignment_corr':corr,'segments':rows,'stats':stats}
 def optional_columns(self,path,reference):
  r=self.score_file(path,reference)
  if not r['segments']:return {**{k:float('nan') for k in COLUMNS if k!='mos_n'},'mos_n':0}
  stat=r['stats'];missing=float('nan')
  return {'utmos_mean':stat.get('utmos',{}).get('mean',missing),'utmos_se':stat.get('utmos',{}).get('se',missing),**{k:stat.get(k,{}).get('mean',missing) for k in ['dnsmos_sig','dnsmos_bak','dnsmos_ovrl']},'mos_n':len(r['segments'])}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('wav',nargs='+',type=pathlib.Path);p.add_argument('--reference',type=pathlib.Path);p.add_argument('--models',default='utmos,dnsmos');p.add_argument('--weights',type=pathlib.Path);p.add_argument('--segment-s',type=float,default=2.5);p.add_argument('--json',type=pathlib.Path)
 a=p.parse_args();s=NaturalnessScorer(a.models.split(','),a.weights);result=[s.score_file(f,a.reference,a.segment_s) for f in a.wav]
 text=json.dumps(result,indent=2,ensure_ascii=False)
 if a.json:a.json.write_text(text+'\n',encoding='utf-8')
 else:print(text)
if __name__=='__main__':main()
