"""Explicit asset download only; does not import or execute downloaded code."""
import pathlib,urllib.request,json,hashlib
root=pathlib.Path(__file__).parent
items={
 'weights/utmos22_strong_step7459_v1.pt':('https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt',450_000_000),
 'weights/sig_bak_ovr.onnx':('https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx',2_000_000),
 'vendor/fairseq_alt.py':('https://raw.githubusercontent.com/tarepan/SpeechMOS/v1.2.0/speechmos/utmos22/fairseq_alt.py',30_000),
 'vendor/utmos_model.py':('https://raw.githubusercontent.com/tarepan/SpeechMOS/v1.2.0/speechmos/utmos22/strong/model.py',10_000),
 'vendor/LICENSE_SpeechMOS':('https://raw.githubusercontent.com/tarepan/SpeechMOS/v1.2.0/LICENSE',10_000),
 'vendor/LICENSE_UTMOS22':('https://raw.githubusercontent.com/sarulab-speech/UTMOS22/master/LICENSE',10_000),
 'vendor/LICENSE_fairseq':('https://raw.githubusercontent.com/facebookresearch/fairseq/main/LICENSE',10_000),
 'weights/LICENSE_DNSMOS_weights':('https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/LICENSE',30_000),
 'vendor/LICENSE_DNSMOS_code':('https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/LICENSE-CODE',10_000),
 'vendor/dnsmos_reference.py':('https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/dnsmos_local.py',30_000),
 'vendor/hubconf_reference.py':('https://raw.githubusercontent.com/tarepan/SpeechMOS/v1.2.0/hubconf.py',10_000),
}
rows=[]
for name,(url,limit) in items.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  with urllib.request.urlopen(url,timeout=120) as f,p.with_suffix(p.suffix+'.part').open('wb') as o:
   total=0
   while True:
    chunk=f.read(1024*1024)
    if not chunk:break
    total+=len(chunk)
    if total>limit:raise RuntimeError(name+' exceeds declared asset cap')
    o.write(chunk)
  p.with_suffix(p.suffix+'.part').replace(p)
 row=dict(path=name,url=url,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest());rows.append(row);print(row,flush=True)
(root/'weights/manifest.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
