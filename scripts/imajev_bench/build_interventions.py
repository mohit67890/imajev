"""Deterministic code-native panels for controlled image interventions; draft only."""
import argparse,copy,hashlib,json,random,sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from imajev_bench.schema import validate_records,model_payload
from imajev_bench.runner import digest,file_digest


def panel(scene,count=None,ab=None,accent=False):
    im=Image.new('RGB',(720,480),'white');d=ImageDraw.Draw(im)
    font=ImageFont.load_default(size=28);big=ImageFont.load_default(size=38)
    d.text((35,25),f'Inspection panel {scene+1}',font=font,fill='#202b3d')
    d.line((35,80,685,80),fill='#cccccc',width=2)
    if count is not None:
        d.text((35,103),'Circle inventory',font=font,fill='#202b3d')
        for i in range(12):
            x=65+(i%6)*110;y=175+(i//6)*115
            d.ellipse((x,y,x+65,y+65),fill='#2466cc' if i<count else '#cccccc',outline='#202b3d',width=2)
        d.rectangle((645,420,680,450),fill='#cc7722' if accent else '#448866')
    else:
        for j,(label,value) in enumerate(zip(('A','B','C'),ab)):
            y=135+j*95;d.rectangle((80,y,635,y+75),outline='#aaaaaa',width=2)
            d.text((115,y+14),label,font=big,fill='#202b3d');d.text((400,y+14),str(value),font=big,fill='#202b3d')
    return im


def build(output):
    output.mkdir(parents=True,exist_ok=False);(output/'assets').mkdir();rows=[];groups=[]
    for scene in range(12):
        family=('count_threshold','numeric_sum','ordinal_band')[scene//4];j=scene%4
        if family=='count_threshold':
            n=2+j;atmost=j%2==0
            state={'rule':f"Return true if the number of blue circles is {'at most '+str(n) if atmost else 'at least '+str(n+1)}; otherwise false. Count only blue circles; ignore gray circles and rectangular color swatches."}
            field={'id':'decision','type':'boolean','question':'Does this panel meet the supplied rule?'}
            values=[atmost,not atmost,atmost]
            images=[panel(scene,count=n),panel(scene,count=n+1),panel(scene,count=n,accent=True)]
            change='Recolor one gray circle blue';irrelevant='Change only the rectangular swatch from green to orange'
        elif family=='numeric_sum':
            a=11+3*j;b=8+j;cap=a+b
            state={'rule':f"Use the numbers in rows A and B only. Choose accept if A+B is no greater than {cap}; otherwise choose reject. Ignore row C."}
            field={'id':'decision','type':'choice','question':'Which decision follows from the supplied rule?','options':[{'value':'accept','description':'A+B is within the stated bound.'},{'value':'reject','description':'A+B exceeds the stated bound.'}]}
            if j%2:field['options'].reverse()
            values=['accept','reject','accept'];images=[panel(scene,ab=(a,b,37)),panel(scene,ab=(a,b+1,37)),panel(scene,ab=(a,b,92))]
            change='Increase row B by one';irrelevant='Change only excluded row C from 37 to 92'
        else:
            lo=2+j;hi=lo+2;n=lo if j%2==0 else hi;before=1 if j%2==0 else 2
            state={'rule':f'Count blue circles only. Level 1: count at most {lo}. Level 2: count {lo+1} through {hi}, inclusive. Level 3: count greater than {hi}. Ignore gray circles and rectangular swatches.'}
            field={'id':'decision','type':'ordinal','question':'Which level applies under the supplied rule?','levels':[{'value':1,'description':'Lower count band.'},{'value':2,'description':'Middle count band.'},{'value':3,'description':'Upper count band.'}]}
            values=[before,before+1,before];images=[panel(scene,count=n),panel(scene,count=n+1),panel(scene,count=n,accent=True)]
            change='Recolor one gray circle blue across a band boundary';irrelevant='Change only the rectangular swatch from green to orange'
        gid=f'intervention-{scene+1:02d}';ids=[]
        for variant,(im,gold) in enumerate(zip(images,values)):
            rid='case-'+hashlib.sha256(f'{scene}:{variant}:2026'.encode()).hexdigest()[:12];ids.append(rid)
            path=output/'assets'/f'{rid}.png';im.save(path)
            rows.append({'id':rid,'group_id':gid,'track':'joint','family':family,'split':'dev','images':[{'path':'assets/'+path.name,'sha256':file_digest(path)}],
                'request':{'request_id':rid,'state':copy.deepcopy(state),'fields':[copy.deepcopy(field)]},'gold':gold,'annotation_status':'draft',
                'provenance':{'synthetic':True,'generator':'build_interventions.py','variant':('base','relevant_change','irrelevant_change')[variant],'label_origin':'deterministic provisional author label','scene':scene}})
        groups.append({'group_id':gid,'family':family,'base':ids[0],'flip':ids[1],'invariant':ids[2],'relevant_change':change,'irrelevant_change':irrelevant})
    rows=validate_records(rows,output)
    (output/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest={'status':'synthetic_development_draft','groups':groups,'records_sha256':file_digest(output/'records.jsonl'),'generator_sha256':file_digest(Path(__file__)),'relationship_gold':'Provisional generator assertions; review required','independence_note':'12 declared groups share three templates; unique images do not imply statistical independence.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    packet=[{'id':r['id'],'input_sha256':digest(model_payload(r)),**model_payload(r)} for r in rows];random.Random(6291).shuffle(packet)
    (output/'blind.json').write_text(json.dumps({'protocol':'imajev-modality-probe-v1','asset_root':str(output.resolve()),'items':packet},indent=2)+'\n')
    return rows,manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r,m=build(a.output);print({'records':len(r),'groups':len(m['groups'])})
