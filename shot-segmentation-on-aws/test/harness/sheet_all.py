#!/usr/bin/env python3
"""Ground truth: contact sheets for every group in audit_groups.json
(built by enumerate_audit.py; frames fetched by fetch_frames.sh)."""
import json, glob
from PIL import Image, ImageDraw
gs = json.load(open("audit_groups.json"))
gs.sort(key=lambda g: (g["run"], -g["n"]))
W,H,LAB,K = 210,124,150,6
per = 9
for page in range((len(gs)+per-1)//per):
    chunk = gs[page*per:(page+1)*per]
    sh = Image.new("RGB",(LAB+K*W, len(chunk)*H),(16,16,16)); d=ImageDraw.Draw(sh)
    for r,g in enumerate(chunk):
        import statistics as st
        dm = st.median(g["dur"])
        d.text((4,r*H+H//2-14),f"{g['run']}",fill=(255,220,90))
        d.text((4,r*H+H//2-2),f"n={g['n']} occ={g['occ']}",fill=(200,200,200))
        d.text((4,r*H+H//2+10),f"dur~{dm:.1f}s",fill=(160,200,255))
        idx = [round(i*(len(g['shots'])-1)/(K-1)) for i in range(K)]
        for c,i in enumerate(dict.fromkeys(idx)):
            sid=g['shots'][i]
            fs=sorted(glob.glob(f"{g['batch']}/frames/{g['run']}/shot_{sid:04d}/*.jpg"))
            if not fs: continue
            sh.paste(Image.open(fs[len(fs)//2]).convert("RGB").resize((W,H-11)),(LAB+c*W,r*H))
            d.text((LAB+c*W+3,r*H+H-10),f"s{sid}",fill=(150,190,255))
    sh.save(f"audit_page{page}.jpg",quality=88); print(f"audit_page{page}.jpg {sh.size}")
