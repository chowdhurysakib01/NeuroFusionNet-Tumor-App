from __future__ import annotations
import io, json, sys, time, zipfile, hashlib
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import yaml
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CHECKPOINT = ROOT / "checkpoints" / "best_model.pth"
CONFIG = ROOT / "configs" / "config.yaml"
OUTPUTS = ROOT / "outputs"
FALLBACK_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]

TEXT = {
"English": {
"tag":"AI-assisted brain MRI research prototype",
"title":"Brain MRI Analysis with Explainable AI",
"sub":"Upload a brain MRI image to view a four-class NeuroFusionNet prediction and its visual explanation.",
"upload":"Upload a brain MRI image",
"hint":"Choose an MRI image (JPG, JPEG, PNG, BMP or TIFF). Maximum 15 MB.",
"analyze":"Analyze MRI","clear":"Clear","result":"Analysis result","pred":"Model-predicted class",
"score":"Model score","probs":"Class probability scores","xai":"Explainable AI visualization",
"original":"Input MRI","cam":"Grad-CAM","campp":"Grad-CAM++","download":"Download analysis package",
"education":"Brain tumor: learn the basics","symptoms":"Possible symptoms","causes":"Causes and risk factors",
"prevention":"Prevention and next steps","contact":"Contact a doctor","contact_text":"A doctor-contact feature is being planned.",
"soon":"Coming Soon","disclaimer":"Important medical notice",
"notice":"This is a research prototype, not a medical device or diagnostic service. An AI prediction or heatmap cannot confirm or rule out a brain tumour. Consult a qualified healthcare professional for MRI interpretation and medical advice.",
"wait":"Your analysis will appear here after you upload an image and select Analyze MRI.",
"upload_error":"Please upload an MRI image first.","working":"Analyzing image with NeuroFusionNet...",
"model_error":"Could not load the model. Check the setup instructions and checkpoint files.",
"xai_error":"Prediction succeeded, but XAI generation failed. Verify the copied src/xai.py file.",
"score_note":"Softmax score from the model; it is not a clinically calibrated probability.",
"research_note":"Research output only — not a clinical diagnosis.",
"footer":"NeuroFusionNet · Academic research prototype",
"symptom_items":["New, persistent or progressively worsening headache","Seizures, including a first-time seizure","Persistent nausea or vomiting","Changes in vision, speech, hearing, balance or walking","New weakness, drowsiness, memory or personality changes"],
"cause_items":["For most brain tumours, the exact cause is unknown.","Some rare inherited genetic syndromes can increase risk.","Radiation exposure to the head, particularly previous radiotherapy, is a recognised risk factor for some tumours.","Having a risk factor does not mean a person will develop a brain tumour."],
"prevention_items":["There is no proven way to prevent most brain tumours.","Do not avoid medically necessary scans or treatment; discuss radiation-related questions with a clinician.","If you have a strong family history or a known inherited syndrome, ask a healthcare professional about your individual risk.","Seek medical assessment for persistent, unusual or worsening symptoms."],
"sources":"Sources"
},
"বাংলা": {
"tag":"AI-সহায়তাপ্রাপ্ত ব্রেইন MRI গবেষণা প্রোটোটাইপ",
"title":"Explainable AI দিয়ে ব্রেইন MRI বিশ্লেষণ",
"sub":"একটি ব্রেইন MRI ছবি আপলোড করে NeuroFusionNet-এর চার-শ্রেণির পূর্বাভাস ও visual explanation দেখুন।",
"upload":"ব্রেইন MRI ছবি আপলোড করুন",
"hint":"MRI ছবি নির্বাচন করুন (JPG, JPEG, PNG, BMP অথবা TIFF)। সর্বোচ্চ ১৫ MB।",
"analyze":"MRI বিশ্লেষণ করুন","clear":"মুছে ফেলুন","result":"বিশ্লেষণের ফলাফল","pred":"মডেলের পূর্বাভাস",
"score":"মডেল স্কোর","probs":"প্রতিটি শ্রেণির স্কোর","xai":"Explainable AI ভিজ্যুয়ালাইজেশন",
"original":"মূল MRI","cam":"Grad-CAM","campp":"Grad-CAM++","download":"বিশ্লেষণ রিপোর্ট ডাউনলোড",
"education":"ব্রেইন টিউমার: প্রয়োজনীয় তথ্য","symptoms":"সম্ভাব্য লক্ষণ","causes":"কারণ ও ঝুঁকির বিষয়",
"prevention":"প্রতিরোধ ও করণীয়","contact":"ডাক্তারের সঙ্গে যোগাযোগ","contact_text":"ডাক্তারের সঙ্গে যোগাযোগের সুবিধাটি ভবিষ্যতে যুক্ত করা হবে।",
"soon":"শীঘ্রই আসছে","disclaimer":"গুরুত্বপূর্ণ চিকিৎসা-সংক্রান্ত সতর্কতা",
"notice":"এটি একটি গবেষণামূলক প্রোটোটাইপ; এটি কোনো চিকিৎসা-যন্ত্র বা রোগ নির্ণয়ের সেবা নয়। AI-এর পূর্বাভাস বা heatmap দিয়ে ব্রেইন টিউমার নিশ্চিত করা বা বাতিল করা যায় না। MRI ও চিকিৎসা-সংক্রান্ত পরামর্শের জন্য যোগ্য চিকিৎসকের সঙ্গে যোগাযোগ করুন।",
"wait":"ছবি আপলোড করে বিশ্লেষণ শুরু করলে ফলাফল এখানে দেখা যাবে।",
"upload_error":"প্রথমে একটি MRI ছবি আপলোড করুন।","working":"NeuroFusionNet দিয়ে ছবি বিশ্লেষণ করা হচ্ছে...",
"model_error":"মডেল লোড করা যায়নি। Setup নির্দেশনা এবং checkpoint ফাইল পরীক্ষা করুন।",
"xai_error":"পূর্বাভাস পাওয়া গেছে, কিন্তু XAI তৈরি হয়নি। কপি করা src/xai.py ফাইল পরীক্ষা করুন।",
"score_note":"এটি মডেলের Softmax score; চিকিৎসাগতভাবে calibrated probability নয়।",
"research_note":"শুধু গবেষণার ফলাফল — চিকিৎসাগত রোগ নির্ণয় নয়।",
"footer":"NeuroFusionNet · একাডেমিক গবেষণা প্রোটোটাইপ",
"symptom_items":["নতুন, দীর্ঘস্থায়ী বা ক্রমশ বাড়তে থাকা মাথাব্যথা","খিঁচুনি বা জীবনে প্রথমবার খিঁচুনি হওয়া","দীর্ঘস্থায়ী বমিভাব বা বমি","দৃষ্টি, কথা, শ্রবণ, ভারসাম্য বা হাঁটার সমস্যায় পরিবর্তন","নতুন দুর্বলতা, অতিরিক্ত ঘুম, স্মৃতি বা আচরণে পরিবর্তন"],
"cause_items":["অধিকাংশ ব্রেইন টিউমারের সঠিক কারণ জানা যায়নি।","কিছু বিরল বংশগত জেনেটিক সিনড্রোম ঝুঁকি বাড়াতে পারে।","মাথায় রেডিয়েশন, বিশেষ করে আগে রেডিওথেরাপি নেওয়া, কিছু টিউমারের স্বীকৃত ঝুঁকির কারণ।","ঝুঁকির কোনো বিষয় থাকলেই যে টিউমার হবে, তা নয়।"],
"prevention_items":["অধিকাংশ ব্রেইন টিউমার প্রতিরোধের কোনো প্রমাণিত উপায় নেই।","চিকিৎসার প্রয়োজনে প্রয়োজনীয় স্ক্যান বা চিকিৎসা এড়িয়ে যাবেন না; রেডিয়েশন বিষয়ে চিকিৎসকের পরামর্শ নিন।","পরিবারে টিউমারের ইতিহাস বা পরিচিত বংশগত সিনড্রোম থাকলে ব্যক্তিগত ঝুঁকি নিয়ে চিকিৎসকের সঙ্গে কথা বলুন।","দীর্ঘস্থায়ী, অস্বাভাবিক বা ক্রমশ বাড়তে থাকা লক্ষণ হলে চিকিৎসা মূল্যায়ন নিন।"],
"sources":"তথ্যসূত্র"
}}
CLASS_BN = {"glioma":"গ্লিওমা","meningioma":"মেনিনজিওমা","notumor":"টিউমার শনাক্ত হয়নি","pituitary":"পিটুইটারি"}

def read_config():
    if CONFIG.exists():
        with open(CONFIG, encoding="utf-8-sig") as f:
            return yaml.safe_load(f) or {}
    return {}

@st.cache_resource(show_spinner=False)
def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")
    cfg = read_config()
    classes = list(cfg.get("dataset", {}).get("classes", FALLBACK_CLASSES))
    image_size = int(cfg.get("dataset", {}).get("image_size", 224))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != "neurofusionnet":
        raise ValueError(f"Expected neurofusionnet checkpoint, got {checkpoint.get('model_name')}")
    if checkpoint.get("class_names") is not None and list(checkpoint["class_names"]) != classes:
        raise ValueError("Checkpoint class order does not match config class order.")
    from Models.factory import build_model
    model = build_model("neurofusionnet", num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, classes, image_size, device, checkpoint

def prep(image, size):
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return tfm(image.convert("RGB")).unsqueeze(0)

def predict(model, tensor, device):
    tensor = tensor.to(device)
    if device.type == "cuda": torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    if device.type == "cuda": torch.cuda.synchronize(device)
    return int(np.argmax(probs)), probs, (time.perf_counter()-start)*1000, tensor

def norm_cam(cam):
    cam = np.squeeze(np.asarray(cam, dtype=np.float32))
    if cam.ndim != 2: raise ValueError(f"Invalid CAM shape: {cam.shape}")
    cam = np.nan_to_num(cam)
    cam -= cam.min()
    if cam.max() > 0: cam /= cam.max()
    return cam

def make_cam(model, tensor, target):
    from src.xai import generate_cam_maps
    result = {}
    for method, key in [("gradcam","gradcam"),("gradcam++","gradcampp")]:
        with torch.enable_grad():
            returned = generate_cam_maps(model, tensor.detach().clone(), target_class=target, method=method)
        if not isinstance(returned, tuple) or len(returned) != 2:
            raise RuntimeError("src.xai.generate_cam_maps must return (logits, cam_maps)")
        maps = returned[1]
        if isinstance(maps, np.ndarray): maps = [maps]
        if not maps: raise RuntimeError(f"No maps returned for {method}")
        result[key] = norm_cam(np.mean(np.stack([norm_cam(m) for m in maps]), axis=0))
    return result

def heatmap(cam):
    return Image.fromarray((plt.get_cmap("jet")(cam)[...,:3]*255).astype(np.uint8))

def overlay(image, cam):
    base = np.asarray(image.convert("RGB").resize((cam.shape[1],cam.shape[0])),dtype=np.float32)/255
    blend = np.clip(.58*base + .42*plt.get_cmap("jet")(cam)[...,:3],0,1)
    return Image.fromarray((blend*255).astype(np.uint8))

def package_download(image, report, cams):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        b=io.BytesIO(); image.save(b,format="PNG"); z.writestr("input_mri.png",b.getvalue())
        for key,cam in cams.items():
            for suffix,img in [("heatmap",heatmap(cam)),("overlay",overlay(image,cam))]:
                b=io.BytesIO(); img.save(b,format="PNG"); z.writestr(f"{key}_{suffix}.png",b.getvalue())
        z.writestr("analysis_report.json",json.dumps(report,indent=2,ensure_ascii=False))
    return buf.getvalue()

def theme_css(dark):
    bg,panel,fg,muted,border,soft = (
        ("#0c1222","#151e32","#f3f4fa","#aab5ca","#2b3650","#1b2740")
        if dark else ("#f7f7fc","#ffffff","#171936","#667085","#e7e5f0","#f1edff")
    )
    st.markdown(f"""
    <style>
    :root{{--bg:{bg};--panel:{panel};--fg:{fg};--muted:{muted};--border:{border};--soft:{soft};}}
    .stApp{{background:var(--bg);color:var(--fg)}}
    .block-container{{
        max-width:1160px;
        padding-top:3rem !important;
        padding-bottom:2rem;
        overflow:visible !important;
    }}
    .app-brand{{
        display:flex;
        align-items:center;
        gap:12px;
        min-height:88px;
        padding:12px 0 18px;
        overflow:visible;
    }}
    .app-brand-icon{{
        font-size:2.2rem;
        line-height:1.3;
        flex-shrink:0;
    }}
    .app-brand-title{{
        font-size:1.6rem;
        font-weight:850;
        line-height:1.4;
        color:var(--fg);
        margin:0;
        padding:0;
    }}
    .app-brand-sub{{
        color:var(--muted);
        font-size:.9rem;
        line-height:1.4;
        margin-top:4px;
    }}
    h1,h2,h3,h4,p,li,label,span{{color:var(--fg)}}
    .hero{{background:linear-gradient(120deg,var(--soft),var(--panel));border:1px solid var(--border);border-radius:22px;padding:28px;margin:12px 0 22px}}
    .hero small{{color:#8b5cf6;font-weight:800;letter-spacing:.06em;text-transform:uppercase}}
    .hero h1{{font-size:clamp(1.8rem,4vw,2.7rem);line-height:1.12;margin:.5rem 0;color:var(--fg)}}
    .hero p,.muted{{color:var(--muted)}}
    .panel,.edu{{background:var(--panel);border:1px solid var(--border);border-radius:17px;padding:18px;margin-bottom:14px}}
    .section{{font-size:1.35rem;font-weight:800;margin:1.35rem 0 .8rem}}
    .result{{background:linear-gradient(120deg,rgba(112,66,200,.15),rgba(16,185,129,.08));border:1px solid #8b5cf6;border-radius:16px;padding:18px}}
    .pred{{color:#8b5cf6;font-size:1.9rem;font-weight:850}}
    .score{{color:#16a36a;font-size:1.65rem;font-weight:850}}
    .edu{{height:100%}} .edu li{{color:var(--muted);margin-bottom:.5rem;line-height:1.45}}
    .contact{{background:linear-gradient(110deg,rgba(112,66,200,.18),var(--panel));border:1px solid #8b5cf6;border-radius:17px;padding:20px;margin-top:16px}}
    .notice{{background:rgba(245,158,11,.09);border:1px solid rgba(245,158,11,.4);border-radius:13px;padding:14px;margin-top:14px}}
    .footer{{color:var(--muted);text-align:center;border-top:1px solid var(--border);padding-top:18px;margin-top:28px;font-size:.86rem}}
    div[data-testid="stFileUploader"]{{background:var(--soft);border:1px dashed #8b5cf6;border-radius:14px;padding:10px}}
    div.stButton>button,div.stDownloadButton>button{{border-radius:11px;min-height:2.7rem;font-weight:700}}
    @media(max-width:700px){{.block-container{{padding-left:1rem;padding-right:1rem}}.hero{{padding:19px}}.panel,.edu{{padding:14px}}}}
    </style>""",unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="NeuroFusionNet | Brain MRI Analysis",page_icon="🧠",layout="wide")
    st.session_state.setdefault("language","English")
    st.session_state.setdefault("dark",False)
    st.session_state.setdefault("analysis",None)
    a,b,c=st.columns([5,1.4,1.5])
    with a:
        st.markdown("""
        <div class="app-brand">
            <div class="app-brand-icon">🧠</div>
            <div>
                <div class="app-brand-title">NeuroFusionNet</div>
                <div class="app-brand-sub">Brain Tumor MRI Analysis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with b:
        lang=st.selectbox("Language",["English","বাংলা"],index=0 if st.session_state.language=="English" else 1,label_visibility="collapsed")
        st.session_state.language=lang
    with c:
        st.session_state.dark=st.toggle("🌙 Dark mode",value=st.session_state.dark)
    theme_css(st.session_state.dark)
    t=TEXT[lang]

    st.markdown(f'<div class="hero"><small>{t["tag"]}</small><h1>{t["title"]}</h1><p>{t["sub"]}</p></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="section">⬆️ {t["upload"]}</div>',unsafe_allow_html=True)
    upload_col,button_col=st.columns([3.3,1.1],vertical_alignment="center")
    with upload_col:
        uploaded=st.file_uploader(t["hint"],type=EXTENSIONS,key="mri_upload")
    with button_col:
        analyze=st.button("🧠 "+t["analyze"],type="primary",use_container_width=True)
        clear=st.button("↻ "+t["clear"],use_container_width=True)
    if clear:
        st.session_state.analysis=None
        st.rerun()

    original=None
    if uploaded:
        raw=uploaded.getvalue()
        if len(raw)>15*1024*1024:
            st.error("Image exceeds 15 MB.")
            return
        try:
            original=Image.open(io.BytesIO(raw)).convert("RGB")
            st.image(original,caption=t["original"],width=360)
        except Exception:
            st.error("Could not read this image.")
            return

    if analyze:
        if original is None:
            st.warning(t["upload_error"])
        else:
            try:
                with st.spinner(t["working"]):
                    model,classes,size,device,ckpt=load_model()
                    tensor=prep(original,size)
                    idx,probs,latency,tensor_device=predict(model,tensor,device)
                    cams=make_cam(model,tensor_device,idx)
                    report={
                        "model":"NeuroFusionNet",
                        "checkpoint":"best_model.pth",
                        "checkpoint_epoch":ckpt.get("epoch"),
                        "predicted_class":classes[idx],
                        "model_score":float(probs[idx]),
                        "probabilities":{k:float(v) for k,v in zip(classes,probs)},
                        "inference_time_ms":float(latency),
                        "image_name":uploaded.name,
                        "image_sha256":hashlib.sha256(raw).hexdigest(),
                        "created_at":datetime.now().isoformat(timespec="seconds"),
                        "device":str(device),
                        "notice":"Research output only; not a clinical diagnosis."
                    }
                    st.session_state.analysis={"report":report,"image":original.copy(),"cams":cams,"classes":classes}
                    out=OUTPUTS/datetime.now().strftime("%Y%m%d_%H%M%S")
                    out.mkdir(parents=True,exist_ok=True)
                    original.save(out/"input_mri.png")
                    for k,cam in cams.items():
                        heatmap(cam).save(out/f"{k}_heatmap.png")
                        overlay(original,cam).save(out/f"{k}_overlay.png")
                    (out/"analysis_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
            except Exception as e:
                st.error(t["model_error"])
                with st.expander("Technical details"): st.exception(e)

    analysis=st.session_state.analysis
    if analysis:
        r=analysis["report"]; classes=analysis["classes"]
        st.markdown(f'<div class="section">📊 {t["result"]}</div>',unsafe_allow_html=True)
        l,rr=st.columns([1,1.6],gap="large")
        with l:
            label=CLASS_BN.get(r["predicted_class"],r["predicted_class"]) if lang=="বাংলা" else r["predicted_class"].replace("notumor","No Tumor").title()
            st.markdown(f'<div class="result"><div>{t["pred"]}</div><div class="pred">{label}</div><div>{t["score"]}</div><div class="score">{r["model_score"]*100:.2f}%</div></div>',unsafe_allow_html=True)
            st.caption(t["score_note"]); st.caption(t["research_note"])
        with rr:
            st.markdown("**"+t["probs"]+"**")
            for cls in classes:
                label=CLASS_BN.get(cls,cls) if lang=="বাংলা" else cls.replace("notumor","No Tumor").title()
                value=r["probabilities"][cls]
                st.markdown(f"<div style='display:flex;justify-content:space-between'><span>{label}</span><b>{value*100:.2f}%</b></div>",unsafe_allow_html=True)
                st.progress(value)
        st.markdown(f'<div class="section">🧠 {t["xai"]}</div>',unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1: st.image(analysis["image"],caption=t["original"],use_container_width=True)
        with c2: st.image(overlay(analysis["image"],analysis["cams"]["gradcam"]),caption=t["cam"],use_container_width=True)
        with c3: st.image(overlay(analysis["image"],analysis["cams"]["gradcampp"]),caption=t["campp"],use_container_width=True)
        st.caption(f"NeuroFusionNet · best_model.pth · {r['inference_time_ms']:.1f} ms · {r['device']}")
        st.download_button("⬇️ "+t["download"],data=package_download(analysis["image"],r,analysis["cams"]),file_name="neurofusionnet_analysis.zip",mime="application/zip",use_container_width=True)
    else:
        st.markdown(f'<div class="panel"><b>📊 {t["result"]}</b><p class="muted">{t["wait"]}</p></div>',unsafe_allow_html=True)

    st.markdown(f'<div class="section" id="education">📚 {t["education"]}</div>',unsafe_allow_html=True)
    cols=st.columns(3,gap="medium")
    for col,icon,title,items in zip(cols,["🧠","🔎","🛡️"],[t["symptoms"],t["causes"],t["prevention"]],[t["symptom_items"],t["cause_items"],t["prevention_items"]]):
        with col:
            lis="".join(f"<li>{x}</li>" for x in items)
            st.markdown(f'<div class="edu"><div style="font-size:1.7rem">{icon}</div><b>{title}</b><ul>{lis}</ul></div>',unsafe_allow_html=True)

    st.markdown(f'<div class="contact"><b>👨‍⚕️ {t["contact"]}</b><p class="muted">{t["contact_text"]}</p><span style="background:#7042c8;color:white;padding:6px 12px;border-radius:20px;font-weight:700">⏳ {t["soon"]}</span></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="notice"><b>⚠️ {t["disclaimer"]}</b><br>{t["notice"]}</div>',unsafe_allow_html=True)
    if lang=="English":
        st.caption("Sources: [NHS](https://www.nhs.uk/conditions/brain-tumours/) · [American Cancer Society](https://www.cancer.org/cancer/types/brain-spinal-cord-tumors-adults/causes-risks-prevention.html) · [Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/brain-tumor/symptoms-causes/syc-20350084)")
    else:
        st.caption("তথ্যসূত্র: [NHS](https://www.nhs.uk/conditions/brain-tumours/) · [American Cancer Society](https://www.cancer.org/cancer/types/brain-spinal-cord-tumors-adults/causes-risks-prevention.html) · [Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/brain-tumor/symptoms-causes/syc-20350084)")
    st.markdown(f'<div class="footer">{t["footer"]} · © {datetime.now().year}</div>',unsafe_allow_html=True)

if __name__=="__main__":
    main()
