# NeuroFusionNet Tumor App

Responsive single-page Streamlit app for the existing NeuroFusionNet checkpoint.

## 1) Copy the original tested model code and checkpoint

Open PowerShell:

```powershell
cd "D:\NeuroFusionNet Tumor App"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup_from_thesis.ps1
```

This copies `Models`, `src`, `configs` and `Results\NeuroFusionNet\Checkpoints\best_model.pth` from `D:\Thesis`. The app uses the existing tested model architecture and Grad-CAM implementation; it does not train or fine-tune.

## 2) Install UI dependencies

Recommended first run: reuse the already working thesis environment to avoid downloading PyTorch again:

```powershell
& "D:\Thesis\.venv\Scripts\python.exe" -m pip install -r "D:\NeuroFusionNet Tumor App\requirements.txt"
```

If you later create a separate `.venv`, install the same compatible PyTorch/TorchVision CUDA build as the thesis environment first (reported versions: PyTorch 2.15.0+cu134 and TorchVision 0.29.2+cu134), then install `requirements.txt`.

## 3) Run

```powershell
cd "D:\NeuroFusionNet Tumor App"
& "D:\Thesis\.venv\Scripts\python.exe" -m streamlit run app.py
```

Streamlit will show a local URL, usually `http://localhost:8501`.

## Project contents

- `app.py`: single-page bilingual responsive application
- `requirements.txt`: web UI dependencies
- `setup_from_thesis.ps1`: copies tested model, XAI, config and checkpoint
- `checkpoints/best_model.pth`: copied trained checkpoint
- `Models/`, `src/`, `configs/`: copied from thesis project
- `outputs/`: saved local analysis images and JSON reports

## Important notes

- The class order is read from `configs/config.yaml` and checked against checkpoint metadata.
- Image preprocessing uses RGB, configured resize (default 224), and ImageNet mean/std normalization. Confirm this exactly matches the original thesis validation/test transform before research use.
- Grad-CAM and Grad-CAM++ are generated for the predicted class using the thesis project's `src.xai.generate_cam_maps`.
- The displayed softmax score is not a clinically calibrated probability.
- This is a research prototype, not a medical device or diagnostic service.
- Most brain tumours have no known prevention method. Educational text is general information only.
- Local Streamlit mode processes the uploaded image on the local machine. Before deployment, add authentication, secure transport, privacy notice, retention controls and appropriate data protection safeguards. Do not deploy with real patient uploads until these are implemented.

## Educational references

- NHS: https://www.nhs.uk/conditions/brain-tumours/
- American Cancer Society: https://www.cancer.org/cancer/types/brain-spinal-cord-tumors-adults/causes-risks-prevention.html
- Mayo Clinic: https://www.mayoclinic.org/diseases-conditions/brain-tumor/symptoms-causes/syc-20350084
