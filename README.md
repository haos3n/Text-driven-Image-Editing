# Text-driven-Image-Editing

This repository implements **prompt-based image editing** using Stable Diffusion and cross-attention control, inspired by the [Prompt-to-Prompt](https://arxiv.org/abs/2208.01626) method.
<img width="1168" height="727" alt="figure1" src="https://github.com/user-attachments/assets/a9d0c60a-216e-445d-9a55-964823ee1a11" />

It allows you to:
- Replace words in the prompt while keeping structure and layout.
- Refine/extend descriptions without altering overall composition.
- Reweight specific words to emphasize or de-emphasize their effect.
---
<img width="1392" height="549" alt="inr1" src="https://github.com/user-attachments/assets/2fc6e85d-db04-4d7b-b3d9-79ec7ea582db" />

## 📂 Project Structure
```
├── Text-driven-image-editing.py  # Main script
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── examples/                     # Output examples (optional)
```

## 📦 Installation
pip install diffusers==0.20.0 transformers accelerate safetensors Pillow

## 🚀 Usage
1. Replace Words (Preserve Layout)
```
python text_edit_sd_attn.py \
  --mode replace \
  --source-prompt "a photo of a lion roaring with a big mouth" \
  --edited-prompt "a photo of a tiger roaring with a big mouth" \
  --inject-steps 25 \
  --seed 123 \
  --out out_replace.png
```
<img width="3347" height="800" alt="tc2" src="https://github.com/user-attachments/assets/2db8c156-fd56-4a7c-8aeb-e07343780772" />
<img width="3347" height="800" alt="tc3" src="https://github.com/user-attachments/assets/4dee4028-1a18-4ece-b216-962e8d7ca8ca" />

2. Refine / Extend Description
```
python text_edit_sd_attn.py \
  --mode refine \
  --source-prompt "a car on the side of the street" \
  --edited-prompt "a red ferrari car on the side of the street in New York" \
  --inject-steps 25 \
  --seed 123 \
  --out out_refine.png
```
   <img width="1275" height="314" alt="ta1" src="https://github.com/user-attachments/assets/c6010b67-7f5e-40f6-a2ce-aa5c4a6f3efb" />

3. Reweight Specific Words
```
python text_edit_sd_attn.py \
  --mode reweight \
  --edited-prompt "a smiling teddy bear" \
  --scale-token "smiling" \
  --scale-factor 1.8 \
  --seed 123 \
  --out out_reweight.png
```
![ts1](https://github.com/user-attachments/assets/38b24daa-1e82-4014-af1f-22a8708dbf3e)


