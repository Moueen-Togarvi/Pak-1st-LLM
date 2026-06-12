# 🧠 MyAI — Master Plan
> Tumhara apna AI — zero dependency, zero cloud, 100% apna

---

## 📌 Project Ka Ek Line Summary
**Ek transformer-based AI banana jo Claude jaisi structure rakhta ho, CPU pe chale, aur sirf tumhara data use kare.**

---

## 🗂️ Saare Documents Ki List

| File | Kya Hai |
|------|---------|
| `PLAN.md` | Yeh file — overall roadmap |
| `ARCHITECTURE.md` | Technical structure — har layer ki detail |
| `DATA.md` | Data kaise banana, format, kahan rakhna |
| `TRAINING.md` | Model train kaise karna step by step |
| `DESIGN.md` | UI/UX — Stonic se behtar interface |
| `ROADMAP.md` | Timeline — kab kya karna hai |

---

## 🎯 Project Goals

### Must Have (MVP)
- [ ] Apna Tokenizer (koi library nahi)
- [ ] Apna Transformer model (numpy only)
- [ ] CPU pe chale (GPU nahi chahiye)
- [ ] Apna data de sako
- [ ] Model save/load ho
- [ ] Basic chat kare

### Should Have (Version 2)
- [ ] Training script
- [ ] Urdu + English dono samjhe
- [ ] Memory — purani baatein yaad rakhe
- [ ] Simple terminal UI

### Nice to Have (Version 3)
- [ ] Electron desktop app (Stonic jaisa)
- [ ] Voice input/output
- [ ] Sci-fi animated interface
- [ ] PC automation commands

---

## 📁 Final Project Structure

```
myai/
│
├── 📄 main.py              ← Yahan se sab run hoga
│
├── 📁 tokenizer/
│   ├── tokenizer.py        ← Words → Numbers
│   └── vocab.json          ← Saved vocabulary (auto-generate hogi)
│
├── 📁 model/
│   ├── transformer.py      ← Poora AI brain
│   ├── weights.npz         ← Trained weights (auto-save)
│   └── weights_config.json ← Model config (auto-save)
│
├── 📁 training/
│   ├── trainer.py          ← Training loop
│   └── loss.py             ← Loss functions
│
├── 📁 data/
│   ├── raw/                ← Tumhara raw data (txt files)
│   ├── processed/          ← Processed training data
│   └── data_prep.py        ← Data processing script
│
├── 📁 interface/
│   ├── terminal_ui.py      ← Pretty terminal chat
│   └── electron/           ← Desktop app (Phase 3)
│
├── 📁 docs/
│   ├── PLAN.md             ← Yeh file
│   ├── ARCHITECTURE.md
│   ├── DATA.md
│   ├── TRAINING.md
│   ├── DESIGN.md
│   └── ROADMAP.md
│
└── 📄 requirements.txt     ← Sirf: numpy
```

---

## ⚡ Quick Start (Abhi Karo)

```bash
# Step 1: Folder mein jao
cd myai

# Step 2: Sirf ek library chahiye
pip install numpy

# Step 3: Setup karo
python main.py setup

# Step 4: Baat karo (untrained hoga abhi)
python main.py chat

# Step 5: Info dekho
python main.py info
```

---

## 🚦 Current Status

```
✅ Phase 0 — Structure    DONE
⏳ Phase 1 — Data         TUMHARA KAAM
⏳ Phase 2 — Training     NEXT
⏳ Phase 3 — Interface    BAAD MEIN
```

---

## 💡 Ek Important Baat

> Yeh AI pehle din se smart nahi hoga.
> Jaise baccha pehle din bolna nahi jaanta.
> Jitna zyada data doge — utna zyada seekhega.
> Start karo, ruko mat.
