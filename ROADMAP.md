# 🗺️ ROADMAP.md — Kab Kya Karna Hai

> Har hafte ka specific kaam — ek ek step

---

## 🚦 Overview

```
Phase 0 ✅  Structure      DONE (Aaj)
Phase 1 ⏳  Core Engine    Week 1-2
Phase 2 ⏳  Training       Week 3-4
Phase 3 ⏳  Terminal UI    Week 5
Phase 4 ⏳  Electron App   Month 2
Phase 5 ⏳  Smart Features Month 3
Phase 6 ⏳  Polish+Launch  Month 4
```

---

## ✅ Phase 0 — Already Done (Aaj)

```
[x] Project structure bani
[x] Tokenizer.py — custom tokenizer
[x] Transformer.py — full architecture
    [x] MultiHeadAttention
    [x] FeedForward
    [x] TransformerBlock
    [x] MyAI main class
[x] main.py — setup/chat/info commands
[x] Docs folder — planning documents
[x] numpy pe run ho raha hai (test passed)
```

---

## ⏳ Phase 1 — Core Engine (Week 1-2)

### Week 1 — Data System

```
Day 1:
[ ] data/ folder banao
[ ] data/raw/conversations.txt banao
[ ] 50 conversation pairs likho (Urdu/English)
[ ] DATA.md padho — format samjho

Day 2:
[ ] data/data_prep.py banao
    [ ] load_conversations() function
    [ ] Text cleaning function
    [ ] Train/test split function

Day 3:
[ ] 100 more conversation pairs
[ ] data_prep.py test karo
[ ] Tokenizer pe naya data run karo

Day 4:
[ ] Tokenizer improve karo
    [ ] Urdu character support verify karo
    [ ] Punctuation handling
    [ ] Numbers handling

Day 5:
[ ] Full pipeline test karo:
    Data → Tokenizer → Model → Output
[ ] main.py mein data loading add karo

Day 6-7:
[ ] 200 total conversation pairs
[ ] Edge cases handle karo (empty input, symbols)
[ ] README.md banao (basic)
```

### Week 2 — Training Script

```
Day 8:
[ ] training/trainer.py banao (TRAINING.md se)
[ ] Cross entropy loss implement karo
[ ] Basic training loop

Day 9:
[ ] Training test karo — 10 pairs, 5 epochs
[ ] Loss print ho rahi hai?
[ ] Loss kam ho rahi hai?

Day 10:
[ ] Checkpoint saving (har 10 epochs)
[ ] Loss history save karo (loss_log.txt)
[ ] Training resume karo saved checkpoint se

Day 11:
[ ] Hyperparameter experiments:
    [ ] LR = 0.01  → Loss track karo
    [ ] LR = 0.001 → Loss track karo
    [ ] LR = 0.0001 → Loss track karo
    Best wala choose karo

Day 12:
[ ] 50 pairs pe train karo, 50 epochs
[ ] Chat karo — fark hai?
[ ] Results document karo

Day 13-14:
[ ] Training script polish karo
[ ] main.py mein 'train' command add karo:
    python main.py train
```

---

## ⏳ Phase 2 — Better Training (Week 3-4)

### Week 3 — Proper Backpropagation

```
[ ] Loss functions improve karo
[ ] Proper gradient descent implement karo
    (Simplified se proper backprop)
[ ] Adam optimizer add karo (better than SGD)
[ ] Learning rate scheduler add karo

Adam Optimizer (copy karo):
----------------------------------------
m = 0, v = 0, t = 0
beta1 = 0.9, beta2 = 0.999, eps = 1e-8

t += 1
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad**2
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
weights -= lr * m_hat / (sqrt(v_hat) + eps)
----------------------------------------
```

### Week 4 — Data Pipeline + Memory

```
[ ] 500 conversation pairs
[ ] Data augmentation (same meaning, diff words)
[ ] Conversation memory add karo:
    - SQLite database
    - Purani baatein store karo
    - Context window mein include karo

Memory Structure:
-----------------
memory.db
  └── conversations table
       ├── id
       ├── timestamp
       ├── user_input
       └── ai_response
```

---

## ⏳ Phase 3 — Terminal UI (Week 5)

```
[ ] interface/terminal_ui.py banao (DESIGN.md se)
[ ] Clear screen function
[ ] Colored output (ANSI codes)
[ ] Typing animation effect
[ ] Status bar (vocab size, loss, model info)
[ ] ASCII header/logo
[ ] Input prompt styling

Test:
python interface/terminal_ui.py

Result:
╔══════════════════════════╗
║  ◈  M Y A I  ◈           ║
╚══════════════════════════╝

>> hello

◈ hello kya haal hai...
```

---

## ⏳ Phase 4 — Electron Desktop App (Month 2)

### Week 6-7 — Electron Setup

```
[ ] Node.js install karo (nodejs.org)
[ ] interface/electron/ folder
[ ] npm init
[ ] Electron install: npm install electron

package.json:
{
  "name": "myai",
  "main": "main.js",
  "scripts": {
    "start": "electron ."
  }
}

[ ] main.js — basic window
[ ] index.html — basic page
[ ] Test: npm start → Window khule
```

### Week 8 — UI Build

```
[ ] index.html — full layout (DESIGN.md se)
[ ] Neural Orb CSS animation
[ ] Chat interface
[ ] Sidebar (stats, navigation)
[ ] Dark theme (#000000 base)
[ ] Cyan/purple gradient (#00D4FF → #7B2FBE)
[ ] Orbitron + JetBrains Mono fonts
```

### Week 9 — Python Bridge

```
[ ] preload.js — Python spawn
[ ] IPC communication setup:
    Electron → Python: user input
    Python → Electron: AI response

[ ] Loading state (orb animation jab soch raha ho)
[ ] Error handling
[ ] Full flow test: Type → Python → Response → Display
```

---

## ⏳ Phase 5 — Smart Features (Month 3)

```
Week 10:
[ ] Voice Input (Web Speech API — browser built-in)
[ ] Voice Output (pyttsx3 — offline TTS)

Week 11:
[ ] PC Commands add karo:
    "notepad kholo"   → subprocess.Popen(['notepad'])
    "calculator"      → subprocess.Popen(['calc'])
    "volume barhao"   → pyautogui (Windows API)

Week 12:
[ ] Custom persona system:
    [ ] Name setting (tum naam do)
    [ ] Personality settings
    [ ] Voice selection (male/female)
```

---

## ⏳ Phase 6 — Polish & Launch (Month 4)

```
[ ] 1000+ training pairs
[ ] Model fine-tuned aur tested
[ ] Bug fixes
[ ] Performance optimization
[ ] Installer banao (.exe — electron-builder)
[ ] Demo video record karo
[ ] TikTok/Instagram pe share karo
[ ] GitHub pe upload karo (source code)
```

---

## 📊 Progress Tracker

```
Mark karte jao:

Phase 0  [##########] 100% ✅
Phase 1  [          ]   0%
Phase 2  [          ]   0%
Phase 3  [          ]   0%
Phase 4  [          ]   0%
Phase 5  [          ]   0%
Phase 6  [          ]   0%
```

---

## 🏆 Milestones

```
🥉 Bronze:  Model run hota hai, kuch bolta hai
            (Phase 1 complete)

🥈 Silver:  Meaningful conversations karta hai
            (Phase 3 complete)

🥇 Gold:    Full desktop app, voice, PC control
            (Phase 5 complete)

👑 Legend:  Launch, users, revenue
            (Phase 6 complete)
```

---

## ⏰ Time Estimates (Realistic)

```
Agar roz 1-2 ghante do:

Phase 0: Done ✅
Phase 1: 2 weeks
Phase 2: 2 weeks
Phase 3: 1 week
Phase 4: 3 weeks
Phase 5: 2 weeks
Phase 6: 2 weeks

Total: ~3 months
```

---

## 💬 Last Baat

> Ek ek phase complete karo.
> Agli phase pe mat jaao jab tak current complete nahi.
> Chota progress bhi progress hai.
> Ruko mat — shuru karo aaj se.

**Aaj ka kaam: data/raw/conversations.txt banao aur 50 pairs likho.**
