# 🏗️ ARCHITECTURE.md — MyAI Ki Technical Structure

> Har layer kya kaam karti hai — bilkul simple zabaan mein

---

## 🧠 Claude Vs MyAI — Comparison

| Cheez | Claude | MyAI (Tumhara) |
|-------|--------|----------------|
| Parameters | ~100 Billion | ~400K |
| Layers | 96 | 2 |
| Attention Heads | 96 | 4 |
| d_model | 8192 | 128 |
| Training Data | Petabytes | Tumhara data |
| GPU | Thousands | ❌ CPU only |
| Structure | Same ✅ | Same ✅ |

**Structure bilkul wahi hai — sirf size choti hai.**

---

## 📊 Data Flow — Ek Word Kaise Banta Hai

```
Tum likhte ho: "mera naam kya hai"
        │
        ▼
┌───────────────────┐
│   TOKENIZER       │  "mera naam kya hai"
│   Words → IDs     │  → [2, 14, 23, 11, 8, 3]
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  TOKEN EMBEDDING  │  Har number ek vector banta hai
│  IDs → Vectors    │  [2] → [0.2, -0.5, 0.8, ...]
│  (vocab × d_model)│  128 numbers ka ek word
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  POSITIONAL ENC   │  "kya" position 3 pe hai
│  Position sense   │  Sine/cosine waves se
└───────────────────┘
        │
        ▼
┌───────────────────────────────┐
│   TRANSFORMER BLOCK × 2       │
│                               │
│  ┌─────────────────────────┐  │
│  │  MULTI-HEAD ATTENTION   │  │  "kya" → "naam" se related hai
│  │  4 heads × d_k(32)      │  │  Har head alag angle se dekhe
│  └─────────────────────────┘  │
│           + (residual)        │
│  ┌─────────────────────────┐  │
│  │  LAYER NORMALIZATION    │  │  Values normalize karo
│  └─────────────────────────┘  │
│  ┌─────────────────────────┐  │
│  │  FEED FORWARD           │  │  128 → 512 → 128
│  │  128 → 512 → 128        │  │  Process karo
│  └─────────────────────────┘  │
│           + (residual)        │
│  ┌─────────────────────────┐  │
│  │  LAYER NORMALIZATION    │  │
│  └─────────────────────────┘  │
└───────────────────────────────┘
        │
        ▼
┌───────────────────┐
│  OUTPUT PROJ      │  Vector → Vocabulary scores
│  d_model → vocab  │  128 → 49 numbers
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  SOFTMAX          │  Scores → Probabilities
│  + SAMPLING       │  "hoon" = 45%, "hai" = 30%...
└───────────────────┘
        │
        ▼
   Next word: "hoon"
```

---

## 🔍 Har Layer Ki Detail

### 1. Tokenizer
```
Kaam: Text ko numbers mein badalna

Input:  "mera naam AI hai"
Output: [2, 14, 23, 45, 8, 3]
         ↑                  ↑
        BOS               EOS

Special Tokens:
<PAD> = 0  → Empty space
<UNK> = 1  → Unknown word
<BOS> = 2  → Sentence start
<EOS> = 3  → Sentence end
<SEP> = 4  → Separator
```

---

### 2. Token Embedding
```
Kaam: Har word ko ek vector mein badalna

Socho: Har word ka ek "address" hota hai
       128 dimensions mein

"mera"  → [0.2, -0.5, 0.8, 0.1, ...]  (128 numbers)
"naam"  → [0.9,  0.3, -0.2, 0.7, ...] (128 numbers)
"AI"    → [-0.1, 0.8,  0.5, -0.3, ...] (128 numbers)

Training mein yeh vectors seekhe jate hain.
Similar words → similar vectors
```

---

### 3. Positional Encoding
```
Kaam: Word ki position batana

Problem: "Ali ne Hamza ko mara"
         vs
         "Hamza ne Ali ko mara"
         
Dono mein same words hain — position important hai!

Solution: Sine aur cosine waves
Position 0: [sin(0), cos(0), sin(0/100), ...]
Position 1: [sin(1), cos(1), sin(1/100), ...]
...

Yeh fixed hai — train nahi hoti
```

---

### 4. Multi-Head Attention ⭐ (Sabse Important)
```
Kaam: Har word ko samajhna context mein

"Main khana khata hoon kyunki bhookha hoon"
                                    ↑
               "bhookha" → "Main" se connected hai
               Attention yeh connection dhundti hai

4 Heads = 4 Alag Perspectives:
Head 1: Grammar connections
Head 2: Meaning connections  
Head 3: Position connections
Head 4: Topic connections

Math:
Q = x × W_q  (Query:  "Main kya dhoond raha hoon?")
K = x × W_k  (Key:    "Mujh mein kya hai?")
V = x × W_v  (Value:  "Jo mila wo do")

Score = Q × K^T / √d_k
Attention = softmax(Score) × V
```

---

### 5. Feed Forward Network
```
Kaam: Information process karna

Structure:
128 → [Linear] → 512 → [GELU] → [Linear] → 128

GELU activation:
- ReLU se better
- Claude bhi GELU use karta hai
- Smooth non-linearity

Residual Connection:
output = LayerNorm(x + FFN(x))
Yeh training stable rakhta hai
```

---

### 6. Layer Normalization
```
Kaam: Values ko normalize karna

Bina normalization: values bohot bari ya choti ho jati hain
Training unstable hoti hai

Formula:
y = γ × (x - mean) / (std + ε) + β

γ (gamma) aur β (beta) learnable hain
```

---

## 📐 Config Values — Kyun Yeh Numbers?

```python
CONFIG = {
    "vocab_size": auto,    # Tokenizer se aayega
    "d_model": 128,        # ← 256 bhi ho sakta lekin slow
    "n_heads": 4,          # ← d_model ka factor hona chahiye
    "n_layers": 2,         # ← zyada = smarter but slow
    "d_ff": 512,           # ← d_model × 4 (standard rule)
    "max_seq_len": 64,     # ← kitne words ek baar
}
```

**CPU pe safe limits:**
- d_model ≤ 256
- n_layers ≤ 4
- max_seq_len ≤ 128

---

## 🔢 Parameters Count

```
Token Embedding:    vocab × 128        = ~6,272
Per Block:
  Attention:        4 × (128×128)      = 65,536
  FFN:              2 × (128×512)      = 131,072
  Layer Norm:       4 × 128            = 512
                                       ─────────
  Per block total:                     ~197,120

2 Blocks:                              ~394,240
Output Projection:  128 × vocab        = ~6,272

TOTAL:                                 ~406,784
                                       (~400K params)
```

**Claude ka estimate: ~100 Billion params**
Tumhara: 400K — 250,000x chota — lekin structure same! ✅

---

## 🔄 Inference vs Training

```
INFERENCE (Abhi ho raha hai):
Input → Forward pass → Output
Fast on CPU — milliseconds

TRAINING (Baad mein):
Input → Forward → Loss → Backward → Update weights
CPU pe slow — lekin hoga!
```

---

## 📁 Files

```
model/transformer.py  ← Poora architecture yahan hai
  - softmax()
  - gelu()  
  - layer_norm()
  - positional_encoding()
  - MultiHeadAttention class
  - FeedForward class
  - TransformerBlock class
  - MyAI class (main model)
```
