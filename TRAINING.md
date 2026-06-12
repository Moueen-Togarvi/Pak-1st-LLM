# 🏋️ TRAINING.md — Model Ko Train Kaise Karna Hai

> Yahan se AI actually seekhna shuru karta hai

---

## 🧠 Training Kya Hota Hai?

```
Seedha explanation:

Model predict karta hai: "mera" ke baad "naam" aayega
Actual answer hai:       "naam"
Difference (Loss):       0.3

Model apni weights thodi adjust karta hai
Agli baar better predict karta hai

Yeh process lakho baar hoti hai
= Training
```

---

## 📐 Training Ka Math (Simple)

```
1. Forward Pass:
   Input → Model → Predicted next word

2. Loss Calculate:
   Cross Entropy Loss = -log(probability of correct word)
   
   Agar model ne socha "naam" ka chance 70% hai → Loss = 0.36
   Agar model ne socha "naam" ka chance 10% hai → Loss = 2.3
   (Low loss = acha)

3. Backward Pass:
   Loss se gradients nikalo (calculus)
   
4. Weight Update:
   weights = weights - learning_rate × gradients
   
5. Repeat 10,000+ times
```

---

## 📁 Training File

```python
# training/trainer.py

import numpy as np
import sys, os, json, time
sys.path.append('..')

from tokenizer.tokenizer import MyTokenizer
from model.transformer import MyAI, softmax


class Trainer:
    def __init__(self, model, tokenizer, learning_rate=0.001):
        self.model = model
        self.tok = tokenizer
        self.lr = learning_rate
        self.losses = []

    def cross_entropy_loss(self, logits, target_id):
        """
        Kitna galat tha model?
        target_id = sahi answer ka token ID
        """
        probs = softmax(logits)
        # Small value add karo (numerical stability)
        loss = -np.log(probs[target_id] + 1e-8)
        return loss, probs

    def train_on_pair(self, input_text, output_text):
        """
        Ek conversation pair pe train karo
        Input:  "tum kaun ho"
        Output: "main AI hoon"
        """
        # Encode karo
        input_ids = self.tok.encode(input_text)
        output_ids = self.tok.encode(output_text)

        # Combined sequence banao: [input] [output]
        full_sequence = input_ids + output_ids[1:]  # BOS skip

        total_loss = 0
        steps = 0

        # Har output token ke liye train karo
        for i in range(len(input_ids), len(full_sequence) - 1):
            context = full_sequence[:i+1]
            target = full_sequence[i+1]

            # Forward pass
            logits = self.model.forward(context)
            loss, probs = self.cross_entropy_loss(logits, target)
            total_loss += loss
            steps += 1

            # Simple gradient update (finite differences)
            # Note: Real training mein backprop use hoti hai
            # Yeh simplified version hai CPU ke liye
            self._simple_update(context, target, probs)

        return total_loss / max(steps, 1)

    def _simple_update(self, context, target_id, probs):
        """
        Simplified weight update
        (Production mein backpropagation use hoti hai)
        """
        # Output projection update (last layer)
        last_hidden = self.model.forward(context)

        # One-hot target
        target_one_hot = np.zeros(self.model.config["vocab_size"])
        target_one_hot[target_id] = 1.0

        # Gradient (simplified)
        grad = probs - target_one_hot

        # Update output projection
        # (Actual backprop bohot complex hai — yeh approximation hai)
        self.model.output_proj -= self.lr * np.outer(
            np.ones(self.model.config["d_model"]) * 0.01,
            grad
        )

    def train(self, data_pairs, epochs=10, save_every=5):
        """
        data_pairs = [("input1", "output1"), ("input2", "output2"), ...]
        epochs = kitni baar poora data dekhe
        """
        print(f"\n🏋️ Training Start!")
        print(f"   Data pairs: {len(data_pairs)}")
        print(f"   Epochs:     {epochs}")
        print(f"   Learning rate: {self.lr}")
        print("-" * 40)

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0

            # Data shuffle karo
            np.random.shuffle(data_pairs)

            for inp, out in data_pairs:
                loss = self.train_on_pair(inp, out)
                epoch_loss += loss

            avg_loss = epoch_loss / len(data_pairs)
            self.losses.append(avg_loss)

            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

            # Save checkpoint
            if epoch % save_every == 0:
                self.model.save(f"model/checkpoint_epoch{epoch}.npz")
                print(f"   💾 Checkpoint saved")

        # Final save
        self.model.save("model/weights.npz")
        print(f"\n✅ Training complete!")
        print(f"   Final loss: {self.losses[-1]:.4f}")
        print(f"   (Loss kam = better)")


# ---- RUN TRAINING ----
if __name__ == "__main__":
    print("Loading model and tokenizer...")

    tok = MyTokenizer()
    tok.load("tokenizer/vocab.json")

    with open("model/weights_config.json") as f:
        config = json.load(f)

    model = MyAI(config)
    model.load("model/weights.npz")

    # Training data — apna data yahan add karo
    TRAINING_DATA = [
        ("tum kaun ho", "main tumhara apna AI hoon"),
        ("hello", "hello kya haal hai"),
        ("tumhara naam", "abhi mera naam nahi tum rakh do"),
        ("kya haal hai", "theek hoon shukriya"),
        ("meri madad karo", "zaroor batao kia chahiye"),
        ("shukriya", "koi baat nahi khushi hui madad karke"),
        ("bye", "khuda hafiz phir milte hain"),
        ("tum kya kar sakte ho", "baat kar sakta hoon aur seekh sakta hoon"),
        ("acha hai", "shukriya tumhari baat acchi lagi"),
        ("samjha nahi", "maafi chahta hoon dobara poochho"),
    ]

    trainer = Trainer(model, tok, learning_rate=0.001)
    trainer.train(TRAINING_DATA, epochs=20, save_every=10)
```

---

## 🚀 Training Run Karo

```bash
# Pehle setup karo (pehli baar)
python main.py setup

# Phir training chalaao
cd training
python trainer.py

# Chat karo
cd ..
python main.py chat
```

---

## 📊 Loss Ko Samajhna

```
Loss = 3.0+  → Model bilkul random hai (shuru mein)
Loss = 2.0   → Thoda seekh raha hai
Loss = 1.0   → Achi progress
Loss = 0.5   → Accha hai
Loss = 0.1   → Bohot acha (overfitting ka risk)

Graph:
3.5 |*
3.0 | **
2.5 |   ***
2.0 |      ****
1.5 |          *****
1.0 |               ******
    └─────────────────────── Epochs
    
Yeh curve chahiye — seedha neeche
```

---

## ⚙️ Hyperparameters — Kya Change Kar Sakte Ho

```python
learning_rate = 0.001  # ↑ Fast learn, unstable
                       # ↓ Slow learn, stable
                       # Sweet spot: 0.0001 - 0.01

epochs = 20            # ↑ Zyada training
                       # Bohot zyada = overfitting

batch_size = 1         # CPU ke liye 1 hi theek hai
```

---

## ⚠️ Common Problems

```
Problem: Loss kam nahi ho rahi
Fix:     Learning rate kam karo (0.0001)

Problem: Loss bohot fast kam ho rahi, phir barhti hai
Fix:     Learning rate kam karo

Problem: Model sirf ek hi cheez bolta hai
Fix:     Data mein variety badho

Problem: Training bohot slow hai
Fix:     n_layers 2 pe rakho, d_model 128 pe
```

---

## 🔄 Training Cycle (Recommended)

```
Day 1:  10 data pairs → 20 epochs → Test
Day 3:  50 data pairs → 50 epochs → Test
Week 1: 200 data pairs → 100 epochs → Test
Week 2: 500+ data pairs → 200 epochs → Deploy
```

Har baar naya data add karo, dobara train karo.
Model improve hota jayega.
