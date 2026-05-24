# 🎙️ Agile Implementation Plan
## Project: Custom English TTS — Based on Qwen3-TTS Architecture

> **Dataset:** LibriSpeech train-clean-100 (100 giờ, duy nhất)
> **Approach:** Train LM Backbone từ đầu (0.6B params) · Tái dùng Qwen3-TTS-Tokenizer-12Hz · English-only
> **Methodology:** Agile Scrum · Sprint 2 tuần · **13 Sprints (~6.5 tháng)**
> **Reference:** [Qwen3-TTS Technical Report, arXiv:2601.15621](https://arxiv.org/abs/2601.15621)

---

## 📌 Project Overview

### Mục tiêu
1. Tái sử dụng **Qwen3-TTS-Tokenizer-12Hz** (speech codec — encode/decode audio)
2. Train **LM backbone 0.6B** theo pipeline 6-stage của Qwen3-TTS paper
3. Dataset duy nhất: **LibriSpeech train-clean-100** (~100h audio)
4. Output: Research demo chạy được local với 4-6 custom English voices

### Tại sao LibriSpeech 100h Clean?
- **Sạch ngay từ đầu**: thu trong studio, SNR rất cao, transcript chính xác
- **Đủ để prove concept**: 100h là ngưỡng tối thiểu để LM-based TTS hoạt động được
- **Rủi ro thấp**: không có data xấu lẫn vào → quá trình train ổn định hơn
- **Chi phí kiểm soát được**: preprocessing, tokenization, training đều nằm trong tầm 1-2 GPU

---

## 🏗️ Kiến trúc Hệ thống

```
┌─────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE                    │
│                                                         │
│  [Text Input]                                           │
│      │                                                  │
│      ▼                                                  │
│  [Text Tokenizer]  ←── Tiktoken cl100k_base             │
│      │                                                  │
│      ▼                                                  │
│  [LM Backbone]  ←── 0.6B Transformer (TRAINED FROM 0)  │
│  (Qwen3-style)      GQA · RoPE · SwiGLU · RMSNorm      │
│      │                                                  │
│      ▼                                                  │
│  [Audio Token IDs]  ←── 16 codebooks × 2048 vocab      │
│      │                                                  │
│      ▼                                                  │
│  [Qwen3-TTS-Tokenizer-12Hz]  ←── REUSED (open weights) │
│  (Lightweight causal ConvNet decoder)                   │
│      │                                                  │
│      ▼                                                  │
│  [Waveform Output .wav]                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🗺️ Full Training Pipeline (6-Stage theo Paper)

```
┌──────────────────────────────────────────────────────────────┐
│                  6-STAGE TRAINING PIPELINE                   │
│              (adapted từ Qwen3-TTS paper S.3.2)              │
│                                                              │
│  PRE-TRAINING ─────────────────────────────────────────────  │
│                                                              │
│  [S1 General Stage]                                          │
│   · Data: Tier-1 + Tier-2 (~90h sau khi filter)             │
│   · Goal: Establish text → audio token mapping              │
│   · max_seq_len = 4,096 tokens                              │
│         │                                                    │
│         ▼                                                    │
│  [S2 High-Quality CPT]                                       │
│   · Data: Tier-1 only (~40h, cleanest subset)               │
│   · Goal: Giảm hallucination từ S1, tăng stability          │
│   · Continual Pre-Training từ S1 checkpoint                 │
│         │                                                    │
│         ▼                                                    │
│  [S3 Long-Context Stage]                                     │
│   · Data: Ghép clip → long sequences (20-60s)               │
│   · Goal: Extend max_seq_len 4,096 → 8,192                  │
│   · Handle longer utterances không bị lỗi                   │
│                                                              │
│  POST-TRAINING ─────────────────────────────────────────────  │
│                                                              │
│  [DPO — Direct Preference Optimization]                      │
│   · Build (chosen, rejected) audio pairs từ S3 model        │
│   · Align outputs với human/automated preferences            │
│         │                                                    │
│         ▼                                                    │
│  [GSPO — Group Relative Policy Optimization]                 │
│   · Rule-based rewards: WER + UTMOS + SIM                   │
│   · RL tối ưu chất lượng + robustness                       │
│         │                                                    │
│         ▼                                                    │
│  [Speaker SFT — Lightweight Fine-tuning]                     │
│   · Fine-tune riêng cho từng voice persona                  │
│   · Freeze backbone, chỉ train speaker layers               │
│                                                              │
│              ▼  Final: English CustomVoice TTS               │
└──────────────────────────────────────────────────────────────┘
```

---

## 🗃️ Dataset — LibriSpeech train-clean-100 (Duy nhất)

### Thông tin dataset

| Thuộc tính | Giá trị |
|-----------|---------|
| **Tên** | LibriSpeech train-clean-100 |
| **Dung lượng gốc** | ~6.3 GB (FLAC) |
| **Thời lượng audio** | ~100 giờ |
| **Số speakers** | 251 speakers (125 nam, 126 nữ) |
| **Loại nội dung** | Audiobooks (đọc sách) |
| **Chất lượng** | Studio recording, SNR rất cao |
| **License** | CC BY 4.0 |
| **Download** | `wget https://openslr.org/resources/12/train-clean-100.tar.gz` |

### Tại sao chỉ dùng 100h đã đủ?
LibriSpeech train-clean-100 là bản đã được Pony Krell lọc kỹ nhất trong toàn bộ LibriSpeech:
- Chỉ chứa các recording có **alignment score cao nhất**
- Transcript được verify bởi forced-alignment tool
- Không có background noise, reverb thấp
- Đây là dataset chuẩn nhất để validate TTS baseline

---

## 🔬 Khái Niệm Nền Tảng

### Quality Tier là gì?

**Quality Tier = phân loại audio theo chất lượng** để dùng đúng data vào đúng training stage.

Dù LibriSpeech 100h đã "clean", vẫn có sự chênh lệch chất lượng giữa các clip:
- Một số speakers đọc rõ hơn, phát âm chuẩn hơn
- Một số clip có micro hiss nhẹ
- Một số clip rất ngắn (< 2s) hoặc sentence boundary bị cắt kỳ lạ

```
Tier 1 — "Premium" (~40h)
  Tiêu chí: SNR > 35dB · ASR-WER < 2% · duration 3-10s · no artifacts
  Mục đích: Dùng cho S2 High-Quality CPT
  → Model học từ data tốt nhất để loại bỏ bad habits từ S1

Tier 2 — "Standard" (~50h)
  Tiêu chí: SNR 25-35dB · WER 2-8% · duration 2-20s
  Mục đích: Dùng cho S1 General Stage (cùng với Tier 1)
  → Model học rộng, establish general mapping

Tier 3 — "Rejected" (~10h)
  Tiêu chí: WER > 8% · clip < 2s · audio artifacts rõ
  Mục đích: Loại khỏi training hoàn toàn
  → Tránh model học lỗi từ misaligned hoặc noisy data
```

**Quy tắc sử dụng:**

| Training Stage | Data dùng | Lý do |
|---------------|-----------|-------|
| S1 General | Tier 1 + Tier 2 (~90h) | Học rộng, establish mapping |
| S2 HQ-CPT | Tier 1 only (~40h) | Fix hallucinations, chỉ dùng best data |
| S3 Long-Context | Tier 1 + Tier 2 (ghép thành long sequences) | Học xử lý câu dài |
| DPO | Tier 1 only (làm base cho preference pairs) | Chỉ dùng sạch nhất để build pairs |

---

### Tokenized là gì?

**Tokenized = chuyển raw audio thành dãy số nguyên (token IDs) và lưu vào đĩa trước khi train.**

Đây là bước **quan trọng nhất** để giảm rủi ro tính toán trong training.

```
Trước khi tokenize (raw audio):
  clip_001.flac  →  [0.002, -0.001, 0.005, ...]
                    24,000 số float32 mỗi giây
                    = 100h × 3,600s × 24,000 = 8.6 tỷ số float
                    ≈ 34 GB RAM nếu load hết lên memory

Sau khi tokenize (discrete tokens):
  clip_001.npy   →  [[342, 1201, 88, ...],   ← codebook 0
                     [901, 445, 203, ...],    ← codebook 1
                     ...
                     [12, 788, 334, ...]]     ← codebook 15
                    12.5 frames × 16 integers mỗi giây
                    = 100h × 3,600s × 12.5 × 16 = 72 triệu integers
                    ≈ 280 MB (int16) — nhỏ hơn 120 lần!
```

**Tại sao phải tokenize offline (trước khi train)?**

| | Tokenize Online (trong lúc train) | Tokenize Offline (trước) |
|--|-----------------------------------|--------------------------|
| Tốc độ | Chậm: mỗi batch phải chạy Tokenizer-12Hz | Nhanh: chỉ đọc .npy integers |
| VRAM | Tốn thêm VRAM cho tokenizer model | Không tốn VRAM |
| Ổn định | Có thể crash nếu tokenizer gặp edge case | Đã xử lý hết edge cases rồi |
| Reproducible | Mỗi lần train có thể ra token khác nhau | Token cố định, training 100% reproducible |
| Storage | Không cần lưu | ~280 MB cho 100h |

**Kết luận:** Luôn tokenize offline, lưu `.npy`, trong training chỉ đọc file số nguyên.

---

## ⚙️ Pre-training Data Processing Pipeline

**Mục tiêu:** Biến 100h raw audio → dataset sẵn sàng train, giảm tối đa rủi ro tính toán.

### Pipeline 11 bước (thứ tự quan trọng)

```
INPUT: LibriSpeech train-clean-100 (FLAC files + transcript .txt)
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 1 — Download & Verify                             │
│  · wget từ openslr.org                                  │
│  · Verify MD5 checksum                                  │
│  · Unpack: tar -xzf train-clean-100.tar.gz              │
│  · Kết quả: ~28,539 clip FLAC + transcript files        │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 2 — Resample về 24kHz                             │
│  · Tokenizer-12Hz yêu cầu: 24,000 Hz                   │
│  · LibriSpeech gốc: 16,000 Hz → cần upsample           │
│  · Tool: ffmpeg -ar 24000 (nhanh hơn librosa)           │
│  · Lưu thành .wav (ffmpeg xử lý FLAC nhanh hơn python) │
│  ⚠️  RỦI RO: Upsample 16k→24k không tăng chất lượng    │
│      nhưng là bắt buộc cho tokenizer. Chấp nhận được.  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 3 — Duration Filter  ← GIẢM RỦI RO OOM           │
│  · Giữ: 2 giây ≤ duration ≤ 20 giây                    │
│  · Loại clip < 2s: quá ngắn, model không học được       │
│  · Loại clip > 20s: gây OOM khi batch, sequence dài     │
│  · LibriSpeech clean: ~95% clip nằm trong 2-20s         │
│  · Mất khoảng ~3-5h audio (acceptable)                 │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 4 — Amplitude Normalize                           │
│  · Peak normalize toàn bộ về -3 dBFS                   │
│  · Tránh clipping artifacts khi tokenizer encode        │
│  · Tránh model học volume differences không cần thiết   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 5 — Text Normalization                            │
│  · "Dr." → "Doctor"                                    │
│  · "100" → "one hundred"                               │
│  · "e.g." → "for example"                              │
│  · "Mr. Smith" → "Mister Smith"                        │
│  · Tool: nemo_text_processing (NVIDIA)                  │
│  ⚠️  Quan trọng: Tokenizer input là normalized text     │
│      Model sẽ học MAP normalized text → audio tokens   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 6 — ASR Alignment Check  ← GIẢM DỮ LIỆU NHIỄU   │
│  · Chạy Whisper-small (fast) trên mỗi clip             │
│  · Tính WER giữa Whisper output vs ground truth        │
│  · Gán WER score vào metadata                          │
│  · Loại clip WER > 10%: text không khớp audio          │
│  · Thời gian: ~2-3h cho 100h audio trên 1 GPU           │
│  · LibriSpeech clean: ~95% clip có WER < 5%            │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 7 — Quality Scoring & Tier Assignment             │
│  · Tính SNR (signal-to-noise ratio) mỗi clip           │
│  · Kết hợp SNR + WER + duration → quality score        │
│  · Gán Tier 1 / Tier 2 / Tier 3                        │
│  · Lưu quality metadata vào manifest                   │
│  KẾT QUẢ EXPECTED cho LibriSpeech clean:               │
│    Tier 1: ~40h   Tier 2: ~50h   Tier 3 (loại): ~10h  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 8 — PRE-TOKENIZE ALL AUDIO  ★ QUAN TRỌNG NHẤT ★  │
│                                                         │
│  · Chạy Qwen3-TTS-Tokenizer-12Hz.encode() trên ~90h    │
│    audio (Tier 1 + Tier 2)                             │
│  · Mỗi clip → file .npy: shape (T_frames, 16)          │
│    Ví dụ: clip 5 giây → shape (62, 16)                 │
│    (62 frames × 12.5 FPS = 5s, 16 codebooks)           │
│  · Lưu token files vào data/tokenized/                  │
│  · Thời gian: ~3-4h cho 90h audio trên 1 GPU           │
│  · Storage: ~250-300 MB tổng (rất nhỏ)                 │
│                                                         │
│  SAU BƯỚC NÀY: Training KHÔNG bao giờ đọc .wav nữa    │
│  Training chỉ đọc .npy (integers) — nhanh, ổn định    │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 9 — Build Long-Form Sequences (cho S3)            │
│  · Ghép 3-5 clip liền tiếp cùng speaker                │
│  · Token sequence dài 2,048 - 4,096 frames             │
│  · Tương đương 2.7 - 5.5 phút audio                    │
│  · Mục đích: S3 Long-Context cần dữ liệu dài           │
│  · Lưu riêng vào data/long_form/                        │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 10 — Pre-compute Sequence Lengths  ← GIẢM PADDING │
│  · Tính token_length của mỗi clip                      │
│  · Lưu vào manifest                                    │
│  · DataLoader sẽ sort/bucket theo length                │
│  · Kết quả: padding trong mỗi batch < 5%               │
│  · Không có điều này: padding có thể chiếm 30-50%      │
│    → Lãng phí VRAM, train chậm hơn 2x                  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 11 — Build Manifest JSONL & Split                 │
│  Mỗi dòng:                                             │
│  {                                                      │
│    "id": "1089-134686-0001",                           │
│    "speaker_id": "1089",                               │
│    "text": "He hoped there would be stew for dinner",  │
│    "tokens_path": "data/tokenized/1089-134686-0001.npy"│
│    "token_len": 87,                                    │
│    "duration_s": 6.97,                                 │
│    "tier": 1,                                          │
│    "snr_db": 38.2,                                     │
│    "asr_wer": 0.0,                                     │
│    "split": "train"                                    │
│  }                                                      │
│                                                         │
│  Split: 95% train / 4% val / 1% test                   │
│  Rule: cùng speaker không xuất hiện ở cả train & test  │
└─────────────────────────────────────────────────────────┘

OUTPUT: ~90h audio tokenized, manifest JSONL, train/val/test split
        → Sẵn sàng train
```

### Thống kê expected sau preprocessing

| Hạng mục | Gốc | Sau xử lý |
|---------|-----|----------|
| **Tổng thời lượng** | ~100h | ~90h (loại ~10h Tier 3) |
| **Số clips** | ~28,539 | ~26,000 |
| **Tier 1 (HQ)** | — | ~40h / ~11,500 clips |
| **Tier 2 (Standard)** | — | ~50h / ~14,500 clips |
| **Storage tokens** | 6.3 GB FLAC | ~280 MB .npy |
| **Storage manifest** | — | ~15 MB JSONL |
| **Thời gian preprocessing** | — | ~8-10h (1 GPU) |

---

## 💻 Infrastructure Requirements

### Minimum (Research Demo)
```
GPU:    1× A100 40GB  hoặc  2× RTX 4090 (24GB each)
RAM:    32GB system RAM  (đủ với 100h data)
Storage: 500 GB SSD  (raw audio 6GB + tokenized 280MB + checkpoints ~50GB per stage)
OS:     Ubuntu 22.04  ·  CUDA 12.1+  ·  Python 3.12
```

### Cloud Cost Estimate (scaled cho 100h data)

| Phase | Stage | GPU | Duration | Cost Est. |
|-------|-------|-----|----------|-----------|
| Data preprocessing | 11 steps | 1× A100 40GB | 1 ngày | ~$20 |
| Pre-training | S1 General (~90h data) | 2× A100 80GB | 5-7 ngày | ~$600 |
| Pre-training | S2 HQ-CPT (~40h data) | 2× A100 80GB | 2-3 ngày | ~$250 |
| Pre-training | S3 Long-Context | 2× A100 80GB | 2-3 ngày | ~$250 |
| Post-training | DPO | 1× A100 40GB | 1-2 ngày | ~$80 |
| Post-training | GSPO | 2× A100 80GB | 2-3 ngày | ~$300 |
| Post-training | Speaker SFT (×6 voices) | 1× A100 40GB | 2 ngày tổng | ~$80 |
| Evaluation | Benchmark | 1× A100 40GB | ongoing | ~$100 |

**Tổng estimate: ~$1,680 — $2,200** *(giảm 3-4× so với 500h plan)*

---

## 📐 LM Backbone Architecture (0.6B)

```
hidden_size:              1024
num_hidden_layers:        28
num_attention_heads:      16   (query heads)
num_key_value_heads:      8    (GQA: 2 queries per KV)
intermediate_size:        2816 (~2.75× hidden)
max_position_embeddings:  4,096 → 8,192 (S3 extend)
audio_vocab_size:         2,048 (per codebook)
num_codebooks:            16    (Qwen3-TTS-Tokenizer-12Hz)
rope_theta:               1,000,000.0
Total params:             ~0.5B
```

### Token Sequence Format (ChatML — theo paper)
```
<|im_start|>system
You are a TTS model. Generate speech tokens for the input.
<|im_end|>
<|im_start|>user
[SPEAKER: Ryan] He hoped there would be stew for dinner.
<|im_end|>
<|im_start|>assistant
[TTS_START] <a₁> <a₂> ... <aT> [TTS_END]
<|im_end|>
```

---

## 🏃 Sprint Details

---

### Sprint 0 — Bootstrap (1 tuần)

**Goal:** Môi trường sẵn sàng, tokenizer verified

| # | Task | Points |
|---|------|--------|
| S0-1 | Setup conda env: Python 3.12, PyTorch 2.x, CUDA 12.1, HuggingFace | 3 |
| S0-2 | Clone Qwen3-TTS repo, nghiên cứu ChatML format + tokenizer API | 5 |
| S0-3 | Test tokenizer round-trip: encode WAV → tokens → decode → WAV, so sánh bằng tai | 3 |
| S0-4 | Setup Git repo, Hydra config system, CI/CD cơ bản | 3 |
| S0-5 | Setup Weights & Biases: project, API key, test log | 2 |

**DoD:** `tokenizer.encode/decode` chạy được · W&B live · CI pass

---

### Sprint 1 — Data Pipeline I: Download → Tier Assignment (2 tuần)

**Goal:** Chạy Bước 1-7 của preprocessing pipeline, có manifest sơ bộ

| # | Task | Points |
|---|------|--------|
| S1-1 | Download LibriSpeech train-clean-100, verify MD5 (~6.3 GB) | 2 |
| S1-2 | Resample pipeline: FLAC → 24kHz WAV (ffmpeg batch script) | 3 |
| S1-3 | Duration filter: script lọc clip < 2s và > 20s | 3 |
| S1-4 | Amplitude normalization: peak normalize -3dBFS toàn bộ | 2 |
| S1-5 | Text normalization: chạy nemo_text_processing trên tất cả transcripts | 5 |
| S1-6 | ASR alignment check: Whisper-small, gán WER score, loại WER > 10% | 8 |
| S1-7 | Quality scoring: tính SNR, kết hợp với WER → gán Tier 1/2/3 | 5 |
| S1-8 | Manifest sơ bộ JSONL với metadata đầy đủ | 3 |

**DoD:** ~90h audio qualified · Tier labels assigned · Manifest JSONL với tier/wer/snr/duration

---

### Sprint 2 — Data Pipeline II: Tokenize → DataLoader (2 tuần)

**Goal:** Chạy Bước 8-11, DataLoader production-ready

| # | Task | Points |
|---|------|--------|
| S2-1 | **Pre-tokenize ~90h audio** (Bước 8): chạy Tokenizer-12Hz, lưu .npy | 8 |
| S2-2 | Verify tokenization: encode → decode 50 clips, nghe kiểm tra | 3 |
| S2-3 | Build long-form sequences (Bước 9): ghép clip → 2k-4k frame sequences | 5 |
| S2-4 | Pre-compute token lengths (Bước 10), sort/bucket index | 3 |
| S2-5 | Build final manifest JSONL + train/val/test split (Bước 11) | 3 |
| S2-6 | PyTorch Dataset + DataLoader: đọc .npy + text, dynamic batching, bucket by length | 8 |
| S2-7 | DataLoader smoke test: verify shapes, padding masks, no OOM trên target GPU | 5 |
| S2-8 | Dataset statistics report: duration dist, speaker dist, tier breakdown | 3 |

**DoD:** Tokenized data ready (~280MB) · DataLoader trả batch `(B, T, 16)` không lỗi · stats report

---

### Sprint 3 — Model Architecture (2 tuần)

**Goal:** LM backbone hoàn chỉnh, unit tests pass

| # | Task | Points |
|---|------|--------|
| S3-1 | Transformer backbone: RMSNorm, GQA (16q/8kv), RoPE (theta=1M), SwiGLU FFN | 13 |
| S3-2 | Multi-codebook LM heads: 16 parallel linear projections | 8 |
| S3-3 | Speaker conditioning: learned embedding table, injected vào position 1 | 5 |
| S3-4 | ChatML sequence builder: format input/output theo paper spec | 5 |
| S3-5 | Model config presets (tiny/small/0.5B) + YAML serialization | 3 |
| S3-6 | Unit tests: shapes, dtypes, KV cache consistency, loss > 0 | 5 |

**DoD:** Forward pass sạch · params ~0.5B · tests pass · loss computable

---

### Sprint 4 — Training Infrastructure (2 tuần)

**Goal:** Training loop đầy đủ, stage-aware, production-ready

| # | Task | Points |
|---|------|--------|
| S4-1 | Training loop: AdamW optimizer + cosine LR warmup + gradient clipping | 8 |
| S4-2 | DeepSpeed ZeRO-2 integration + multi-GPU launch script | 8 |
| S4-3 | bfloat16 mixed precision | 3 |
| S4-4 | Stage-aware checkpoint: save/resume theo từng stage (S1→S2→S3→DPO→GSPO→SFT) | 5 |
| S4-5 | Training config YAML riêng cho từng stage (LR, seq_len, data_tier khác nhau) | 5 |
| S4-6 | Logging: loss, grad norm, LR, tokens/sec, WER proxy mỗi 500 steps | 3 |
| S4-7 | Validation loop: eval mỗi epoch, log val loss | 3 |
| S4-8 | Smoke test: 1,000 steps với tiny config, verify loss giảm, không crash | 3 |

**Training Configs (key differences)**
```yaml
S1_general:
  data_tiers: [1, 2]       # ~90h
  learning_rate: 1e-4
  max_seq_len: 4096
  warmup_steps: 2000

S2_hq_cpt:
  data_tiers: [1]          # ~40h Tier-1 only
  learning_rate: 2e-5      # thấp hơn (fine-tune từ S1)
  max_seq_len: 4096
  warmup_steps: 500

S3_long_context:
  data_tiers: [1, 2]       # + long-form sequences
  max_seq_len: 8192        # ← extend ở đây
  learning_rate: 1e-5
```

**DoD:** Distributed training clean · stage configs hoạt động · smoke test pass

---

### Sprint 5 — Pre-training S1: General Stage (2 tuần)

**Goal (Paper S1):** Establish text → audio mapping trên 90h English data

| # | Task | Points |
|---|------|--------|
| S5-1 | Launch S1 training: Tier-1+2 data, ~50K-80K steps | 13 |
| S5-2 | Monitor: loss curve, grad norm, không diverge | 5 |
| S5-3 | First audio samples: generate 20 test sentences sau mỗi 5K steps | 5 |
| S5-4 | WER eval tại checkpoint S1: Whisper-large transcribe output | 5 |
| S5-5 | Identify hallucination patterns: repetition, word skipping, babbling | 5 |
| S5-6 | Save S1 final checkpoint → input cho S2 | 2 |

**Targets S1 checkpoint (realistic với 100h data)**
| Metric | Target |
|--------|--------|
| WER | < 20% |
| Audio intelligible | Có (nghe hiểu được dù chưa tự nhiên) |
| Training loss | Converged, không diverge |

**DoD:** S1 checkpoint saved · audio nghe hiểu được · loss converged

---

### Sprint 6 — Pre-training S2: High-Quality CPT (2 tuần)

**Goal (Paper S2):** CPT trên Tier-1 data → giảm hallucination, tăng stability

| # | Task | Points |
|---|------|--------|
| S6-1 | Resume từ S1 checkpoint, switch sang Tier-1 only data (~40h) | 3 |
| S6-2 | Train S2 CPT: LR=2e-5, ~20K-30K steps | 13 |
| S6-3 | Hallucination comparison: đếm repetitions/skips S1 vs S2 trên 50 test sentences | 8 |
| S6-4 | WER/UTMOS evaluation: quantify improvement vs S1 checkpoint | 5 |
| S6-5 | Stress test: generate 10 paragraphs (200+ words), check stability | 5 |
| S6-6 | Save S2 final checkpoint → input cho S3 | 2 |

**Targets S2 checkpoint**
| Metric | S1 Baseline | S2 Target |
|--------|-------------|-----------|
| WER | < 20% | < 10% |
| UTMOS | > 2.8 | > 3.3 |
| Repetition rate | ~20% của outputs | < 5% |

**DoD:** S2 checkpoint saved · hallucinations giảm rõ rệt (đo được) · WER < 10%

---

### Sprint 7 — Pre-training S3: Long-Context (2 tuần)

**Goal (Paper S3):** Extend seq_len 4096→8192, stable long-form generation

| # | Task | Points |
|---|------|--------|
| S7-1 | Extend RoPE: cập nhật `max_position_embeddings` lên 8192 trong config | 3 |
| S7-2 | Resume từ S2 checkpoint, switch sang long-form data mix | 5 |
| S7-3 | Train S3: LR=1e-5, ~20K steps với extended context | 13 |
| S7-4 | Long-form eval: generate 30-60s utterances, measure WER + prosody | 8 |
| S7-5 | S3 final checkpoint = **Base Model** (input cho post-training) | 2 |

**Targets S3 — Base Model**
| Metric | S2 | S3 Target |
|--------|-----|-----------|
| WER (ngắn) | < 10% | < 7% |
| WER (dài > 20s) | Chưa test | < 12% |
| Max ổn định | ~10s | 30-60s |
| UTMOS | > 3.3 | > 3.5 |

**DoD:** S3 Base Model saved · long-form generation không crash · WER targets met

---

### Sprint 8 — Post-training: DPO (2 tuần)

**Goal (Paper Post Stage 1):** Align outputs với preferences qua preference pairs

**DPO hoạt động như thế nào:**
```
1. Lấy 500-1000 text prompts từ val set
2. Generate 4 audio samples mỗi prompt (khác seeds/temperatures)
3. Auto-score bằng UTMOS + WER: chọn tốt nhất = "chosen", xấu nhất = "rejected"
4. Human spot-check 10% pairs (50-100 pairs)
5. Train DPO: model học prefer "chosen" over "rejected"
```

| # | Task | Points |
|---|------|--------|
| S8-1 | Generate preference dataset: 500+ text prompts × 4 samples = 2000 audios | 8 |
| S8-2 | Auto-score: UTMOS + WER → chọn (chosen, rejected) pairs | 8 |
| S8-3 | Human spot-check: verify 50-100 pairs manually | 5 |
| S8-4 | Implement DPO loss (có thể dùng TRL library) | 5 |
| S8-5 | DPO training: LR=1e-6, 1-2 epochs | 8 |
| S8-6 | Eval: so sánh DPO vs Base Model trên 100 test sentences | 5 |

**Targets after DPO**
| Metric | Base Model | After DPO |
|--------|------------|-----------|
| UTMOS | > 3.5 | > 3.65 |
| WER | < 7% | < 6% |
| "Prefer DPO" (human) | — | > 60% |

**DoD:** DPO dataset built (~500 pairs) · DPO training converged · measurable improvement

---

### Sprint 9 — Post-training: GSPO (2 tuần)

**Goal (Paper Post Stage 2):** Rule-based RL tối ưu quality + robustness

**GSPO hoạt động như thế nào:**
```
Với mỗi input text:
  1. Generate G=8 audio samples từ current policy
  2. Score mỗi sample: r = 0.4×(1-WER) + 0.3×UTMOS + 0.2×SIM + 0.1×length_OK
  3. Normalize scores: r̂ = (r - mean) / std
  4. Policy gradient: tăng prob của samples có r̂ cao, giảm prob của r̂ thấp
  5. Repeat cho mỗi batch
```

| # | Task | Points |
|---|------|--------|
| S9-1 | Implement reward function: WER (Whisper) + UTMOS + SIM + length | 13 |
| S9-2 | Implement GSPO training loop: generate group, compute rewards, update | 13 |
| S9-3 | GSPO training: resume từ DPO checkpoint | 8 |
| S9-4 | Monitor: reward curves, output diversity (chống reward collapse) | 5 |
| S9-5 | Edge case test: số, tên riêng, câu phức tạp | 5 |

**Targets after GSPO**
| Metric | After DPO | After GSPO |
|--------|-----------|------------|
| WER | < 6% | < 4% |
| UTMOS | > 3.65 | > 3.8 |
| Edge case robustness | Moderate | High |

**DoD:** Reward curves converged · benchmark improvement · model ổn định trên edge cases

---

### Sprint 10 — Post-training: Speaker SFT & CustomVoice (2 tuần)

**Goal (Paper Post Stage 3):** Fine-tune 4-6 giọng English riêng biệt

| # | Task | Points |
|---|------|--------|
| S10-1 | Define 6 voice personas (từ LibriSpeech speakers có nhiều data nhất) | 3 |
| S10-2 | Curate per-speaker data: chọn 2-5h audio Tier-1 cho mỗi voice | 8 |
| S10-3 | Freeze backbone (layers 0-23), chỉ train speaker_embed + top 4 layers | 5 |
| S10-4 | Speaker SFT mỗi voice: LR=5e-6, 3-5 epochs | 13 |
| S10-5 | Per-voice eval: SIM score, naturalness, distinctiveness | 5 |
| S10-6 | Cross-voice test: cùng câu, 6 giọng khác nhau | 3 |

**6 Voice Personas (từ LibriSpeech speakers)**
| Voice ID | Gender | LibriSpeech Speaker | Estimated Hours |
|----------|--------|--------------------|----|
| Ryan | Male, adult | Speaker 1284 | ~5h |
| Aiden | Male, young | Speaker 3575 | ~4h |
| Emma | Female, adult | Speaker 2961 | ~5h |
| Sophia | Female, clear | Speaker 1221 | ~4h |
| Oliver | Male, deep | Speaker 4992 | ~3h |
| Isabella | Female, bright | Speaker 5142 | ~3h |

**Targets after Speaker SFT**
| Metric | After GSPO | After SFT |
|--------|------------|-----------|
| SIM per speaker | > 0.72 | > 0.82 |
| Voices distinct | Moderate | Clear (human test) |
| UTMOS | > 3.8 | > 3.9 |

**DoD:** 6 distinct voices working · SIM > 0.82 · CustomVoice API functional

---

### Sprint 11 — Evaluation & Benchmarks (2 tuần)

**Goal:** Đo đạc đầy đủ, ablation study, so sánh với baseline

| # | Task | Points |
|---|------|--------|
| S11-1 | WER benchmark: LibriSpeech test-clean (Whisper-large) | 5 |
| S11-2 | UTMOS + SIM evaluation: 100 test sentences × 6 voices | 5 |
| S11-3 | Long-form benchmark: 30-60s stability test | 5 |
| S11-4 | **Ablation**: S1 → S1+S2 → +S3 → +DPO → +GSPO → +SFT | 8 |
| S11-5 | Human evaluation: MOS score, 20 samples × 5 raters | 8 |
| S11-6 | Inference speed: RTF, first-packet latency measurement | 5 |

**Final Targets (realistic với 100h data)**
| Metric | Target | Note |
|--------|--------|------|
| WER (test-clean) | < 5% | Whisper-large baseline: ~3% on real speech |
| UTMOS | > 3.8 | Qwen3-TTS-0.6B: ~4.16 (trained 5M hours) |
| SIM | > 0.80 | Per speaker |
| First-packet latency | < 300ms | Single GPU inference |

**DoD:** Full benchmark report · ablation table hoàn chỉnh · human MOS collected

---

### Sprint 12 — Demo & Release (2 tuần)

**Goal:** Gradio demo, package, documentation

| # | Task | Points |
|---|------|--------|
| S12-1 | Gradio web UI: text input → voice selector → audio playback | 8 |
| S12-2 | Python package: `pip install custom-en-tts` | 8 |
| S12-3 | CLI: `python synthesize.py --text "..." --speaker Ryan` | 3 |
| S12-4 | Streaming demo: show chunk-by-chunk generation | 5 |
| S12-5 | HuggingFace Model Card: training pipeline, benchmarks, limitations | 3 |
| S12-6 | Reproduce guide: từ download data đến chạy demo | 5 |
| S12-7 | Docker container | 3 |

**DoD:** Gradio demo live · `pip install` hoạt động · docs đầy đủ

---

## 📊 Project Timeline

```
Month     1           2           3           4           5           6
Week   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26

S0     [Boot]
S1        [─ Data Pipeline I: Download→Tier ─]
S2               [─ Data Pipeline II: Tokenize→DataLoader ─]
S3                     [─ Architecture ─]
S4                           [─ Training Infra ─]
S5                                 [─ Pretrain S1: General ─]
S6                                       [─ Pretrain S2: HQ-CPT ─]
S7                                             [─ Pretrain S3: Long-Ctx ─]
S8                                                   [─ DPO ─]
S9                                                         [─ GSPO ─]
S10                                                              [─ Speaker SFT ─]
S11                                                                    [─ Eval ─]
S12                                                                          [─ Demo ─]
```

**Milestones:**
- **M1 (Week 8):** Data pipeline hoàn chỉnh, tokenized data ready
- **M2 (Week 12):** Architecture + Infra complete, sẵn sàng train
- **M3 (Week 16):** S1 Base — first audio samples
- **M4 (Week 20):** S3 Base Model — pre-training complete
- **M5 (Week 24):** Post-training complete — 6 custom voices
- **M6 (Week 26):** Demo released

---

## 📁 Project Structure

```
custom-en-tts/
├── configs/
│   ├── model/
│   │   └── backbone_0.6b.yaml
│   └── training/
│       ├── s1_general.yaml
│       ├── s2_hq_cpt.yaml
│       ├── s3_long_context.yaml
│       ├── dpo.yaml
│       ├── gspo.yaml
│       └── speaker_sft.yaml
│
├── data/
│   ├── raw/                      # LibriSpeech FLAC (6.3 GB)
│   ├── processed/                # Resampled 24kHz WAV
│   ├── quality_tiers/
│   │   ├── tier1/                # ~40h Tier-1 (SNR>35, WER<2%)
│   │   └── tier2/                # ~50h Tier-2
│   ├── tokenized/                # .npy token files (~280MB tổng)
│   ├── long_form/                # Ghép clip cho S3
│   ├── dpo_pairs/                # (chosen, rejected) JSONL
│   ├── speaker_sft/              # Per-speaker data
│   │   ├── ryan/
│   │   ├── aiden/
│   │   └── ...
│   └── manifests/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
│
├── src/
│   ├── model/
│   │   ├── config.py
│   │   ├── backbone.py           # RMSNorm, GQA, RoPE, SwiGLU, TransformerBackbone
│   │   └── tts_model.py          # Speaker embed + LM heads + full model
│   ├── data/
│   │   ├── dataset.py
│   │   ├── collator.py           # Bucketed batching
│   │   └── preprocessing/
│   │       ├── audio_filter.py   # Bước 2-4
│   │       ├── asr_check.py      # Bước 6: Whisper alignment check
│   │       ├── quality_scorer.py # Bước 7: SNR + WER → Tier
│   │       ├── tokenize_audio.py # Bước 8: offline tokenization
│   │       ├── build_longform.py # Bước 9: ghép clip
│   │       └── text_normalize.py # Bước 5
│   ├── training/
│   │   ├── pretrain/
│   │   │   └── trainer.py
│   │   ├── dpo/
│   │   │   ├── dpo_trainer.py
│   │   │   └── pair_builder.py
│   │   ├── gspo/
│   │   │   ├── gspo_trainer.py
│   │   │   └── reward_fn.py      # WER + UTMOS + SIM composite
│   │   └── sft/
│   │       └── speaker_sft.py
│   ├── inference/
│   │   ├── generator.py
│   │   └── streaming.py
│   └── evaluation/
│       ├── wer.py
│       ├── utmos.py
│       ├── speaker_sim.py
│       └── benchmark.py
│
├── scripts/
│   ├── 01_download_data.sh
│   ├── 02_resample.sh
│   ├── 03_filter_and_score.py
│   ├── 04_tokenize_offline.py    ★ chạy 1 lần trước training
│   ├── 05_build_manifest.py
│   ├── train_s1.py
│   ├── train_s2.py
│   ├── train_s3.py
│   ├── train_dpo.py
│   ├── train_gspo.py
│   ├── train_speaker_sft.py
│   ├── evaluate.py
│   └── synthesize.py
│
├── demo/
│   └── gradio_app.py
├── tests/
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_preprocessing.py
│   ├── test_dpo.py
│   └── test_gspo_reward.py
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md
```

---

## ⚠️ Risks & Mitigations

| Risk | Xác suất | Impact | Mitigation |
|------|----------|--------|------------|
| S1 loss không converge với 100h | Medium | High | Validate tại 5K steps; nếu không giảm → giảm LR hoặc check data pipeline |
| S2 CPT gây catastrophic forgetting | Medium | High | Mix S1 data 20% vào S2 batch (không dùng thuần Tier-1) |
| S3 OOM khi extend seq_len 8192 | High | Medium | Gradient checkpointing + giảm batch size × 2 |
| DPO: auto-scoring không tương quan với human preference | Medium | Medium | Human spot-check 10%; nếu noisy → dùng UTMOS+WER thay vì pair-wise |
| GSPO: reward collapse (model lặp lại 1 pattern) | Medium | High | Entropy bonus trong reward; monitor output diversity mỗi 500 steps |
| Speaker SFT: overfitting với ít data | Medium | Medium | LoRA thay full fine-tune; early stopping trên val SIM |
| LibriSpeech 100h chưa đủ để WER < 5% | High | Medium | Chấp nhận WER < 8% là success; không so sánh trực tiếp với Qwen3-TTS |

---

## 🎯 Success Metrics (Adjusted cho 100h data)

### Gate 1 — Pre-training Complete (End of Sprint 7)
- [ ] WER < 7% (short utterances)
- [ ] UTMOS > 3.5
- [ ] Stable generation up to 30 seconds
- [ ] Loss curves converged ở cả 3 stages

### Gate 2 — Post-training Complete (End of Sprint 10)
- [ ] WER < 4%
- [ ] UTMOS > 3.8
- [ ] SIM > 0.80 per speaker
- [ ] 6 distinct English voices working

### Gate 3 — Demo Ready (End of Sprint 12)
- [ ] Gradio demo live
- [ ] Ablation study: mỗi stage đóng góp improvement measurable
- [ ] Reproduce guide: người khác có thể chạy lại từ đầu

---

## 📚 References

- [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621)
- [LibriSpeech: train-clean-100](https://www.openslr.org/12) — Dataset duy nhất
- [DPO Paper — Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)
- [GRPO/GSPO — DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [TRL Library — DPO/GRPO reference](https://github.com/huggingface/trl)
- [UTMOS](https://github.com/sarulab-speech/UTMOS22) — MOS proxy scoring
- [nemo_text_processing](https://github.com/NVIDIA/NeMo-text-processing) — Text normalization
- [WeSpeaker](https://github.com/wenet-e2e/wespeaker) — Speaker verification / SIM scoring
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) — ZeRO-2 distributed training

---

*Document version: 3.0 · Updated: 2026-05-24*
*Changes: Dataset → LibriSpeech 100h Clean only · Data processing pipeline 11 bước · Quality Tier + Tokenized explained · Cost revised ~$1,700-2,200 · Success metrics adjusted*
