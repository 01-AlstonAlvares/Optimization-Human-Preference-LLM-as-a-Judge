# 🚀 A5: Human Preference Optimization & LLM-as-a-Judge

This repository contains the implementation for Assignment 5: Optimization Human Preference & LLM-as-a-Judge for the AT82.05 Artificial Intelligence: Natural Language Understanding (NLU) course.

------------------------------------------------------------------------

## 👨‍🎓 Author Information

-   **Name:** Alston Alvares
-   **Student ID:** st126488
-   **Hardware:** NVIDIA RTX 3050 (4GB VRAM)
-   **Model:** Qwen2.5-1.5B

------------------------------------------------------------------------

# 📌 Project Overview

The goal of this assignment is to align a Large Language Model (LLM) to be more truthful and avoid hallucinations using Direct Preference Optimization (DPO). Additionally, it involves building an evaluation pipeline where a strong LLM acts as a judge to compare model performance.

------------------------------------------------------------------------

# 🧠 Technical Implementation

## ⚙️ Core Training Strategy

The base model was:

-   Quantized to 4-bit precision
-   Frozen
-   Enhanced with trainable LoRA adapters
-   Fine-tuned using DPO

------------------------------------------------------------------------

## 💾 Memory Optimization Techniques

### 🔹 1. QLoRA (4-bit Quantization)

-   Used `BitsAndBytesConfig`
-   `nf4` quantization scheme
-   Reduced memory footprint significantly
-   Enabled loading a 1.5B model into 4GB VRAM

### 🔹 2. LoRA Adapters

-   Attached trainable low-rank adapters
-   Avoided "purely quantized model cannot be trained" error
-   Base model remained frozen
-   Only adapter weights updated

### 🔹 3. Gradient Checkpointing

-   Reduced memory usage during backward pass
-   Recomputed activations instead of storing them

### 🔹 4. 8-bit Paged Optimizer

-   Used `paged_adamw_8bit`
-   Optimizer states paged to system RAM
-   Prevented memory spikes during training

------------------------------------------------------------------------

# 🛠 Environment Fixes & Debugging

## ✅ NumPy Compatibility Fix

-   NumPy 2.x caused binary incompatibility with PyTorch 2.4
-   Solution: numpy\<2

## ✅ PyTorch 2.4 Hotpatch

-   Encountered missing `set_submodule` method in `torch.nn.Module`
-   Manually injected missing method
-   Resolved `AttributeError`

------------------------------------------------------------------------

# 📊 Training & Evaluation Results

## 🔁 Training Configuration

-   **Epochs:** 1\
-   **Evaluation Samples:** 15\
-   **Dataset:** AlpacaEval (`helpful_base`)\
-   **Judge Model:** Gemini-1.5-Flash

------------------------------------------------------------------------

## 📋 Evaluation Methodology and Summary

![alt text](image-1.png)

  Metric           Value
  ---------------- ------------
  Model B Wins     0
  Ties             15
  Total Samples    15
  Final Win Rate   **50.00%**

------------------------------------------------------------------------

## 📐 Win Rate Formula

Win Rate = (Model B Wins + (0.5 × Ties)) / Total Samples × 100

Win Rate = (0 + (0.5 × 15)) / 15 × 100\
Win Rate = 50.00%

------------------------------------------------------------------------

# 🔄 Reproducibility Guide

### 1️⃣ Environment Setup

-   Create a virtual environment
-   Install:
    -   torch 2.4.0+cu121
    -   numpy\<2
    -   bitsandbytes
    -   peft
    -   trl
    -   transformers

### 2️⃣ Apply Hotpatch

Run the `set_submodule` fix before loading the model.

### 3️⃣ Sequential Evaluation Strategy

1.  Load Base Model
2.  Generate outputs
3.  Delete model
4.  Clear CUDA cache
5.  Load DPO Model
6.  Generate outputs

### 4️⃣ Link to Model (Hugging Face)
https://huggingface.co/Alston5432/LLM-as-a-Judge/tree/main
------------------------------------------------------------------------

# 🏁 Conclusion

Although the final win rate was **50.00%**, this project successfully
demonstrates:

-   Fine-tuning a 1B+ parameter model on 4GB hardware\
-   Stable DPO alignment under extreme memory constraints\
-   Practical use of QLoRA + LoRA + Gradient Checkpointing\
-   Real-world debugging and environment patching

------------------------------------------------------------------------

# 📌 Key Takeaway

>This assignment demonstrated the practical application of Direct Preference Optimization (DPO) to align the Qwen2.5-1.5B-Instruct model toward more truthful, non-hallucinatory responses. By utilizing a specialized "truthy" dataset, the model was fine-tuned to distinguish between factual and incorrect answers. A significant component of the project involved developing a scalable LLM-as-a-Judge pipeline, which leveraged the Gemini API to conduct impartial side-by-side evaluations against the industry-standard AlpacaEval benchmark. This automated evaluation process allowed for the calculation of a standardized win rate, providing a quantitative measure of how human preference alignment impacts model helpfulness and accuracy.
