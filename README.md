# 🧠 Applied Deep Learning & Machine Learning Portfolio

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

## 👋 About Me
Hi, I'm Florent LE NY.
This repository showcases my journey in **Deep Learning**, featuring end-to-end implementations of modern neural architectures. 
I focus on building reproducible pipelines, from **Data Engineering** and **EDA** to **Model Deployment** and **Evaluation**.

## 🚀 Featured Projects

Here is a curated list of projects covering **NLP**, **Computer Vision**, **Audio Processing**, and **Generative AI**.

| Domain | Project Name | Architecture / Tech | Key Takeaway |
| :--- | :--- | :--- | :--- |
| 🤖 **Modern NLP** | [**Machine Translation**](./machine_translation_transformer.ipynb) | **Transformers**, Attention Mechanism | Implemented "Attention is All You Need" architecture for En-Fr translation. |
| 🖼️ **Vision** | [**Image Denoising**](./image_denoising.ipynb) | **CNN**, Autoencoder (U-Net style) | Removed Gaussian noise from STL10 images using convolutional reconstruction. |
| 🎨 **GenAI** | [**GAN Image Generation**](./image_generation_GAN.ipynb) | **DCGAN** (Deep Convolutional GAN) | Trained adversarial networks (Generator vs Discriminator) to synthesize MNIST digits. |
| 🎵 **Audio** | [**Audio Classification**](./speech_classification.ipynb) | **1D-CNN**, Signal Processing | End-to-end pipeline to classify Music vs. Speech signals from raw waveforms. |
| 📜 **Classic NLP** | [**Text Generation**](./text_generation_LSTM.ipynb) | **LSTM**, RNN | Character-level language modeling with temperature sampling for creative writing. |

---

## 🔬 Project Details

### 1. Neural Machine Translation (Transformers)
* **Goal:** Translate English sentences to French using the SOTA architecture.
* **Highlights:**
    * Manual implementation of **Multi-Head Attention** and Positional Encoding.
    * Handling variable-length sequences with Padding Masks.
    * **Why it matters:** Demonstrates deep understanding of the architecture behind LLMs (GPT, BERT).

### 2. Image Denoising (Autoencoder)
* **Goal:** Restore clean images from noisy inputs.
* **Highlights:**
    * Built a custom **Convolutional Autoencoder**.
    * Optimized reconstruction loss (MSE) to recover fine details.
    * **Why it matters:** Shows proficiency in feature extraction and reconstruction tasks in Computer Vision.

### 3. Generative Adversarial Network (DCGAN)
* **Goal:** Generate realistic handwritten digits from random noise.
* **Highlights:**
    * Stabilized the training of two competing networks (Generator & Discriminator).
    * Addressed common GAN challenges like **Mode Collapse**.
    * **Why it matters:** Proves ability to work with unstable, complex training loops in Generative AI.

### 4. Audio Classification
* **Goal:** Distinguish between Speech and Music.
* **Highlights:**
    * **Data Engineering:** Created a custom `Dataset` class for audio loading and sampling.
    * Used **1D Convolutions** to process temporal signal data directly.
    * **Why it matters:** Demonstrates versatility beyond images and text (Signal Processing).

### 5. Text Generation with LSTM (Classic NLP)
* **Goal:** Generate creative text character-by-character (Alice in Wonderland style).
* **Highlights:**
    * Implemented a **Recurrent Neural Network (LSTM)** to manage long-term dependencies in sequences.
    * Created a sampling function with a **Temperature parameter** to control creativity vs coherence.
    * **Why it matters:** Demonstrates strong foundational knowledge of Sequence Modeling, the precursor to Transformers.

---

## 🛠️ Tools & Libraries used
* **Core:** Python, PyTorch (torch, nn, optim).
* **Data Manipulation:** NumPy, Pandas.
* **Visualization:** Matplotlib, Seaborn.
* **Data Loading:** Torchvision, Torchaudio.
