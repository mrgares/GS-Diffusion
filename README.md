
# GS-Diffusion

Diffusion model to refine output from a Gaussian Splatting-based model for autonomous driving use case.

This project trains a conditional diffusion model to enhance novel-view renderings (e.g., from DrivingForward or similar 3DGS models) by learning to map artifact-prone renderings to their corresponding high-quality ground-truth camera views.

---

## 🚀 Features

- Conditional image-to-image refinement using a shallow diffusion U-Net
- Designed for Gaussian Splatting pipelines in autonomous driving
- Docker-ready, GPU-accelerated environment with PyTorch and CUDA
- Based on Hugging Face's `diffusers` library

---

## 📦 Requirements

All dependencies are handled via Docker. You just need:

- Docker
- NVIDIA Container Toolkit (for GPU access)
- A dataset structured as:

```

data/train/
├── inputs/    # Rendered views from 3DGS (e.g., DrivingForward)
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── targets/   # Ground-truth camera views
│   ├── 0001.png
│   ├── 0002.png
│   └── ...

````

---

## ⚙️ Quick Start

### 1. Clone and build

```bash
git clone https://github.com/your-username/gs-diffusion.git
cd gs-diffusion
docker-compose build
````

### 2. Launch the container

```bash
docker-compose up -d
```

### 3. Attach to the container

```bash
docker exec -it diffusion_env bash
```

---



## 📁 Project Structure

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── train.py
├── data/
│   └── train/
│       ├── inputs/
│       └── targets/
└── README.md
```

## 📬 License

This project is licensed under the [MIT License](./LICENSE). See the LICENSE file for details.

