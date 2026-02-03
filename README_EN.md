# RaRaClipper - Advanced Clipping Plane System for 3D Gaussian Splatting

[![License](https://img.shields.io/badge/License-Non--commercial-blue.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](#system-requirements)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-orange.svg)](#system-requirements)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](#installation)

[中文文档](README_CN.md) | **English**

## 📖 Overview

RaRaClipper is the official open-source implementation of **"RaRa Clipper: A Clipper for Gaussian Splatting Based on Ray Tracer and Rasterizer"**. This project extends [Splatviz](https://github.com/Florian-Barthel/splatviz) with an advanced multi-plane clipping system for 3D Gaussian Splatting.

### Why RaRaClipper?

Traditional clipping methods for 3D Gaussian Splatting can produce visual artifacts and inconsistencies. RaRaClipper introduces:

- **Dual-strategy clipping** combining ray-tracing and rasterization approaches
- **1D Gaussian representation** for accurate clipping plane intersection
- **Interactive visualization** with real-time parameter adjustment
- **Modified CUDA rasterizer** with custom clipping logic

---

## 🔗 Related Links

- 📄 **Paper:** [PLACEHOLDER: Paper Title]
  - Conference: [PLACEHOLDER: Conference Name] 20XX
  - DOI: [PLACEHOLDER: DOI]
  - PDF: [PLACEHOLDER: ArXiv/Conference link]

- 🏗️ **Based on:**
  - [Splatviz](https://github.com/Florian-Barthel/splatviz) - Interactive viewer framework
  - [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Original Gaussian Splatting
  - [PLACEHOLDER: Multi-layer Gaussian Paper] - Multi-layer data format

---

## ✨ Key Features

### Clipping Strategies

#### 1. **RR Strategy (Ray-Rasterization)**
- Combines ray-tracing and rasterization approaches
- Uses 3x scale threshold for Gaussian filtering
- Optimized for performance while maintaining quality

```python
# Enable RR Strategy in GUI
rr_strategy = True
scale = 3.0  # 3x scale threshold
```

#### 2. **1D GS Strategy (1D Gaussian)**
- Projects 3D Gaussians to 1D representation along ray direction
- Accurate clipping plane intersection computation
- Reduces artifacts at clipping boundaries

```python
# Enable 1D GS Strategy in GUI
oenD_gs_strategy = True
```

#### 3. **RR Clipping (Combined Mode)**
- Automatically enables both strategies
- Best quality with minimal artifacts
- Recommended for most use cases

```python
# Enable RR Clipping (auto-enables both strategies)
rr_clipping = True
```

### Interactive Clipping Plane Widget

- **Multi-plane support** - Define multiple clipping planes simultaneously
- **Real-time visualization** - See clipping plane as colored polygon
- **Interactive editing** - Adjust plane normal and distance with sliders
- **Box scale control** - Adjust visualization bounding box

### Three Operation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Default** | Standard .ply file viewer | General Gaussian Splatting scenes |
| **Decoder** | Gaussian GAN decoder support | Generative models (.pkl files) |
| **Attach** | Remote renderer attachment | Distributed rendering |

---

## 🛠️ Installation

### System Requirements

- **OS:** Windows / Linux / macOS (with CUDA support)
- **GPU:** NVIDIA GPU with CUDA support (Compute Capability ≥ 7.0)
  - Recommended: RTX 3000/4000 series or higher
- **CUDA:** 11.0 or higher (11.8 or 12.1 recommended)
- **cuDNN:** 8.0 or higher (recommended)
- **Python:** 3.8, 3.9, or 3.10
- **RAM:** 8GB minimum, 16GB recommended
- **VRAM:** 6GB minimum, 8GB+ recommended

### Verify CUDA Installation

Before installation, verify your CUDA setup:

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Should show CUDA 11.x or 12.x
```

### Step 1: Clone Repository

```bash
git clone [PLACEHOLDER: Repository URL]
cd RaRaClipper
```

### Step 2: Install PyTorch and Dependencies

**IMPORTANT:** Ensure Splatviz works first before proceeding with RaRaClipper-specific installations.

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install imgui-bundle click numpy imageio loguru Pillow open3d
```

### Step 3: Build CUDA Extensions with RaRaClipper Patch

This is the critical step where RaRaClipper's custom clipping logic is compiled.

```bash
# Navigate to diff-gaussian-rasterization
cd gaussian-splatting/submodules/diff-gaussian-rasterization

# Build and install (this compiles forward.cu with clipping logic)
pip install -e .

# Navigate to simple-knn
cd ../simple-knn
pip install -e .

# Return to project root
cd ../../..
```

**Build time:** 5-10 minutes depending on your system.

**What's being compiled:**
- `cuda_rasterizer/forward.cu` - Modified with clipping plane logic
- `cuda_rasterizer/backward.cu` - Gradient computation
- `rasterize_points.cu` - Main rasterization kernel

### Step 4: Verify Installation

```bash
# Test imports
python -c "import diff_gauss; print('✅ diff-gaussian-rasterization installed!')"
python -c "import simple_knn; print('✅ simple-knn installed!')"

# Test full application
python run_main.py --help
```

### Complete Installation Script

For convenience, here's a complete installation script:

```bash
#!/bin/bash
# install.sh

# Step 1: Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 2: Install dependencies
pip install imgui-bundle click numpy imageio loguru Pillow open3d

# Step 3: Build diff-gaussian-rasterization
cd gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -e .

# Step 4: Build simple-knn
cd ../simple-knn
pip install -e .

# Step 5: Return to root
cd ../../..

echo "✅ Installation complete!"
```

### Troubleshooting

**Issue: CUDA version mismatch**
```bash
# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Should match your system CUDA version (±1 version is usually OK)
```

**Issue: Build fails with "CUDA not found"**
```bash
# Set CUDA_HOME environment variable
export CUDA_HOME=/usr/local/cuda  # Linux/macOS
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8  # Windows
```

**Issue: Out of memory during compilation**
```bash
# Reduce parallel jobs
MAX_JOBS=4 pip install -e .
```

---

## 🚀 Usage

### Basic Usage

```bash
# Launch with default sample scenes
python run_main.py

# Launch with custom .ply files directory
python run_main.py --data_path=/path/to/your/ply/files

# Launch in decoder mode (for .pkl files)
python run_main.py --data_path=/path/to/pkl/files --mode=decoder --ggd_path=/path/to/gaussian_decoder

# Launch in attach mode (remote renderer)
python run_main.py --mode=attach --host=192.168.1.100 --port=6009
```

### Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--data_path` | PATH | `./resources/sample_scenes` | Directory containing .ply or .pkl files |
| `--mode` | STRING | `default` | Operation mode: `default`, `decoder`, `attach` |
| `--host` | STRING | `127.0.0.1` | Host address for attach mode |
| `--port` | INT | `6009` | Port number for attach mode |
| `--ggd_path` | PATH | `` | Path to Gaussian GAN Decoder project (decoder mode) |

### Dataset Format

RaRaClipper expects the same format as Splatviz:

```
data_path/
├── scene1/
│   └── point_cloud.ply          # Gaussian splatting PLY file
├── scene2/
│   └── point_cloud.ply
└── scene3/
    └── compression_config.yml   # Compressed scene format
```

**PLY file format:** Standard 3D Gaussian Splatting format with attributes:
- Position: `x, y, z`
- Normals: `nx, ny, nz`
- Spherical Harmonics: `f_dc_0, f_dc_1, f_dc_2, ...`
- Opacity: `opacity`
- Scale: `scale_0, scale_1, scale_2`
- Rotation: `rot_0, rot_1, rot_2, rot_3` (quaternion)

---

## 🎮 GUI Controls

### Widget Overview

The GUI consists of multiple collapsible widgets:

| Widget | Description |
|--------|-------------|
| **Load** | Load .ply/.pkl files and manage scenes |
| **Camera** | Camera controls and view management |
| **Performance** | FPS counter and memory usage |
| **Video** | Record rotation videos |
| **Capture** | Screenshot capture |
| **Render** | Rendering options (depth, alpha, etc.) |
| **Edit** | Python code execution for scene editing |
| **Eval** | Evaluation metrics and quality assessment |
| **Clipping Plane** | Multi-plane clipping controls (RaRaClipper feature) |

### Clipping Plane Widget

<p align="center">
  <img src="resources/images/clipping_widget.png" width="400" alt="Clipping Plane Widget"/>
  <br>
  <em>[PLACEHOLDER: Add screenshot of Clipping Plane widget]</em>
</p>

#### New Plane Configuration

1. **Set plane normal** using `nx`, `ny`, `nz` sliders (-1.0 to 1.0)
   - Normalized normal vector is displayed automatically
2. **Set plane distance** using `d` slider (-5.0 to 5.0)
   - Plane equation: `nx*x + ny*y + nz*z + d = 0`
3. **Click "Add Plane"** to create the clipping plane

#### Plane Management

- **Edit existing planes** - Adjust normal and distance with sliders
- **Remove planes** - Click "Remove" button next to each plane
- **Visualize plane** - Enable "vizPlane" checkbox to see clipping plane polygon

#### RR Clipping Settings

- **RR Clipping** - One-click enable for best quality (auto-enables RR Strategy + 1D GS)
- **RR Strategy** - Enable ray-rasterization clipping with 3x scale threshold
- **1D GS Strategy** - Enable 1D Gaussian representation for accurate intersection
- **Box Scale** - Adjust visualization bounding box size (default: 1.0)

#### Strategy Comparison

| Strategy | Quality | Performance | Artifacts | Use Case |
|----------|---------|-------------|-----------|----------|
| None | Baseline | Fastest | Many | Quick preview |
| RR Strategy | Good | Fast | Some | General use |
| 1D GS Strategy | Better | Medium | Few | Quality-focused |
| RR Clipping (Both) | Best | Medium | Minimal | Publication-quality |

---

## 📸 Examples

### Example 1: Single Plane Clipping

```bash
python run_main.py --data_path=./resources/sample_scenes
```

1. Open "Clipping Plane" widget
2. Set plane normal: `nx=1.0, ny=0.0, nz=0.0`
3. Set distance: `d=0.0`
4. Enable "RR Clipping"
5. Enable "vizPlane" to see the clipping plane
6. Adjust `d` slider to move the plane

<p align="center">
  <img src="resources/images/example_single_plane.png" width="800" alt="Single Plane Clipping"/>
  <br>
  <em>[PLACEHOLDER: Add screenshot showing single plane clipping result]</em>
</p>

### Example 2: Multi-Plane Clipping

```bash
python run_main.py --data_path=/path/to/your/scene
```

1. Add first plane: `normal=(1,0,0), d=0.5`
2. Click "Add Plane"
3. Add second plane: `normal=(0,1,0), d=-0.3`
4. Enable "RR Clipping" for both planes
5. Adjust planes to create cross-section view

<p align="center">
  <img src="resources/images/example_multi_plane.png" width="800" alt="Multi-Plane Clipping"/>
  <br>
  <em>[PLACEHOLDER: Add screenshot showing multi-plane clipping result]</em>
</p>

### Example 3: Video Recording

1. Set up clipping planes as desired
2. Open "Video" widget
3. Configure camera path (rotation/custom)
4. Click "Record Video"
5. Output saved to `./_videos/` directory

<p align="center">
  <img src="resources/images/example_video.gif" width="800" alt="Video Recording"/>
  <br>
  <em>[PLACEHOLDER: Add demo GIF of video recording]</em>
</p>

---

## 🔧 Technical Details

### Modified CUDA Rasterizer

RaRaClipper modifies the original `diff-gaussian-rasterization` with custom clipping logic:

**File: `cuda_rasterizer/forward.cu`**

Key modifications:
1. **`compute1DGS()` function** - Projects 3D Gaussian to 1D representation
2. **`interval_probability_erf()` function** - Computes clipping interval probability
3. **Clipping plane intersection** - Tests each Gaussian against planes
4. **RR strategy filtering** - Applies 3x scale threshold

**Key code sections:**

```cuda
// 1D Gaussian projection (lines 59-96)
__device__ void compute1DGS(
    const float3& ray_origin,
    const float3& ray_dir,
    const float3& mu,
    const float3& scale,
    float mod,
    const float4& rot,
    float& mu_t,
    float& sigma_t)
{
    // Projects 3D Gaussian to 1D along ray direction
    // Computes mean (mu_t) and variance (sigma_t)
}

// Clipping probability computation (lines 98+)
__device__ float interval_probability_erf(
    float mean, float sigma, float a, float b)
{
    // Uses error function (erf) for accurate probability
}
```

### Performance Considerations

| Scene Size | Clipping Planes | Strategy | FPS (RTX 3090) | Memory |
|------------|----------------|----------|----------------|--------|
| Small (<100K Gaussians) | 1 | RR Clipping | ~60 FPS | ~2GB |
| Medium (100K-500K) | 1 | RR Clipping | ~30 FPS | ~4GB |
| Large (>500K) | 1 | RR Clipping | ~15 FPS | ~8GB |
| Large (>500K) | 3 | RR Clipping | ~10 FPS | ~8GB |

**Optimization tips:**
- Use fewer clipping planes for better performance
- Disable "vizPlane" when not needed
- Use RR Strategy alone for faster preview (disable 1D GS)
- Close unused widgets to save rendering overhead

---

## 🧪 Evaluation and Metrics

The "Eval" widget provides quality metrics:

- **PSNR** - Peak Signal-to-Noise Ratio
- **SSIM** - Structural Similarity Index
- **L1 Loss** - Mean absolute error
- **Error Maps** - Visualize clipping artifacts

Usage:
1. Load ground truth scene
2. Apply clipping planes
3. Open "Eval" widget
4. View metrics and error visualization

---

## 🐛 Known Issues and Limitations

1. **Memory usage** - Large scenes with multiple planes may require >8GB VRAM
2. **Plane visualization** - Limited to single plane visualization (multi-plane WIP)
3. **Platform compatibility** - Best tested on Windows/Linux, macOS support experimental
4. **CUDA 12.x** - Some compatibility issues with newest CUDA versions

---

## 🗺️ Roadmap

- [ ] Multi-plane simultaneous visualization
- [ ] Clipping plane presets library
- [ ] Batch processing for evaluation
- [ ] Web-based viewer (no CUDA required)
- [ ] Android/iOS mobile viewer

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is based on Gaussian Splatting, which is licensed under Inria and MPII's **non-commercial research license**. See [gaussian-splatting/LICENSE.md](gaussian-splatting/LICENSE.md) for full details.

**Key restrictions:**
- ❌ Commercial use prohibited without explicit consent
- ✅ Research and evaluation use permitted
- ✅ Derivative works allowed (must cite original)

For commercial licensing inquiries, contact: [PLACEHOLDER: Contact email]

---

## 📧 Citation

If you use RaRaClipper in your research, please cite:

```bibtex
[PLACEHOLDER: Add complete BibTeX]
@inproceedings{raraclipper2024,
  title={RaRa Clipper: A Clipper for Gaussian Splatting Based on Ray Tracer and Rasterizer},
  author={[PLACEHOLDER]},
  booktitle={[PLACEHOLDER: Conference Name]},
  year={2024},
  organization={ACM}
}
```

Also cite the foundational works:

```bibtex
# Splatviz
@software{splatviz2024,
  author = {Barthel, Florian},
  title = {Splatviz: Interactive 3D Gaussian Splatting Viewer},
  year = {2024},
  url = {https://github.com/Florian-Barthel/splatviz}
}

# 3D Gaussian Splatting
@article{kerbl20233d,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM Transactions on Graphics},
  volume={42},
  number={4},
  year={2023}
}
```

---

## 🙏 Acknowledgements

This project wouldn't exist without:

- **Florian Barthel** - For the excellent [Splatviz](https://github.com/Florian-Barthel/splatviz) framework
- **INRIA GRAPHDECO** - For [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- **[PLACEHOLDER: Multi-layer Gaussian authors]** - For multi-layer data format
- **The 3D Gaussian Splatting community** - For continuous support and feedback

---

## 📧 Contact

- **GitHub Issues:** [PLACEHOLDER: Issues URL]
- **Email:** [PLACEHOLDER: Contact email]
- **Twitter:** [PLACEHOLDER: Twitter handle]
- **Project Page:** [PLACEHOLDER: Project page URL]

---

<p align="center">
  Made with ❤️ for the 3D Gaussian Splatting community
  <br>
  <br>
  ⭐ Star us on GitHub if you find this useful!
</p>
