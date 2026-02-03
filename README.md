# RaRaClipper: Advanced Clipping Plane System for 3D Gaussian Splatting

<p align="center">
  <img src="resources/images/raraclipper_demo.gif" width="800" alt="RaRaClipper Demo"/>
  <br>
  <em>[PLACEHOLDER: Add demo GIF showing multi-plane clipping in action]</em>
</p>

## About | 关于

**English:** Official open-source implementation of "RaRa Clipper: A Clipper for Gaussian Splatting Based on Ray Tracer and Rasterizer", featuring an advanced clipping plane system for interactive 3D Gaussian Splatting visualization.

**中文：** "RaRa Clipper: 基于光线追踪器和光栅化器的高斯点云裁剪器"的官方开源实现，提供先进的裁剪平面系统，用于交互式 3D 高斯点云可视化。

---

## 📄 Related Paper | 相关论文

**RaRa Clipper: A Clipper for Gaussian Splatting Based on Ray Tracer and Rasterizer**
- **Conference:** [PLACEHOLDER: ACM Conference Name] 20XX / ACM Transactions on Graphics
- **Authors:** [PLACEHOLDER: Author names]
- **DOI:** [PLACEHOLDER: DOI link]
- **Paper PDF:** [PLACEHOLDER: Paper link]

---

## 🔗 Based on | 基于

This project builds upon excellent prior work:

- **[Splatviz](https://github.com/Florian-Barthel/splatviz)** - Interactive 3D Gaussian Splatting Viewer by Florian Barthel
- **[PLACEHOLDER: Multi-layer Gaussian Paper]** - Multi-layer Gaussian data representation

---

## ✨ Key Features | 核心特性

- 🎯 **Multi-Plane Clipping** - Advanced clipping plane system with interactive editing
- ⚡ **Dual Clipping Strategies** - RR Strategy (3x scale threshold) and 1D GS Strategy
- 🎨 **Interactive GUI** - Real-time visualization with ImGui-based controls
- 🔧 **Modified CUDA Rasterizer** - Custom diff-gaussian-rasterization with clipping logic
- 💾 **Scene Compression** - Support for compressed scene formats
- 🎬 **Media Export** - Video recording and screenshot capture
- 🎮 **Three Operation Modes** - Default, Decoder, and Attach modes

---

## 📚 Documentation | 文档

Choose your preferred language to get started:

选择您偏好的语言开始使用：

### 📘 English Documentation
For detailed installation, usage, and technical information, please refer to **[README_EN.md](README_EN.md)**

- ✅ Complete Installation Guide
- ✅ Quick Start Tutorial
- ✅ Technical Details
- ✅ GUI Controls Reference
- ✅ Citation Information

### 📗 中文文档
详细的安装、使用和技术说明，请参考 **[README_CN.md](README_CN.md)**

- ✅ 完整安装指南
- ✅ 快速开始教程
- ✅ 技术细节说明
- ✅ 界面控制说明
- ✅ 引用信息

---

## 🚀 Quick Start | 快速开始

```bash
# Install dependencies
pip install torch torchvision imgui-bundle click numpy imageio loguru Pillow open3d

# Build CUDA extensions
cd gaussian-splatting/submodules/diff-gaussian-rasterization && pip install -e .
cd ../simple-knn && pip install -e .
cd ../../..

# Run the application
python run_main.py --data_path=/path/to/your/ply/files
```

For detailed installation instructions, please see the full documentation in your preferred language above.

---

## 🛠️ System Requirements | 系统要求

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Windows / Linux | Windows / Linux |
| **GPU** | NVIDIA GPU (Compute Capability ≥ 7.0) | RTX 3000/4000 series+ |
| **CUDA** | 11.0 | 11.8 or 12.1 |
| **Python** | 3.8 | 3.8 - 3.10 |
| **RAM** | 8GB | 16GB |
| **VRAM** | 6GB | 8GB+ |

---

## 🙏 Acknowledgements | 致谢

This project builds upon:

- **[Splatviz](https://github.com/Florian-Barthel/splatviz)** - Interactive viewer framework
- **[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)** - Original implementation by INRIA GRAPHDECO
- **[PLACEHOLDER: Multi-layer Gaussian Paper]** - Multi-layer Gaussian data format

---

## 📄 License | 许可证

This project is based on Gaussian Splatting, which is licensed for **non-commercial research and evaluation use only**. See [gaussian-splatting/LICENSE.md](gaussian-splatting/LICENSE.md) for details.

本项目基于 Gaussian Splatting，仅供**非商业研究和评估使用**。详见 [gaussian-splatting/LICENSE.md](gaussian-splatting/LICENSE.md)。

For commercial licensing inquiries, please contact the original paper authors.

---

## 📧 Citation | 引用

If you find this work useful, please cite our paper:

```bibtex
[PLACEHOLDER: Add BibTeX citation]
@inproceedings{raraclipper2024,
  title={RaRa Clipper: A Clipper for Gaussian Splatting Based on Ray Tracer and Rasterizer},
  author={[PLACEHOLDER]},
  booktitle={[PLACEHOLDER]},
  year={2024}
}
```

---

## 📧 Contact | 联系方式

- **For implementation issues:** Open an issue on GitHub
- **For research questions:** [PLACEHOLDER: Contact email]
- **For Splatviz-related questions:** See [Splatviz Repository](https://github.com/Florian-Barthel/splatviz)

---

<p align="center">
  ⭐ If you find this useful, please star the repository!
  <br>
  如果您觉得这个项目有用，请给我们一个星标！
</p>
