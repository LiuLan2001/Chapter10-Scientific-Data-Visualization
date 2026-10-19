# 第十章：科学数据可视化配套代码

本仓库为可视化教材**第十章“科学数据可视化”**的配套代码，提供教材中部分示例的完整实现，便于读者结合教材内容进行实践和学习。

目前仓库主要包含以下章节的示例代码：

- **10.2.3**：科学数据可视化相关示例
- **10.2.5**：科学数据绘制相关示例
- **10.3.3**：基于隐式神经表示的科学数据超分辨率示例

其中，科学数据可视化部分主要使用 **VTK（Visualization Toolkit）** 实现，超分辨率部分使用 **PyTorch** 实现，并采用 **SIREN（Sinusoidal Representation Networks）** 对科学数据进行连续隐式表示和高分辨率重建。

---

## 1. 环境配置

建议使用 Python 3.9 及以上版本。

主要依赖包括：

- Python
- NumPy
- VTK
- PyTorch
- Matplotlib

可以使用 `pip` 安装主要依赖：

```bash
pip install numpy vtk matplotlib
