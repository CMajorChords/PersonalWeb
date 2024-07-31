<h2 style='pointer-events: none;'>Ubuntu 安装指南</h2>

 

Ubuntu 是一个基于 Linux 的免费操作系统，它是开源的，适用于个人电脑、服务器和其他设备。以下是 Ubuntu Desktop 的基本安装指南。在开始之前，请确保你有以下几样东西：

- 一台兼容的电脑。
- 至少 4GB 的 USB 闪存盘。
- 稳定的互联网连接。
<h3 style='pointer-events: none;'>1 安装ubuntu desktop</h4>
<h4 style='pointer-events: none;'>1.1 下载 Ubuntu ISO 文件</h4>

1. 访问 [Ubuntu 官网](https://www.ubuntu.com/download/desktop)。
2. 选择适合你的 Ubuntu 版本。对于大多数用户来说，最新的稳定版是最佳选择。
3. 点击 "Download" 按钮下载 ISO 文件。
<h4 style='pointer-events: none;'>1.2 制作启动 U 盘</h4>

1. 下载并安装一个 USB 刻录工具，如 [Rufus](https://rufus.ie/) 或 [UNetbootin](https://unetbootin.github.io/)。
2. 将 USB 插入电脑。
3. 打开你选择的刻录工具，选择下载的 ISO 文件和你的 USB 设备。
4. 点击开始，等待过程完成。
<h4 style='pointer-events: none;'>1.3 从 USB 启动</h4>

1. 将制作好的 USB 启动盘插入电脑。
2. 重启电脑，并在启动时按下相应键进入 BIOS 设置（通常是 F2, F12, Delete 或 Esc）。
3. 在 BIOS 设置中，改变启动顺序，使 USB 设备优先。
<h4 style='pointer-events: none;'>1.4 安装 Ubuntu</h4>

1. 电脑从 USB 启动后，将出现 Ubuntu 的安装界面。
2. 按照屏幕上的指示选择语言，然后点击 “安装 Ubuntu”。
3. 按照指示选择键盘布局、连接 Wi-Fi、分区设置等。
4. 根据指示完成安装过程。
<h4 style='pointer-events: none;'>1.5 完成安装</h4>

1. 安装完成后，系统会提示重启电脑。
2. 从 USB 设备启动时拔掉 U 盘，让电脑从硬盘启动。
3. 按照屏幕上的指示完成安装后的设置。
<h3 style='pointer-events: none;'>2 安装独立显卡驱动，配置深度学习环境</h4>

在 Ubuntu 上安装显卡驱动，特别是针对 NVIDIA 或 AMD 的独立显卡，是提高系统性能和兼容性的重要步骤。
<h4 style='pointer-events: none;'>2.1 NVIDIA 显卡驱动</h4>

1. 打开终端。
2. 更新系统的软件包列表：
```
sudo apt update
```
3. 安装 NVIDIA 驱动：
```
sudo ubuntu-drivers autoinstall
```
或者，你可以使用 `sudo apt install nvidia-driver-版本号` 来安装特定版本的驱动。

4. 安装完成后，重启电脑。
<h4 style='pointer-events: none;'>2.2 AMD 显卡驱动</h4>

对于 AMD 显卡，Ubuntu 通常会自动使用开源驱动。如需安装官方驱动：

1. 访问 [AMD 官网](https://www.amd.com/en/support) 并下载适合你显卡的驱动。
2. 解压下载的文件。
3. 按照 AMD 官方文档中的说明进行安装。
<h4 style='pointer-events: none;'>2.3 配置深度学习环境</h4>

深度学习环境的配置主要包括安装 Python、CUDA（如果使用 NVIDIA 显卡）、以及深度学习框架如 TensorFlow 或 PyTorch。
<h4 style='pointer-events: none;'>2.4 安装 Python</h4>

Ubuntu 默认已经安装了 Python。你可以通过以下命令检查 Python 版本：
```
python3 --version
```
如果需要安装或更新 Python，可以使用：
```
sudo apt install python3
```
<h4 style='pointer-events: none;'>2.5 安装 CUDA（NVIDIA 显卡专用）</h4>

如果你使用 NVIDIA 显卡，你需要安装 CUDA 来充分利用 GPU 的性能。

1. 访问 [NVIDIA CUDA 下载页面](https://developer.nvidia.com/cuda-downloads)。
2. 选择适合你的操作系统和版本。
3. 按照页面上的指示安装 CUDA。
<h4 style='pointer-events: none;'>2.6 安装深度学习框架</h4>

如果安装 TensorFlow，可以使用以下命令：
```
pip install tensorflow
```
或者，如果你使用 NVIDIA GPU，安装 GPU 支持的版本：
```
pip install tensorflow-gpu
```
如果安装 PyTorch，访问 [PyTorch 官网](https://pytorch.org/)，选择合适的安装命令，例如：
```
pip install torch torchvision torchaudio
```
如果安装 Keras，可以使用以下命令：
```
pip install keras
```