# CARLA 仿真环境数据集采集使用文档

## 1 项目概述

本项目基于 **CARLA 0.9.14** 自动驾驶仿真平台，实现车载多模态仿真数据的自动化采集。通过提供的 `sampler.py` 采样脚本，用户可以批量采集以下数据：

- **RGB 图像**（前置/环视相机）
- **深度图**（测距信息）
- **语义分割标签**（逐像素类别）
- **实例分割标签**（逐像素实例 ID）
- **车辆位姿**（位置、朝向、速度、加速度）
- **3D 包围盒**（其他车辆、行人、障碍物的精确框）
- **IMU 数据**（加速度、角速度、罗盘）
- **GNSS 数据**（经纬度、海拔）

采集的数据可用于 **3D 对抗伪装**、**目标检测**、**语义/实例分割**、**多传感器融合**等视觉感知算法的训练与评估。

---

## 2 环境依赖与 CARLA 安装部署

### 2.1 硬件与系统要求

| 项目         | 最低要求                                   |
| ------------ | ------------------------------------------ |
| 操作系统     | Windows 10 或更高版本（64 位）             |
| 显卡         | NVIDIA 独立显卡，显存 ≥ 8 GB，支持 CUDA 10.2+ |
| 内存         | 16 GB RAM（推荐 32 GB）                   |
| 处理器       | Intel Core i7 / AMD Ryzen 7 及以上        |
| 存储空间     | 至少 50 GB 可用空间（含地图扩展包）        |

> **注意**：若使用 Linux 系统，可参考 CARLA 官方文档进行编译安装，但本项目文档以 Windows 平台为主。

---

### 2.2 CARLA 下载与安装

#### 2.2.1 下载 CARLA 0.9.14

1. 访问 CARLA 官方 GitHub Release 页面：  
   https://github.com/carla-simulator/carla/releases/tag/0.9.14

2. 下载以下文件：
   - **CARLA_0.9.14.zip**（主程序包，约 2 GB）
   - **AdditionalMaps_0.9.14.zip**（可选，扩展城镇地图，约 7 GB）

3. 将压缩包解压至**无中文、无空格、无特殊符号**的路径，例如：D:\CARLA_0.9.14

若下载了附加地图，解压后将 `AdditionalMaps` 文件夹内的地图文件复制到 `D:\CARLA_0.9.14\CarlaUE4\Content\Carla\Maps` 目录下。

#### 2.2.2 安装 Python 依赖与 CARLA Python API

CARLA 0.9.14 的 Python API 以 `.whl` 文件形式提供，位于解压目录的 `PythonAPI` 文件夹中，文件名示例：
carla-0.9.14-py3.7-cp37-cp37m-win_amd64.whl

这表明该版本适用于 **Python 3.7**（64 位）。请确保你的 Python 环境版本匹配（推荐使用 Python 3.7.9 或 3.7.x）。

**安装步骤**：

1. 创建并激活 Python 虚拟环境（推荐）：
   ```bash
   python -m venv carla_env
   carla_env\Scripts\activate
   ```
2. 安装依赖包：
   ```bash
   pip install  基本的包
   ```

3.安装 CARLA Python API：
需要将此whl文件移动到工程目录，即此目录下，然后运行以下命令：
  ```bash
  pip install carla-0.9.14-py3.7-cp37-cp37m-win_amd64.whl
  ```
或者已知此whl文件的绝对路径，直接运行：
  ```bash
  pip install D:\CARLA_0.9.14\PythonAPI\carla-0.9.14-py3.7-cp37-cp37m-win_amd64.whl
  ```

4. 验证安装：
   ```bash
  python -c "import carla; print(carla.__version__)"
  ```
  应该输出 `0.9.14`，表示安装成功。



## 3 数据采集
### 3.1 启动 CARLA 仿真服务器
1.进入 CARLA 解压目录，双击运行 CarlaUE4.exe（或使用命令行启动）：
   ```bash
  D:\CARLA_0.9.14\CarlaUE4\CarlaUE4.exe
  ```   
2.如果是双击运行exe,等待服务器窗口出现，默认开启 TCP 端口 2000 和 流式端口 2001;如果是使用命令行启动，控制台提示 Server is ready 表示启动成功。


3.若需加载指定地图（如 Town03），可添加参数：

```bash
CarlaUE4.exe -quality-level=Low -carla-map=Town03
```

### 3.2 启动数据采集脚本
1.确保CARLA 仿真服务器已启动。修改sampler.py里面的参数进行采集。

2.进入项目根目录，运行以下命令启动数据采集脚本：
  ```bash
  python sampler.py
  ```
3.启动后，脚本会自动打开 CARLA 仿真服务器，进入指定地图（如 Town03），并开始采集数据。如果程序运行采集失败，可能是由于debug等导致CARLA在后台被占用，此时需要手动关闭CARLA服务器，然后重新运行sampler.py。
