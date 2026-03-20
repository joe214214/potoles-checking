# ECE 659 Project — 基于分层无线传感网络的实时路面坑洞检测系统

## 目录

1. [项目简介](#1-项目简介)
2. [快速启动](#2-快速启动)
3. [系统架构](#3-系统架构)
4. [数据流水线详解](#4-数据流水线详解)
5. [信道模型与丢包仿真](#5-信道模型与丢包仿真)
6. [TDMA时隙调度](#6-tdma时隙调度)
7. [Cluster Head：M/D/1排队 + 预过滤 + 数据融合](#7-cluster-headmd1排队--预过滤--数据融合)
8. [终端服务器：Random Forest坑洞判断](#8-终端服务器random-forest坑洞判断)
9. [可视化输出](#9-可视化输出)
10. [文件结构](#10-文件结构)
11. [技术栈](#11-技术栈)

---

## 1. 项目简介

本项目为 ECE 659（无线网络系统）课程项目，构建了一套**分层无线传感器网络（Hierarchical WSN）仿真系统**，用于检测真实道路上的路面坑洞（Pothole）。

系统基于真实 GPS 轨迹数据（Pittsburgh 城区），模拟 **50 辆搭载 IMU 传感器的车辆**在道路上行驶，通过无线网络将振动数据逐级上传、聚合、分析，最终在真实地图上实时标注坑洞位置。

**核心课程知识点覆盖：**

| 模块 | 课程知识点 |
|------|-----------|
| 无线信道 | Log-distance 路径损耗、对数正态阴影衰落 |
| 突发丢包 | Gilbert-Elliott 两态马尔可夫信道 |
| MAC 层 | TDMA 时分多路复用 |
| 排队论 | M/D/1 队列（验证 Little's Law） |
| 数据处理 | 加权平均数据融合 |
| 机器学习 | Random Forest 分类器 |

---

## 2. 快速启动

### 2.1 环境要求

```
Python 3.10+
```

### 2.2 安装依赖

```bash
pip install numpy pandas scikit-learn scipy joblib folium matplotlib
```

### 2.3 运行步骤（按顺序执行）

```bash
# 步骤 1：训练 Random Forest 模型（如已有 model/ 目录可跳过）
python train_model.py

# 步骤 2：运行 WSN 仿真主程序
#   输出：wsn_results.json、wsn_animation_data.json
python wsn_main.py

# 步骤 3：生成实时动态地图动画（HTML，浏览器打开）
python wsn_animate.py

# 步骤 4：生成静态分析图表和 Folium 地图
python wsn_visualize.py
```

### 2.4 查看结果

| 输出文件 | 查看方式 |
|---------|---------|
| `wsn_animation.html` | 浏览器打开（Chrome/Firefox），实时动画 |
| `wsn_map.html` | 浏览器打开，静态 Folium 地图 |
| `wsn_analysis.png` | 图片查看器，M/D/1 队列分析图 |

---

## 3. 系统架构

### 3.1 网络拓扑

```
┌─────────────────────────────────────────────────────────┐
│                     Pittsburgh 城区                      │
│                                                         │
│  [车辆 1-10]    →    [CH 0]  ─┐                        │
│  Trip 1 传感器   Haversine最近  │                        │
│                               │                        │
│  [车辆11-20]    →    [CH 1]  ─┤                        │
│  Trip 2 传感器                 ├──→  [Base Station]     │
│                               │     Random Forest      │
│  [车辆21-30]    →    [CH 2]  ─┤     坑洞判断            │
│  Trip 3 传感器                 │                        │
│                               │                        │
│  [车辆31-40]    →    [CH 3]  ─┤                        │
│  Trip 4 传感器                 │                        │
│                               │                        │
│  [车辆41-50]    →    [CH 4]  ─┘                        │
│  Trip 5 传感器                                          │
└─────────────────────────────────────────────────────────┘
```

### 3.2 节点数量与分布

| 层级 | 数量 | 说明 |
|------|------|------|
| 传感器节点 | 50 个 | 5 次行程 × 10 辆车，每辆搭载 IMU |
| Cluster Head | 5 个 | 固定物理位置，坐标为该 Trip 所有 GPS 点的均值中心 |
| Base Station | 1 个 | 终端服务器，运行 RF 分类器 |

### 3.3 Cluster Head 位置

每个 CH 的 GPS 坐标 = 该 Trip 所有车辆所有 GPS 采样点的**地理均值**（从数据自动计算），部署在路段中心位置，作为固定基础设施节点。

### 3.4 动态路由

传感器**每次发包时**动态计算到 5 个 CH 的 **Haversine 球面距离**，选择最近的 CH 发送。随着车辆行驶，归属的 CH 可能动态切换。

---

## 4. 数据流水线详解

```
传感器原始数据（CSV）
        ↓
  滑动窗口（20个采样点）
        ↓
  提取 11 维特征向量
  [ax_mean, ax_std, ay_mean, ay_std, az_mean, az_std,
   mag_mean, mag_std, mag_max, speed_mean, speed_std]
        ↓
  TDMA 时隙调度（等待分配时隙）
        ↓
  ┌─ 无线信道模型 ──────────────────┐
  │  路径损耗 + 阴影衰落 + 马尔可夫  │
  └─────────────────────────────────┘
        ↓ 成功 / 丢失
  Cluster Head M/D/1 队列
        ↓
  CH 粗过滤（边缘预判断）
        ↓ 通过
  加权数据融合
        ↓
  Base Station → Random Forest
        ↓
  坑洞位置标注（地图 + JSON）
```

特征提取逻辑与 `train_model.py` **完全相同**，保证训练/推理一致性。

---

## 5. 信道模型与丢包仿真

本项目设计了三层叠加的现实信道模型，模拟真实城市道路环境中多种丢包来源。

### 5.1 Log-distance 路径损耗

$$PL(d) = PL(d_0) + 10n \cdot \log_{10}\left(\frac{d}{d_0}\right)$$

| 参数 | 值 | 说明 |
|------|-----|------|
| 载波频率 $f_c$ | 900 MHz | 典型 IoT 频段 |
| 路径损耗指数 $n$ | 2.5 | 郊区/城市环境 |
| 发射功率 $P_{tx}$ | 20 dBm | |
| 有效噪声底 $P_{noise}$ | −87 dBm | 含接收机灵敏度限制 |
| 参考距离 $d_0$ | 1 m | |

由 SINR 推导 BPSK 误码率：$\text{BER} = \frac{1}{2}\,\text{erfc}\!\left(\sqrt{\text{SNR}}\right)$，再由误码率推导误包率：$\text{PER} = 1-(1-\text{BER})^L$，$L=352$ bits。

### 5.2 对数正态阴影衰落

$$PL_{\text{total}} = PL(d) + X_\sigma, \quad X_\sigma \sim \mathcal{N}(0,\,\sigma^2)$$

- $\sigma = 12\ \text{dB}$（密集城区，含建筑物遮挡、多径效应）
- 每次发包**独立采样**，模拟随机信号波动
- 在平均 SNR ≈ 10 dB 区域造成约 30～40% 丢包

### 5.3 Gilbert-Elliott 两态马尔可夫信道

模拟**突发性信号中断**（穿越隧道、建筑密集区、短暂干扰等）：

```
         P(好→坏) = 0.03            P(坏→好) = 0.25
             ┌───────────────────────────────┐
    GOOD ───→                               ←─── BAD
   正常信道                                    PER = 0.95
```

- 每个传感器节点**独立**维护一个马尔可夫状态
- 坏状态下强制 PER = 95%（模拟信号完全阻断）
- 平均连续丢包数 ≈ 4 个（$1 / P_{\text{坏→好}}$）

### 5.4 综合丢包效果

| Cluster Head | 平均距离 | 仿真 PDR |
|-------------|---------|---------|
| CH 0 | 271 m | ~70% |
| CH 1 | 251 m | ~69% |
| CH 3 | 431 m | ~51%（距离远，路径损耗更大）|
| CH 4 | 278 m | ~67% |
| **整体** | | **~66%** |

---

## 6. TDMA 时隙调度

为避免同一 Cluster Head 下多个传感器同时发包造成碰撞，采用 **TDMA（时分多路复用）** 调度：

```
TDMA 帧结构（每帧 3 秒）：
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ S0 │ S1 │ S2 │ S3 │ S4 │ S5 │ S6 │ S7 │ S8 │ S9 │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
  0.3 s/时隙，共 10 个时隙，按 vehicle_id 分配
```

- 时隙时长 = 0.3 s（与传感器采样间隔对齐）
- 帧长 = 10 × 0.3 = 3 s
- 传感器等待自己的时隙到来后再发包，增加少量延迟但**消除同簇内碰撞**
- 不同 CH 的 TDMA 帧**独立运行**

---

## 7. Cluster Head：M/D/1排队 + 预过滤 + 数据融合

### 7.1 M/D/1 排队模型

CH 的数据包处理遵循 **M/D/1 队列**：

| 符号 | 含义 |
|------|------|
| **M** | 到达过程近似泊松（多车辆独立随机到达） |
| **D** | 服务时间固定 $D = 0.1\ \text{s}$（确定性） |
| **1** | 单服务器（CH 处理核心） |

理论公式：

$$W_q = \frac{\rho}{2\mu(1-\rho)}, \qquad L_q = \frac{\rho^2}{2(1-\rho)}, \qquad \rho = \lambda D$$

**队列溢出**：队列超过 40 包时执行 tail-drop，模拟嵌入式设备内存耗尽。

仿真结果与理论值对比通过 **Little's Law**（$L = \lambda W$）验证，见 `wsn_analysis.png`。

### 7.2 粗过滤（边缘预处理）

CH 转发前进行轻量级坑洞预判断，过滤明显正常路面数据，减少上行传输量：

```python
def pre_filter(features):
    mag_max = features[8]   # 加速度量级峰值
    az_std  = features[5]   # 垂直方向加速度标准差
    return mag_max > 1.2 or az_std > 0.15
```

策略：**宁可误报，不可漏报**（高召回率，宽松阈值）。精确判断交由 Base Station 完成。

### 7.3 加权数据融合

同一时间窗口（3 s 内）通过预过滤的多个数据包在 CH 端进行融合：

$$\mathbf{f}_{\text{fused}} = \frac{\displaystyle\sum_i w_i \cdot \mathbf{f}_i}{\displaystyle\sum_i w_i}, \qquad w_i = \frac{1}{PL_i\ \text{[dB]}}$$

- 权重 = 路径损耗的倒数（**信号更好的包贡献更大**）
- 融合后输出单个 11 维特征向量上报 Base Station

---

## 8. 终端服务器：Random Forest坑洞判断

Base Station 加载预训练的 **Random Forest** 分类器，对 CH 上报的融合特征向量进行精确分类：

```
输入：11 维融合特征向量
输出：0（正常路面）/ 1（坑洞）
```

- 模型文件：`model/pothole_rf_model.pkl`
- 训练脚本：`train_model.py`
- 检测结果包含：时间戳、GPS 坐标、来源 CH 编号

**两级处理设计对比：**

| 层级 | 位置 | 算法 | 特点 |
|------|------|------|------|
| 第一级 | Cluster Head | 阈值规则 | 低算力、高召回率、减少回传流量 |
| 第二级 | Base Station | Random Forest | 高精度、最终决策 |

---

## 9. 可视化输出

### 9.1 实时动态地图（`wsn_animation.html`）

用 Chrome/Firefox 打开，功能：

- 50 辆车按真实时间戳在 OpenStreetMap 上移动（每个 Trip 不同颜色）
- **绿色闪线** = 成功传输的数据包
- **红色虚线** = 信道丢失的数据包（路径损耗过大 / 马尔可夫坏状态）
- CH 队列长度实时数字显示
- ⚠️ 坑洞图标在被检测时动态出现
- 播放/暂停、进度条拖拽、速度调节（1x ～ 120x，默认 30x）

### 9.2 静态 Folium 地图（`wsn_map.html`）

- 50 条车辆轨迹（按 Trip 着色）
- 5 个 CH 位置标注（蓝色圆形）
- 所有检测到的坑洞位置（红色圆点，可点击查看详情）

### 9.3 统计分析图（`wsn_analysis.png`）

- M/D/1 队列长度时间序列
- 等待时间分布直方图
- Little's Law 验证柱状图（仿真值 vs 理论值）
- 各 CH 的 PDR 对比柱状图

---

## 10. 文件结构

```
project/
├── wsn/                        # WSN 仿真模块包
│   ├── __init__.py
│   ├── channel.py              # 信道模型（路径损耗 + 阴影衰落 + 马尔可夫）
│   ├── topology.py             # CH 位置计算 + Haversine 最近 CH 路由
│   ├── sensor_node.py          # 传感器节点（读 CSV + 特征提取）
│   ├── cluster_head.py         # CH（TDMA + M/D/1 + 预过滤 + 融合）
│   └── base_station.py         # 终端服务器（RF 模型推理）
│
├── simulated_trips/            # 原始数据（5 Trip × 10 辆车 CSV）
│   └── trip{i}_vehicle{j}.csv
│
├── model/
│   └── pothole_rf_model.pkl    # 预训练 Random Forest 模型
│
├── train_model.py              # 模型训练脚本
├── wsn_main.py                 # 仿真主程序（生成 JSON 结果）
├── wsn_animate.py              # 生成实时动画 HTML
├── wsn_visualize.py            # 生成静态地图 + 分析图表
│
├── wsn_results.json            # 仿真统计结果
├── wsn_animation_data.json     # 动画事件数据（供 wsn_animate.py 读取）
├── wsn_animation.html          # 实时动态地图（浏览器打开）
├── wsn_map.html                # 静态 Folium 地图
├── wsn_analysis.png            # M/D/1 队列分析图
└── README.md                   # 本文档
```

---

## 11. 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 数据处理 | NumPy, Pandas |
| 机器学习 | scikit-learn（Random Forest）|
| 信道仿真 | SciPy（erfc 函数，BPSK BER 计算）|
| 静态地图 | Folium（基于 OpenStreetMap）|
| 动态动画 | Leaflet.js（嵌入自包含 HTML）|
| 图表 | Matplotlib |
| 模型持久化 | Joblib |

---

> **数据来源**：Pittsburgh 城区真实道路 GPS + IMU 数据（经人工扩展至 50 辆车）
> **课程**：ECE 659 — Wireless Network Systems
