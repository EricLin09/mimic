# Mimic — 当前架构

Mimic 是 Windows 用户态实时变声器：Rust 负责 WASAPI 音频、DSP 和推理调度，ONNX Runtime DirectML 运行固定的 RVC v2 模型，Tauri 2 + React 提供本地界面。训练和导出使用单独的 Python + PyTorch 环境（`train/`、`tools/export/`），运行时不依赖 Python。

本文描述已实现路径。当前默认为 **240 ms 块、2.5 s 历史上下文、60 ms 前瞻、50 ms 交叉淡化**；模型输出 40 kHz，设备通路 48 kHz。推理会话按布局固定维度，RNNoise 只用于门限副本，音高对齐声线中心，响度跟随为 0.5。

## 开发约束

- 保持现有 Applio + VB-Cable 工作环境不变，不写 `<APPLIO_ROOT>`；只读 Python 操作使用 `-I -B`，不写缓存。
- 不改变 Windows 默认设备；先选监控耳机或 WAV 输出，仅显式选择后才打开虚拟输出设备。
- 不添加驱动、服务、启动项或自动启动任务；虚拟设备由用户单独安装。
- 不读取/修改游戏进程，不注入 DLL；通过普通用户态音频设备连接语音软件。
- GPU 导出、parity 和重型 benchmark 前确认游戏退出，不把空闲性能当作游戏并发保证。
- 角色模型、训练集、角色素材和个人录音保持本地；公开模型唯一例外为 MIT 的 `models/silero_vad.onnx`，版权见根 LICENSE。

## 组成与模型

| 位置 | 职责 |
|---|---|
| `engine/src/realtime.rs` | CPAL/WASAPI 回调、无锁队列、独立处理线程、预热、遥测和错误处理 |
| `engine/src/pipeline.rs` | 布局、滚动上下文与音高缓存、逐块推理、输出 DSP |
| `engine/src/models.rs` | 进程级 ORT environment、typed helpers、可选固定维度和 IO binding API |
| `engine/src/dsp/` | RNNoise、双条件门限、Silero VAD、mel/f0、重采样、确定性噪声、SOLA/淡化 |
| `engine/src/gpu.rs` / `probe.rs` | 本进程 GPU 占用（界面遥测）、编码脉冲往返延迟探测（`live --measure`） |
| `engine/src/prepare.rs` | 训练数据准备：切片、重采样、contentvec 特征、RMVPE f0 |
| `engine/src/bin/mimic.rs` | `convert`、`devices`、`live`、`gate`、`prepare` CLI |
| `app/` | Tauri 2 bridge、React UI、mock engine；本地角色素材不随源码分发 |
| `train/` | RVC v2 网络定义（取自 Applio）、底模下载校验、单卡训练 |
| `tools/export/` / `tools/bench/` | ONNX 导出/parity、DSP 回归、客观度量和研究脚本 |

| 产物 | 输入 → 输出 | 说明 |
|---|---|---|
| `contentvec.onnx` | 16 kHz `[1,N]` → `[1,T,768]`，50 fps | HuBERT/contentvec，全局 attention |
| `rmvpe.onnx` | log-mel `[1,128,F]` → salience `[1,F,360]` | F 补到 32 倍数；mel、decode、coarse 在 Rust |
| `rmvpe_mel.f32` | 128×513 little-endian f32 | 与参考一致的 mel basis |
| `voices/<id>/retrieval.onnx` | `[T,768]` + index_rate → `[T,768]` | 内嵌索引，L2 top-8，逆平方距离权重 |
| `voices/<id>/synth.onnx` | phone、lengths、pitch、pitchf、sid、skip_head → audio | enc_p + flow + NSF HiFi-GAN |
| `voices/<id>/synth_det.onnx` | 上述输入加 latent/source noise、phase0 → audio | 整窗 flow 后裁 decoder 输入，引擎控制随机性和相位 |
| `silero_vad.onnx` | 16 kHz 音频 + recurrent state → probability / 新 state | CPU 单线程；每 32 ms 更新，可选 MIT 模型 |

`synth_det` 不存在时回退 legacy 路径。声线导出只需训练环境；`contentvec.onnx`、`rmvpe.onnx` 两张共享网络目前仍从 Applio 安装（`APPLIO_ROOT`）自带的上游代码和权重导出。权重和索引由用户准备。

## 默认实时路径

```text
48 kHz WASAPI input → input ring → processing thread, one 240 ms block
  ├─ gate copy: RNNoise, 10 ms frames → own 48→16 kHz resampler
  │     → >150 Hz RMS + Silero VAD + hysteresis / hold → convert this block or not
  └─ raw block (RNNoise too only with --denoise-audio)
       → stateful 48→16 kHz resample → rolling conversion window
       window = 2.5 s history + block + 50 ms fade, aligned to the 10/20 ms grids
       ├─ pitch: newest audio + 80 ms context → mel (64 frames) → RMVPE → decode → centre on the voice's median f0
       │          (f0_stats.json) + pitch offset → coarse pitch → rolling/frozen f0 cache
       ├─ features: contentvec → duplicate final frame → retrieval of newest region
       │             → nearest ×2 → unvoiced protect blend
       └─ synth_det, fixed-shape sessions: full-window enc_p + flow, keyed latent noise
            → decoder with 12-frame warm-up, keyed source noise, continuous NSF phase
            → 40→48 kHz resample → discard warm-up and 60 ms lookahead tail
            → envelope follower: 0.5 of each 10 ms loudness deviation, within ±6 dB
  → SOLA/crossfade (search default 0, phase-vocoder default)
  → output ring → explicitly selected WASAPI output / virtual microphone
```

回调与重型推理分离。实时启动预热模型并预填输出；队列不足、设备错误和耗时进入遥测/错误通路，不自动改变系统默认设备。

seamless 模式把块对齐到 20 ms，保持 contentvec 的 320-sample / 16 kHz 网格不漂移。默认窗口为 44,640 个 16 kHz 采样、279 个 100 fps 帧；contentvec 输出 139 帧，补一帧后 ×2。skip=250、lookahead=6、decoder margin=12，decoder 从第 232 帧开始渲染 47 帧，再丢预热/前瞻部分。

## 门限与降噪

确定：默认 RNNoise 只处理一份给门限的 48 kHz 副本，副本走独立的 48→16 kHz 重采样器；送进模型的是原始音频，`--denoise-audio` 或界面的“降噪进引擎”让 RNNoise 也作用于主干。RNNoise 自身的 voice probability 不用于开门。gate 同时检查 150 Hz high-pass side chain RMS 与 Silero 语音证据：

- 开门电平默认 −48 dBFS，已开门时低 6 dB。
- Silero 默认阈值 0.5，已开门时减 0.15；至少一个 32 ms chunk 给出语音证据。
- hold 名义 400 ms，按块向上取整；240 ms 默认块对应两块保持时间。
- 无 VAD 模型或 `--vad 0` 时退回电平条件；`--no-denoise` 完全不用 RNNoise，门限也看原始音频。

门关时仍推进上下文和音高/相位的时间轴，跳过主推理并消耗淡化尾音。VAD 只判断语音，不识别说话人，旁人讲话仍可能开门。RNNoise 默认只用于 gate；对干净输入，主干降噪与否到 offline 渲染的距离无显著差异，配对差 +0.004 ± 0.044，取舍依据听感。

## 一致性与优化边界

已有方法包括绝对时间键控 latent/source noise、连续 NSF phase、已输出帧 f0 冻结、decoder 左侧 12 帧预热和 6 帧 lookahead。相邻窗口仍重算全局 attention，同一帧表示可能变化；seamless 是工程模式名，不代表数学上逐样本一致。

默认仍为 phase-vocoder crossfade。相关度自适应淡化已提供独立模块与可选模式，它补偿统计功率而非任意短时 RMS，不因合成测试而自动改变默认。冻结特征/latent 与 decoder overlap-save 仍是 [流式设计](streaming-overlap-save-design.md)，没有接入默认管线。

固定维度 session 是 CLI 和界面的默认：`BlockProcessor::load` 从与处理器共用的 `Geometry` 推出各图维度，`--dynamic-shapes` 复现旧路径。A/B 总块中位数 53.0→40.4 ms，另一次独立测得 54.04→41.42 ms，采样 max error 4.465e−6。IO binding 仍只是接口。CPU 补帧/切片、×2 upsample、protect、f0 decode 和输出 DSP 仍构成设备边界，不能宣称 binding 自动实现四图全链零拷贝。

## 验证与目标

DSP 回归现用确定性 sweep + harmonics + seeded noise，fixtures/manifest 不含用户录音。普通 parity 不需要个人音频，生成新参考仍依赖本地只读参考环境。

本地 137 段对齐语音的 mean log-mel L1 到 offline 渲染距离：Mimic 240 ms / 2.5 s context 为 4.03 dB，Applio realtime 250 ms 为 4.14 dB；这是特定模型/测试集结果，不是通用听感保证。音高对齐使用声线全部原声浊音帧的中位数（`f0_stats.json`），说话人按 120 Hz 计。

远期目标仍是块≤150 ms、端到端≤200 ms、RTF≤0.5 与无明显接缝。**当前 240 ms 块加 60 ms 前瞻已经超过 200 ms，不满足该延迟目标。** 真实往返延迟还含队列和设备缓冲：UI 显示的是估算，设备通路用 `live --measure` 回环实测。空闲 GPU 性能不代表游戏争用时尾延迟。
