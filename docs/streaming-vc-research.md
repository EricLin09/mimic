# 固定 RVC v2 的流式去接缝调研

调研日期：2026-10-27。适用对象：Mimic，固定 RVC v2 权重，Rust + ONNX Runtime，不能重新训练。

**结论：应先让每个绝对时间的特征、pitch、latent 和 excitation 只有一个已提交版本，再对有限感受野的 decoder 做有上下文的 valid-region 拼接；继续改交叉淡化只能补偿部分症状。** 全局 attention 的滑窗推理不能借助普通 KV cache 精确还原整段推理；冻结可以消除“同一帧被不同版本覆盖”，却不能保证与 offline 等价，也不能自动消除新旧成熟帧之间的表示变化。

本文用 **[来源事实]** 标记源码、论文或作者资料支持的结论；**[本地已有证据]** 是仓库中已有日志，非本次重新跑模型；**[工程推断]** 是针对 Mimic 的建议，不能当作已验证的音质或速度承诺。所有延迟数字保留原测量口径，不能直接横向排名。

## 1. 现有 RVC 客户端：重叠重算、SOLA 和 crossfade 是常规方案，增大块长的改善同时包含调度和模型上下文因素

### 1.1 w-okada voice-changer

**一句话结论：其公开 legacy RVC 路径通过保留历史音频、重算窗口、局部互相关选偏移、cos²/sin² crossfade 来拼块，没有使两次重叠区推理成为同一个结果。**

**[来源事实]** `VoiceChanger.py` 的 `on_request_sola` 使用 12 ms SOLA 搜索，overlap 为配置值与当前 block 长度的较小值；淡化曲线在指定 offset/end 区间由 cos² 生成。RVC `generate_input` 拼入新音频，并截取 `input + crossfade + search + extraConvertSize` 的窗口。[VoiceChanger.py](https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/VoiceChanger.py)，[RVC.py](https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/RVC/RVC.py)。

**[来源事实]** 官方旧版 RVC 教程将 CHUNK 解释为一次收集/转换的长度，将 EXTRA 解释为历史上下文；建议出现断续时增大 CHUNK，并要求缓冲时间足以覆盖处理时间。这里的 EXTRA 不是额外等待未来输入。[官方 RVC 教程](https://github.com/w-okada/voice-changer/blob/master/tutorials/tutorial_rvc_en_latest.md)。

**[来源事实：用户经验，非受控实验]** 官方仓库的 [issue #1323](https://github.com/w-okada/voice-changer/issues/1323) 报告 v2 客户端在较小 chunk 下更容易出现 glitch，怀疑与旧版可调选项缺失有关。该报告只能证明存在这类反馈，不能证明因果关系。VCClient v1/v2、Beatrice/RVC 后端的实现不同，不能把上述 legacy Python 路径描述为所有当前版本的共同实现。

**[工程推断]** 对固定 RVC 直接适用，代价是重复计算；SOLA 可以选一个整体时间偏移，不能同时修复所有谐波和非平稳成分的相位。优先记录实际 `block samples / sample rate`，不要把客户端 CHUNK 的 UI 单位直接当毫秒。

### 1.2 RVC-WebUI realtime

**一句话结论：`extra_time` 提供历史上下文，`crossfade_time` 和 SOLA 保留/消费重叠；`rms_mix_rate` 是输入输出包络混合，不是相关度自适应保功率淡化。**

**[来源事实]** 为避免当前 `main` 文件路径变动，本节核查官方 tag `2.2.231006`。`gui_v1.py` 构造 `extra + crossfade + 10 ms search + block` 音频窗口；SOLA 后按淡化窗混合并保存尾段。其 RMS 处理为 `infer_wav *= (rms_input / max(rms_output, eps)) ** (1-rms_mix_rate)`，发生在 SOLA 前。[官方固定版本 gui_v1.py](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/2.2.231006/gui_v1.py)。

**[来源事实]** `tools/rvc_for_realtime.py` 对当前整个输入窗口调用 HuBERT `extract_features`；RVC v2 取第 12 层，检索 top-8，并缓存/滚动更新 pitch。保留 pitch cache 不等于保留全部 Transformer hidden states。[官方固定版本 rvc_for_realtime.py](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/2.2.231006/tools/rvc_for_realtime.py)。

**[工程推断]** 对 Mimic 应分别做 `rms_mix_rate=1` 与当前值的 A/B：如果块同步电平起伏显著减小，应先查包络估计与边缘插值。加长 `extra_time` 不直接增加历史等待，但会增加推理耗时；延长 crossfade 可能增加保留尾段的播放延迟。上述 tag 没有 `phase_vocoder` 分支，不能把 Applio 的 phase crossfade 追溯到该固定版本。

### 1.3 Applio realtime

**一句话结论：Applio 仍以 SOLA 接重渲染块，phase-vocoder crossfade 是可选补救，并非模型状态连续。**

**[来源事实]** 核查时的 `core.py` 使用 10 ms SOLA search 和 sin²/互补淡化窗；phase 分支从重叠两段的 FFT 幅度与包裹相位差合成过渡分量，普通分支直接时域混合。实现还处理 gate 后的有限尾音消耗。[Applio core.py](https://github.com/IAHispano/Applio/blob/main/rvc/realtime/core.py)。

**[工程推断]** 这个 phase 算法不是“输出功率严格等于两侧目标功率”的约束求解；不同谐波、噪声和瞬态会影响合成分量。可在相同冻结输入下独立测试 crossfade，而不把所有机械感归因于 DSP。

### 1.4 为什么 250 ms 比 120 ms 自然

**[工程推断，需实验拆分]** 至少有四个独立因素：

1. 接缝次数由约 8.33 次/秒降到 4 次/秒，局部同等级误差更少出现。
2. 每块保留区域的帧拥有不同数量的块内未来上下文，增大块长改变了“成熟度”的分布；即使额外 lookahead 相同也如此。
3. 更大的块给 GPU 调度、数据传输、F0 提取与音频队列更多余量，减少欠载；这是系统问题，需要 underrun 计数区分。
4. 重新估计的包络、features、f0 和随机源变化的频率降低；周期性电平扰动也变慢。

没有找到能把同一固定 RVC 模型的 120/250 ms 差异拆成上述因素的公开受控 MOS 数据。应避免把社区“块大更好”概括成普适音质定律。

## 2. Beatrice v2、LLVC 和 StreamVC：低延迟来自为流式设计的网络和持久状态，不能靠给 RVC 加缓存复制

**一句话结论：这些系统限制未来依赖，并让后续 chunk 继承有效状态；不存在“几十毫秒且已证明完全没有拼接感”的通用结论。**

| 系统 | 来源证实的机制/数字 | 固定 RVC 可迁移部分 | 限制与质量影响 |
|---|---|---|---|
| Beatrice 2 | 官方发布训练代码，含 `CausalConv1d`、带显式 delay 的 ConvNeXt，以及分组时间序列上的因果 attention mask | 明确未来预算、持久状态、分阶段接口 | 架构/权重不同；不能将 RVC 权重直接装进去 |
| LLVC (2023) | dilated causal conv encoder + causal chunk Transformer；推理携带 `enc_buf/dec_buf/out_buf/convnet_pre_ctx` | 缓存边界定义和 stream/offline 验证方式 | 需要其专门训练模型，论文数字不是 Windows WASAPI 的整条往返延迟 |
| StreamVC (2024) | causal convolution、streaming-aware modules；20 ms 触发，论文报告 60 ms architecture + 10.8 ms compute | 持久卷积状态、明确 lookahead 计账 | 训练 content encoder 和 decoder；不能替换固定 RVC 的 HuBERT 而不验证分布 |

**[来源事实]** Beatrice 官方网站给出的产品延迟约 0.05 s，注明是外部录音设备实测；公开训练代码证明它包含上述因果/有限 delay 元件。官方优化推理库与公开训练实现不是同一份完整代码，因而本报告不声称已经审计其每层缓存实现，也不采用第三方包装项目的 sub-10 ms 宣传。[Beatrice 官方网站](https://prj-beatrice.com/)，[官方训练代码](https://huggingface.co/fierce-cats/beatrice-trainer/blob/main/beatrice_trainer/__main__.py)（页面标识的文件版本：[f34836d](https://huggingface.co/fierce-cats/beatrice-trainer/commit/f34836de014b86956096878aecb8d3b17feaaa0b)）。

**[来源事实]** LLVC 论文报告在 16 kHz、消费级 CPU 上低于 20 ms 的延迟及约 2.8 倍实时速度；官方源码的 `infer_stream` 连续传递缓存并拼接新输出，而不是每次独立整窗合成再做 RVC 式 crossfade。[LLVC 论文](https://arxiv.org/abs/2311.00873)，[model.py](https://github.com/KoeAI/LLVC/blob/main/model.py)，[infer.py](https://github.com/KoeAI/LLVC/blob/main/infer.py)。

**[来源事实]** StreamVC 的论文数字来自单核 Pixel 7 的模型计算测量；lookahead 由训练对齐和 F0 所需上下文共同决定。[StreamVC §2.4](https://arxiv.org/html/2401.03078v1#S2.SS4)。

**[工程推断]** Mimic 可以采用同样的状态管理纪律；在“不重训”的约束下，必须保留固定 RVC 的表示和声码器，并接受有限窗口对全局 attention 的近似。为了低延迟直接改成 causal mask，是另一个模型行为，不能称为等价缓存优化。

## 3. 非因果声码器：已有 post-training 精确流式先例，但未来感受野必须由延迟支付

**一句话结论：有限时间支持的预训练卷积网络可用上下文裁剪或 cached convolutions 精确流式化；全局运算、随机源和相位必须另行处理。**

### 3.1 最直接先例：cached_conv

**[来源事实]** Caillon 与 Esling 的 *Streamable Neural Audio Synthesis With Non-Causal Convolutions* (DAFx 2022) 通过训练后重配置、cached padding 和分支延迟对齐，把非因果卷积模型改为 buffer-based streaming；作者在 RAVE 上验证生成质量保持，并提供实现。[论文与作者项目页](https://acids-ircam.github.io/cached_conv/)，[代码和 ConvTranspose 示例](https://github.com/acids-ircam/cached_conv)。这比寻找“更好的 crossfade”更直接对应本任务。

**[工程推断]** 适用于 RVC decoder 的局部卷积部分，也适用于固定输入后的 flow。它不是把任意现成 ONNX 一键转换：Rust/ONNX 应显式导出每层 cache 输入输出，或者先采用带 halo 的完整子图推理，以建立等价性基准。

### 3.2 overlap-save / valid-region inference 的正确含义

**[来源事实]** 原版 HiFi-GAN generator 由局部 Conv1d、ConvTranspose1d、残差分支和逐点非线性组成，标准权重归一化不是运行时跨时间统计。[HiFi-GAN 作者 models.py](https://github.com/jik876/hifi-gan/blob/master/models.py)。

**[工程推断：推导]** 如果某段输出的所有依赖都落在输入区间 `[t-L, t+B+R)` 内，并且该区间与全流输入逐值相同，那么计算这个片段后裁掉左右受 padding 污染的区域，保留 `[t,t+B)`，可与全段的同一区域一致。这是依赖局部性，不要求网络线性；不能把 FFT overlap-save 的线性卷积公式直接套到整个神经网络。

- halo 推理仍重复计算上下文；准确说法是“每个输出采样只提交一次”。只有进一步引入逐层 activation cache 才减少相应重复算子。
- 右侧 R 不足时仍会产生边界误差；左侧 12 帧预热不能代替右侧 lookahead。
- 对 ConvTranspose 必须保留全局 stride 相位、padding 和 output_padding，所有分支按同一时间轴对齐。不能在每块局部原点重新选择 crop。
- 全段与块推理必须使用相同的绝对时间噪声、连续 source phase、相同 f0 插值及 padding；否则不是卷积等价性测试。
- 先在 CPU FP32 固定输入上检查 max error `<1e-4`，再核查目标 ONNX provider；SNR 高不等价于最大误差已过门槛。

### 3.3 iSTFTNet / HiFTNet 不等于无条件流式化

**[来源事实]** iSTFTNet 用显式 inverse STFT 替换部分高采样率网络层以减少计算；HiFTNet 又加入 harmonic-plus-noise source filter。它们首先是声码器架构，不是“任意 HiFi-GAN 权重转流式”的工具。[iSTFTNet](https://arxiv.org/abs/2203.02395)，[HiFTNet](https://arxiv.org/abs/2309.09493)。

**[工程推断]** 对固定 RVC 不直接适用替换权重。值得借鉴的是显式维护 oscillator/source 和 iSTFT overlap state；iSTFT 还必须维护 analysis/synthesis 窗的归一化和时间原点，错误裁剪同样会点击。

## 4. HuBERT / WavLM / contentvec：滑窗冻结有依据，普通 KV cache 不提供双向模型的精确增量推理

**一句话结论：固定窗口、固定采样网格、延迟提交成熟帧是可行近似；公开研究支持其任务可用性，但未给出 Mimic 所需的特征余弦与听感保证。**

### 4.1 可用的方法与不可直接迁移的缓存

**[来源事实]** *Effective Context in Neural Speech Models* (Interspeech 2025) 不改 HuBERT 架构或权重，而用 history + lookahead 滑窗模拟 streaming；第 9 层 phone probing 中，400 ms lookahead、无限 history 为 12.5% error，完整上下文为 11.9%；400 ms lookahead、2 s history 为 13.7%。这是 phoneme probing 数据，不是 60–90 ms lookahead 的 VC 无损结论。[原论文 §4](https://www.isca-archive.org/interspeech_2025/meng25b_interspeech.pdf)。

**[工程推断]** 三种选项应明确区分：

| 方法 | 能否不重训 | 是否精确等于原双向全段模型 | Mimic 的意义 |
|---|---|---|---|
| 同长度滑窗，提交具有 L 未来上下文的帧 | 能 | 否 | 首选可控近似，记录每帧来源窗口 |
| 加 causal/chunked attention mask 并 KV cache | 可机械修改，但效果未保证 | 否 | 改了依赖关系，必须单独验质量 |
| 对原双向 attention 缓存旧 K/V 而不重算 | 可实现 | 通常否 | 深层旧 hidden states 会因新未来输入改变，旧 K/V 随之过期 |
| 专门训练/蒸馏 streaming encoder | 能训练时可行 | 与原模型不同 | 超出固定 RVC 任务约束 |

推导：第一层 query/key/value 若由固定局部输入构造，部分投影可以复用；但加入新 key 后 softmax 分母和旧帧 attention output 已改变。第二层 K/V 依赖第一层更新后的旧 hidden states，因此不能把 causal LLM 的 append-only KV cache 直接搬过来声称精确。

### 4.2 具体执行规则

**[工程推断]** 以绝对 16 kHz 输入采样时间为基准，50 fps 的 contentvec 起点对齐到 320-sample 网格；100 fps 的 f0/latent 使用明确的 160-sample 网格和固定的 feature ×2 映射。窗口长度、位移、特征索引必须同时考虑 CNN 有效核宽及导出 padding，不能只按张量长度猜时间戳。

成熟之后保存 feature、retrieval 结果、protect 混合与 f0，后续使用这些提交值。enc_p 也要确定自己的窗口和提交延迟：只冻结输入 feature，并不能让 enc_p 的全局 attention 输出自动保持不变。对归一化层另做轴审计：若 HuBERT 前端使用跨时间 GroupNorm，即使 CNN 卷积核有限，归一化结果仍可随窗口变化；不能只缓存 CNN valid 区就承诺等价。

**[公开数据缺口]** 本次检索未找到与“同一 RVC contentvec 帧、窗口右移 120 ms、分不同 lookahead、报告余弦及 seam MOS”完全匹配的公共数据。任务给出的 `0.19/0.80/0.92` 是项目测量输入，不能标成 HuBERT 普遍结论。

建议本地 sweep `L_feat = 20/40/60/80/100 ms`，对同一绝对帧报告：余弦 p5/p50、normalized L2、top-8 邻居变更率、enc_p 的 m/logs/z 差异；再测接缝周边 log-mel 差异与块频率附近的包络调制。分开评估“冻结后没有重写”和“冻结得足够成熟”。

## 5. 拼接 DSP：相关度自适应保功率最适合作为轻量补丁，phase 方法应按 voiced/transient 分流

**一句话结论：先对齐，再根据相关度补偿功率；仅对稳定有声音段考虑 harmonic phase alignment，任何拼接器都不能补回缺失的音素上下文。**

**[来源事实]** Fink、Holters、Zölzer 的 DAFx 2016 论文明确推导了相关系数驱动的 power-complementary crossfade，覆盖相关与负相关信号。[Signal-matched power-complementary cross-fading and dry-wet mixing](https://www.hsu-hh.de/ant/wp-content/uploads/sites/699/2017/10/Fink-Holters-Z%C3%B6lzer-2016-Signal-matched-power-complementary-cross-fading-and-dry-wet-mixing.pdf)。

**[工程推断：功率关系]** 假设两侧局部均值为零、功率同为 P，相关系数为 ρ，输出 `y=a*x+b*z` 的期望功率为 `P*(a²+b²+2ρab)`。对原有互补增益 `u=1-v`，可用

```text
den = sqrt(u*u + v*v + 2*rho*u*v)
a = u / den
b = v / den
```

令功率模型恒定。中点：ρ=0 时原曲线为 −3.01 dB，ρ=0.5 时为 −1.25 dB；直接换 equal-power sin/cos 则在 ρ=1 时造成 +3.01 dB。局部功率不同应把两侧方差和互协方差显式放入分母，或先做平滑电平匹配。短窗随机噪声、瞬态和时变相关性意味着“期望保功率”不保证任意 5 ms RMS 精确恒定。

| 方法 | 适合情况 | 延迟/计算 | 固定 RVC 的取舍（工程推断） |
|---|---|---|---|
| SOLA / WSOLA | 大致同一波形，主要差异为时间位移 | 需要 search 区，计算低至中 | 已有 SOLA 先保留；不要把一次全带偏移当成全部谐波相位一致 |
| 相关度自适应 crossfade | 已对齐，ρ 约 0.3–0.95，局部电平接近 | 时域 O(N)，可不增加既有 overlap 延迟 | 优先 A/B，能针对块同步功率凹陷；不解决音色改变 |
| phase-vocoder crossfade | 稳定有声、谱峰相近、相位偏移明显 | FFT 或显式正弦合成；已有 overlap 可复用 | 可能减少抵消；对瞬态/清辅音可能模糊，需客观电平指标和听测 |
| harmonic / peak phase alignment | F0 稳定、谐波可可靠追踪 | 需要谱窗、相位跟踪与状态 | 只动 harmonic 分量，噪声分量保留统计；F0 跳变处退回简单淡化 |
| 4–8 子带的相关度增益 | 各频带相关性差异大 | STFT/filterbank，需维持 overlap state | 增益平滑且保持复数相位；额外 group delay 必须计入预算 |

**[来源事实]** WSOLA 的原始研究以局部 waveform similarity 选择同步位置，主要用于 time-scale modification，而不是神经表示重渲染一致性。[Roelands 与 Verhelst，1993](https://www.isca-archive.org/eurospeech_1993/roelands93_eurospeech.html)。谱峰 phase locking 则针对 phase-vocoder 跨 bin coherence；其原始作用场景与 RVC 两次独立重渲染并不相同。[Laroche 与 Dolson，1999，DOI](https://doi.org/10.1109/89.759041)。

**[工程推断]** 可将 phase alignment 后的两段重新估计 ρ，再应用受限的增益补偿；不要把 phase 合成过渡的输出功率仍按原始两段的 ρ 计算。ρ 接近 −1 时正向混合会近乎抵消并要求很大增益，应限制增益、拒绝低置信度对齐，必要时缩短交接；不要逐频带随意翻相。RMS 稳定也不等于听感自然，须保留频谱/瞬态与盲听检查。

## 6. 2023–2026 的进展：主要方向是从建模上控制上下文和跨块条件，而不是一个可直接修复固定 RVC 的通用后处理

**一句话结论：有直接关注跨块 consistency 或 small-chunk degradation 的方法，但多数需要新训练，最接近固定模型限制的仍是卷积缓存及有延迟的表示提交。**

| 年份/工作 | 来源证实的主要机制 | 对“重渲染不一致”的关系 | 固定 RVC 适用性 |
|---|---|---|---|
| 2023 LLVC | causal encoder、chunk Transformer、状态传递 | 避免每块从零独立生成历史 | 借接口/缓存纪律，模型不可直接迁移 |
| 2023 DualVC 2 | dynamic chunk mask 的非因果卷积、Conformer、流式训练；报告 186.4 ms | 显式限定可见未来并利用块内上下文 | mask/训练不能当无损后处理 |
| 2024 StreamVC | streaming-aware causal codec 风格 encoder/decoder | 持续状态产生新音频 | 需要新模型 |
| 2024 StreamVoice | causal context-aware LM、teacher foresight、semantic masking | 学习补偿未来缺失，旧 autoregressive 历史作为条件 | 需要训练；LM 零 lookahead 不等于完整 pipeline 零延迟 |
| 2024 CosyVoice 2 | chunk-aware flow matching、lookahead convolution、流式/非流式统一训练 | 通过训练约束 chunk 依赖；官方代码也保留 source/audio cache 和 fade | source cache 思路有用，不能假设所有 streaming TTS 都已去掉 fade |
| 2025 Effective Context | 无需改权重的 HuBERT 滑窗、history/lookahead sweep | 为近似冻结提供表示级依据 | 直接相关，但论文实验未来量超出目标预算 |
| 2026 MeanVC 2 | future-receptive chunking，有限未来 mask；单核首包约 110 ms | 直接针对小 chunk 退化和跨块稳定性 | 需要其训练，不能套到固定 enc_p |
| 2026 StyleStream 2.0 | few-step MeanFlow + autoregressive feedback fine-tuning | 用流式 feedback 训练提高生成稳定性 | 需要训练；作者报告 520 ms，不满足 Mimic 的 200 ms 目标 |

**[来源事实]** DualVC 2 的原文强调因果卷积不能充分利用 chunk 内未来，改为 dynamic masked convolution；这说明“小块”与“因果”不是免费的音质收益。[DualVC 2](https://arxiv.org/html/2309.15496v1)。StreamVoice 的 teacher-guided foresight 与 masking 是训练策略。[ACL 2024 原论文](https://aclanthology.org/2024.acl-long.396/)。

**[来源事实]** CosyVoice 2 的 chunk-aware flow matching 具有专门的未来卷积和训练设计；核查到的官方 `CosyVoice2Model.token2wav` 仍把缓存 mel/source 接到新输入，保存 source/speech 并做 waveform fade。[CosyVoice 2 论文 §2.4](https://arxiv.org/html/2412.10117v1#S2.SS4)，[官方 model.py](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/cli/model.py)。它是 TTS 先例，不能用其文本首包延迟代替 live VC 的 mic-to-output 延迟。

**[来源事实]** MeanVC 2 的 40 ms 输出 chunk 加一块有界未来，在 BNF 层形成 80 ms 首包缓冲；109.88 ms 是单核 AMD EPYC 7542 的首包 pipeline 测量，不含本机 WASAPI 的实测承诺。[MeanVC 2，2026-06 预印本](https://arxiv.org/html/2606.09050v1)。StyleStream 2.0 作者页报告 RTX 4060 上 520 ms，并明确使用 feedback fine-tuning。[StyleStream 2.0 作者项目页](https://berkeley-speech-group.github.io/StyleStream2.0/)。

本次检索没有找到可直接对固定 RVC v2 的 HuBERT + enc_p 全局 attention + NSF decoder 施加、无需训练、同时证明 `<200 ms` 与完全无重渲染接缝的现成方案。

## 7. Mimic 已有本地证据及优先级

**一句话结论：仓库已经有 decoder 分块近似可行、但 DirectML 拆图反而变慢的证据，应先复用这些结论制定验收，再投入更复杂缓存。**

**[本地已有证据；本次仅阅读日志，未重新测量]**

- rf_chunking.log：decoder 左 12 帧、右 8 帧的保留区 SNR 为 61.2 dB；右 12 帧为 63.9 dB。是该测试的聚合 SNR，不是 max error `<1e-4` 的证明。enc_p 各层 attention 在 ±10 帧内的质量比例差异很大，部分层仅约 2%–6%，不能把 relative position window 10 当作 attention mask。
- sim_frozen.log：`rt_la3_le2_r8` 与相同冻结 latent 的 whole decode SNR 54.9 dB；冻结设计相对 full 的 mel-L1 仍为 4.34 dB。冻结 decoder 输入的一致性，与全流特征是否足够接近 offline，是两个问题。
- export_dec_only.log：encflow/decoder split 对原 `synth_det` 的同输入 max difference `9.03e-7`。这是整段拆图 parity，不是任意上下文长度的 chunk parity。
- bench_dml_split.log：120 ms 的冻结拆图估算总计 63.4 ms，RTF 0.53；原 monolithic synth 子阶段为 17.0 ms，而拆出的 decoder 约 39.6 ms。前者总 pipeline 与后者单 synth 不能当整条 pipeline 直接相除；但 decoder 的形状/图优化成本明显，理论计算量变小不能保证墙钟时间降低。

以下排序是 **[工程推断]**，置信度分开评价“机制成立”与“在本模型上可听改善”：

| 优先级 | 行动 | 预期收益 | 成本 | 置信度 / 验收条件 |
|---|---|---|---|---|
| 1 | 在现有重叠录音上离线 A/B：原 crossfade、相关度自适应、phase crossfade；关闭包络混合再对比 | 快速识别块同步功率凹陷与 phase 模糊的贡献 | 低 | 功率机制高；听感中。RMS/包络 PSD 改善且辅音不恶化 |
| 2 | 固定绝对帧索引、f0、feature/retrieval/z 提交版本与 source；记录 maturity | 消除历史数据被不同推理版本反复替换，建立可定位基线 | 中 | 状态一致性高；音质中。提交后逐值不变，漂移曲线与盲听通过 |
| 3 | 先只验证 decoder valid crop：L/R sweep、全段 parity、短 chunk 改相同 padding | 证明接缝是否来自 decoder boundary，并确定真实 lookahead 价格 | 中 | 理论高，本模型需测。max error、边缘误差、全流无丢/重采样 |
| 4 | 同一 provider、同一 shape 做拆图性能 sweep 与 p95 调度余量测试 | 防止采用理论省 FLOPs、实际更慢的方案 | 低至中 | 本地已有反例，必要性高；无持续欠载 |
| 5 | 在质量/延迟可行后做逐层 cached convolution 和分支 delay alignment | 降低 halo 重复计算，稳定小块成本 | 高 | 方法有直接先例，ONNX/Rust 迁移中；每层 parity 与状态重置测试 |
| 6 | voiced-only harmonic phase alignment 或分带自适应增益 | 修复剩余周期信号抵消，减少全带补偿失配 | 中至高 | 中；瞬态、f0 跳变、无声段需单独回退 |
| 7 | 原双向 HuBERT/enc_p 直接改 mask + 普通 KV cache | 可能减少算力和等待 | 高 | 无损置信度低；改变模型行为，暂不作为首选路线 |

预算必须显式写成 `块等待 + feature maturation + enc/flow maturation + decoder 右上下文 + 计算/调度 + 音频 I/O`。flow 的理论右感受野约 24 个 10 ms 帧意味着严格计算需要 240 ms，已经超过整条 200 ms 目标；若只给更短前瞻，必须把它标为实测控制的近似。推荐设计文档同时保留严格 parity 模式与低延迟近似模式，不能通过省略某阶段等待时间让预算看起来达标。

## 复核范围

本次实际执行了联网检索、论文/作者页阅读和源码路径核对，以及上述本地日志阅读；没有运行模型 benchmark、没有采集新的听测结果、没有修改主引擎或 Applio。RVC 使用固定 tag，其余 `main/master` 链接代表调研时可读版本，后续实现前应固定 commit。2026 年工作仍应按预印本或作者项目声明的证据级别理解，公开 demo 和总体 MOS 不构成“无任何拼接感”的证明。
