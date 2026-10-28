# Mimic 固定 RVC v2 的 frozen streams / overlap-save 实施设计

日期：2026-10-27。范围：设计与只读验证；不修改现有 engine、模型、Applio 或设备配置。

## 1. 结论与可保证的边界

**先冻结唯一的 latent / pitch 流，再对确定性解码器做 overlap-save，可以消除“同一时刻反复渲染后拿不同波形混合”的机制；它不能保证冻结流与离线全上下文 RVC 相同，也不能直接保证 ≤200 ms。**

这里的“不重复渲染”指每个输出采样只提交一次。普通 overlap-save 仍重复计算左右 halo 中的卷积和被丢弃的音频；要做到每层激活只计算一次，需要另行导出带状态的卷积图。不能把两个目标混为一谈。

另外，“输入相同”必须包含该输出的**全部依赖邻域**：相同的 z、f0、source noise、phase 和 padding/采样网格。仅中心帧相同而右侧 halo 不足，仍会出现边界误差。冻结帧消除了同一帧的版本冲突，却可能留下每批冻结帧之间的特征或 z 跳变；无交叉淡化不等于听感必然无接缝。

建议分两条验收线：

1. **数值正确性基线**：decoder 大 halo，通过整段/分块 `max_abs_error < 1e-4`，延迟先不限；隔离窗口漂移与 DSP 错误。
2. **实时近似候选**：较短 feature/encflow/decoder 前瞻，冻结后只发布一次；实测质量、deadline 和物理端到端延迟，合格才取代现有路径。

## 2. 本地证据：已有实验与本次验证分开

以下历史日志是仓库已有结果，本次没有重跑其 GPU 实验，不应当作新的性能认证：

| 本地来源 | 已记录的观察 | 能支持的结论 / 限制 |
|---|---|---|
| rf_chunking.log | decoder 保留6帧、L=12：R=4/6/8/12帧的 SNR 为29.3/42.0/61.2/63.9 dB | R=8 很接近，但 SNR 不等价于最大误差 `<1e-4`；必须单独验证 |
| 同上 | enc_p `m_p` 对全上下文的 cosine，距右端1/4/8/12帧为 .768/.906/.948/.962 | 冻结 enc_p 输出仍有可观的上下文近似误差 |
| 同上 | attention layer 0 的 ±50帧质量占比仅 .09 | `window_size=10` 是相对位置嵌入范围，并非局部 attention mask |
| sim_frozen.log | `rt_la3_le2_r8` 相对全上下文 mel-L1=4.34 dB，分块/整段 decoder SNR=54.9 dB | 解码器边界接近并未使全链路等价；mel-L1 不是听感评分 |
| export_dec_only.log | split graphs 与 synth_det，max误差 `9.03e-7` | 一组新长度输入的图拆分 parity 成立；不是所有分块配置都已通过 |
| bench_dml_split.log | 拆分设计约63–65 ms/块；120/60/40 ms块对应RTF .53/1.07/1.56 | 缩短块会增加重复全窗口 attention 的单位时间成本；60/40 ms当前结果不可实时 |
| 同上 | monolithic synth_det 118/88中位17 ms，split decoder 26帧约40 ms | 图形状、provider fallback和融合比理论 FLOPs 更重要 |

本次另提供 CPU-only validate_decoder_overlap.py，结果保存为 decoder-overlap-results.json。它比较相同合成 z/f0/source 的整段和分块解码，测试不同 halo，只验证 decoder 契约，不打开设备、不运行 GPU、不改模型。最终选择 halo 应以这个测试及真实语音、多 seed/provider 的扩展测试共同决定。

本次实际结果（96帧固定输入，B=6，含清浊切换，seed=20261027）：

| 左/右halo（10ms帧） | max绝对误差 | SNR dB | `<1e-4` |
|---|---:|---:|---|
| 12 / 4 | 7.85e-2 | 27.99 | 未通过 |
| 12 / 8 | 4.42e-4 | 77.78 | 未通过 |
| 12 / 10 | 1.38e-5 | 96.85 | 通过 |
| 12 / 12 | 1.38e-5 | 96.85 | 通过 |
| 16 / 16 | 1.53e-5 | 96.85 | 通过 |

这说明“8帧已足够”的判断至少不满足本次最大误差门槛；10帧只对这一测试通过，尚不能据此把更大的理论感受野裁掉作为所有输入保证。大halo仍有约1e-5误差，符合分段phase累计与FP32运算顺序带来的数值差异，需用统一外部激励及长流测试进一步定位。

注意历史 `bench_dml_split.py` 已有过后续修改，和日志展示的配置不完全相同。使用日志中的数字作为历史记录，不将当前脚本的未执行配置误当新结果。

## 3. 时间网格和数据所有权

所有区间使用半开 `[start,end)`，`end` 是下一帧；所有时间戳来自输入采样计数，不使用线程唤醒时刻作为音频时间。

| 网格 | 单位 | 绝对索引 |
|---|---|---|
| capture | 48 kHz sample | `n48` |
| contentvec input | 16 kHz sample | `n16`，与48k resampler相位连续 |
| contentvec | 50 fps | `j50`，原点固定，步长320个16k采样 |
| pitch / latent | 100 fps | `t100`，步长160个16k采样；feature的`j`复制给`2j,2j+1` |
| decoder output | 40 kHz sample | `n40=400*t100+r` |
| render | 48 kHz sample | 连续40→48k resampler，400输入严格对应480输出，保留滤波状态 |

从本地 contentvec 配置的 kernel `[10,3,3,3,3,2,2]`、stride `[5,2,2,2,2,2,2]` 得出 CNN 步长320、感受野400采样（25 ms）：

`jump_0=1, rf_0=1; rf_next=rf+(kernel-1)*jump; jump_next=jump*stride`。

若无 CNN padding，frame j 的原始支持区间是 `[320j,320j+400)`，因此不能把输入长度整除320直接当作已有完整feature数：`F(N)=max(0,1+floor((N-400)/320))`。Transformer positional convolution还引入窗口依赖。`feat_extract_norm="group"` 也需检查实际归一化轴；若跨时间归一化，则 CNN 中间激活也不能按有限感受野直接复用。

每个 cache 都保存 `origin`, `committed_end`, `epoch`，只允许在末尾追加。禁止修改 `[origin,committed_end)`。每条流保留有限历史，按最慢消费者的水位回收，不能让 cache 随运行时间无限增长。

每次推理使用固定长度与固定原点的窗口。窗口**起点**必须落在绝对320采样网格；窗口长度可以是18880（59×320）或19040（59×320+160）等固定值，但必须按真实输出长度和映射导出索引。不能声称长度只有320整数倍一种合法选择，也不能把末帧复制到尚未成熟的新时间。处理步长 B 建议为20 ms的整数倍；100 fps层可以内部10 ms步进，但不能使contentvec网格每块交替移动半帧。

## 4. 阶段划分、成熟条件与上下文

### 4.1 Feature、retrieval、pitch

**contentvec 的成熟是政策定义，不是有限 L 后数学上不再变化。**

固定 `W_cv=18880` 个16k采样作为首个候选，`L_feat=1/3/7` 个50 fps未来完整帧（20/60/140 ms）作消融。每个20或40 ms tick在同样的完整帧规则下选新成熟特征；成熟后将 raw feature 与 retrieval feature 一起冻结，retrieval只计算新帧。查询 top-8 保持原索引、权重和 index_rate；修改检索参数采用后续生效帧，不回写历史。

RMVPE继续用固定窗口。其 centered mel、网络padding与右边裁剪纳入可用性测试，不能把100 fps输出率当作10 ms固有延迟。先使用历史实验的约4960采样窗口/候选 `L_pitch=20–30 ms`，记录精确时间戳；窗口越短并不保证推理越快。

pitch的raw f0先冻结；pitch shift和coarse量化在统一的参数epoch中冻结；protect需要raw feature、retrieval feature和voicing全部成熟才提交phone。不可先冻结retrieval blend后再改变历史protect权重。

### 4.2 enc_p + flow 两种模式

**推荐先用 combined graph，以新成熟 z 的只发布一次换取可控近似；不要宣传它是严格的全流 overlap-save。**

模式A（实用近似）：enc_p+flow在固定 `W_ef=118` 帧窗口上运行。每次窗口末端为 `E`，选定 `L_ef`，仅提交尚未提交且属于 `[E-L_ef-B,E-L_ef)` 的 z。保留的所有 z 来自本次窗口的同一模型计算；历史缓存 z 不再替换。左历史约 `W_ef-B-L_ef` 帧，右侧 `L_ef` 是 **combined graph 的近似前瞻**，包括 flow 右边界影响，不能再把它称为严格 ±24帧flow。

`pitchf` 在标准模型的enc_p中不参与计算（用的是coarse pitch）；可在wrapper中用统一接口携带pitchf，但 ONNX optimizer 可能消除未使用输入。运行时按实际图签名建feed，不能强行喂一个不存在的输入。speaker id应作为输入或在每声线图中固定并写metadata。

模式B（数值基线）：enc_p固定窗口输出 `m_p,logs_p`，冻结 `z_p`，随后独立flow对冻结的z_p做overlap-save。四个coupling块，每块3层kernel5/dilation1，单块半径`3*(5-1)/2=6`帧，串联保守半径24帧（240 ms）。左/右均至少24帧，再按图依赖和误差测试确认。严格flow单独的右前瞻已超200 ms预算。模式B隔离enc_p误差，但不是本次低延迟默认项。

| 阶段 | 左上下文 / 状态 | 右上下文 | 保证 |
|---|---|---|---|
| capture→16k | 连续resampler历史/相位 | FIR自身group delay | 对连续滤波契约一致 |
| contentvec | 固定约1.18 s窗口内的历史 | L_feat + CNN完整性边界 | 近似；冻结后不可变 |
| retrieval | 每帧独立 | 0 | 对冻结输入可精确复用 |
| RMVPE/protect | 固定RMVPE窗口、冻结voicing | 实测L_pitch与mel边界 | 近似；统一成熟 |
| encflow A | 固定118帧内的历史 | L_ef=0/2/4等候选 | 近似；每个z只提交一次 |
| enc_p→flow B | enc_p固定窗口；flow左24帧 | enc_p lookahead + flow右24帧 | flow相对冻结z_p可精确；enc_p仍近似 |
| decoder | 左12帧是经验起点；严格候选14–16帧 | R_dec=8/12/14/16等候选 | 只有通过parity的halo可称满足误差门槛 |
| 40k→48k | 连续滤波历史/相位 | 小量滤波delay | 不按每块独立zero-pad |

### 4.3 Decoder overlap-save

等待冻结z和f0覆盖到 `end+R_dec`。令 `s=max(0,start-L_dec)`、`e=end+R_dec`，解码 `[s,e)`，只提交 `audio[(start-s)*400..(end-s)*400]`。左右halo都丢弃；下一块同理，采样索引严格相邻，不使用SOLA时间移位。

启动时真实序列起点使用与整段图相同的自然padding；不要在图外为latent填虚构的0帧再当作真实history，否则bias/nonlinearity会使行为不同。静音也推进绝对时间、phase/noise索引；可输出0但不得误重用旧epoch的latent。flush只有在明确结束输入后才允许补尾或裁到真实总长度；实时运行不能把未来未知帧当结束padding。

## 5. ONNX 图接口和可复用的现有工作

本地已有 export_dec_only.py、`encflow.onnx` 和 `dec.onnx`。先复用只读scratch图完成验证，再经现有模型管理流程提升为正式artifact；无需在这份设计中重新写入 models。

| 图 | 输入 | 输出 | 导出要求 |
|---|---|---|---|
| contentvec保持 | `audio:f32[1,N]` | `feats:f32[1,F,768]` | 固定runtime窗口shape；保留原模型语义 |
| encflow | `phone:f32[1,T,768]`, `phone_lengths:i64[1]`, `pitch:i64[1,T]`, `noise:f32[1,192,T]`, 可选`sid` | `z:f32[1,192,T]` | `noise`对应现有`zp_noise`；全长flow后再裁，禁止在flow前skip；标准图不消耗pitchf |
| decoder | `z:f32[1,192,T_dec]`, `f0:f32[1,T_dec]`, `src_noise:f32[1,400*T_dec,1]`, `phase0:f32[]`, 可选`sid` | `audio:f32[1,1,400*T_dec]` | 对应scratch的`pitchf`；去掉图内随机采样；eval模式；phase0单位cycles |
| 可选enc_p基线 | phone、lengths、pitch、noise | `z_p`，可加m/logs调试输出 | 供模式B和误差定位 |
| 可选flow基线 | `z_p`, mask, sid | z | 独立有限RF验证，不假设与模式A冻结z相同 |

每个配置使用固定shape的独立ORT session、预分配binding/内存，线程上不在音频callback中推理。记录provider partition与CPU fallback，禁止用“启用DirectML”替代“全图在GPU高效执行”的验证。图内exp/logs的scale保留现有checkpoint实现的精确常量（本仓库为0.66666），不要随意改成0.666而混入parity误差。

### 转置卷积对齐与可证明的 halo

一维转置卷积输入i对输出有贡献的条件是 `o=i*stride-padding+k*dilation`，`k∈[0,K)`；输出长度是 `(T-1)*stride-2*padding+dilation*(K-1)+output_padding+1`。导出保留原图每级 stride/kernel/pad/output_padding，不能只取same-padding近似。

对目标最终采样区间反向传播整数依赖范围：普通conv按stride/pad/dilation展开；ConvTranspose用上述不等式及ceil/floor求输入index；残差/NSF多分支取并集；3个ResBlock是并联，取最大范围，不能把三支半径相加。NSF source injection的stride/padding也要包括，不能只分析z路径。该依赖检查应从实际ONNX属性得到，而不是拿估算±136 ms硬编码。GN/全局reduction若存在，有限halo论证失效，应单独划分。

只按整latent帧裁切，累积上采样乘积400，因此局部起点s与全局输出偏移必须严格对应`400*s`。检验每一级输出长度和全图40k长度；任何额外偏移写入metadata并由impulse测试固定，不能交给SOLA掩盖。

验证矩阵：

1. 同一模型、provider、dtype、固定完整z/f0/source，整段一次解码作为ref。
2. 分块B=2/4/6/8/12帧；所有边界余数和非整块尾部；起点不只测试0。
3. 左/右halo分别扫描0..16；多seed随机输入、真实语音latent、低f0/high f0、voiced/unvoiced、pitch变化。
4. 断言长度/索引一致；输出max绝对误差`<1e-4`，同时报告SNR、边界局部峰值及phase误差。FP32不要求bitwise相等。
5. CPU通过后再测DirectML/CUDA实际provider；不同shape kernel的数值顺序可能不同。扩展到分钟级流避免phase累计漂移。
6. 独立检查40→48k streaming resampler；禁止每块重新zero-pad造成新的接缝。

## 6. 调度伪代码：水位而不是不断重写滚动窗口

```rust
struct Frozen<T> {
    origin: Frame,       // Absolute index of first retained item.
    committed_end: Frame,
    epoch: u64,
    ring: BoundedRing<T>,
}

fn append_only<T>(s: &mut Frozen<T>, start: Frame, values: &[T]) {
    assert_eq!(start, s.committed_end);
    s.ring.append(values);
    s.committed_end += values.len();
}

fn tick(state: &mut State, capture: &[f32]) -> Result<()> {
    // Worker thread only. Callback only moves samples through SPSC rings.
    state.input16.append(state.down48_16.process(capture));
    let cv_range = state.cv_policy.new_mature_range(state.input16.end());
    if !cv_range.is_empty() {
        let window = state.cv_policy.fixed_aligned_window(cv_range.end);
        let all = state.cv.run(state.input16.slice(window))?;
        let new_raw = extract_absolute_frames(all, window, cv_range);
        let new_retr = state.retrieval.run(&new_raw)?;
        append_only(&mut state.raw50, cv_range.start, &new_raw);
        append_only(&mut state.retr50, cv_range.start, &new_retr);
    }
    state.pitch.commit_new_mature(&state.input16)?;
    // Commit phone only after both feature and pitch/protect are immutable.
    state.commit_phone_and_epoch()?;

    let ef_end = state.phone100.committed_end;
    let z_end = ef_end.saturating_sub(state.cfg.ef_right);
    if z_end >= state.z.committed_end + state.cfg.block100 {
        let end = state.z.committed_end + state.cfg.block100;
        let w_end = end + state.cfg.ef_right;
        let w_start = w_end - state.cfg.ef_window; // Startup is handled separately.
        let noise = keyed_latent_noise(state.epoch, w_start..w_end);
        let z_all = state.encflow.run(state.phone(w_start..w_end), noise)?;
        let start = state.z.committed_end;
        append_only(&mut state.z, start, z_all.at_absolute(start..end));
    }

    while state.z.committed_end >= state.next_audio + state.cfg.block100
                                         + state.cfg.dec_right {
        let a = state.next_audio;
        let b = a + state.cfg.block100;
        let s = a.saturating_sub(state.cfg.dec_left);
        let e = b + state.cfg.dec_right;
        assert!(state.f0.committed_end >= e);
        let src = keyed_source_noise(state.epoch, 400*s..400*e);
        let phase0 = state.phase_at(s); // Prefix over frozen, shifted f0.
        let y = state.decoder.run(state.z.slice(s..e),
                                  state.f0.slice(s..e), src, phase0)?;
        let kept = &y[400*(a-s)..400*(b-s)];
        state.output48.push(state.up40_48.process(kept))?;
        state.next_audio = b; // Each global sample is committed exactly once.
    }
    state.evict_history_before_min_consumer_watermark();
    Ok(())
}
```

需要避免一个隐藏的质量/成本冲突：每批取B帧时，其最早帧比最后帧多看B−1帧未来，因而批边界仍可能产生不均匀质量。若要每帧完全相同的成熟相对位置，必须每帧/20ms单独跑固定窗口或batch多个移位窗口，计算量会增加。历史`sim_frozen.py`的latent按单帧窗口求值，而`bench_dml_split.py`每块只跑一次encflow；两者不是完全相同算法，不能直接把前者质量和后者耗时拼成已实现结果。第一版明确采用每块同窗的模式，单帧canonical窗口作为离线对照。

## 7. 延迟预算：≤200 ms 是需要验证的约束

**串行阶段的前瞻相加，feature与pitch并行取max；不能只写 `块+L_feat+L_dec` 而漏掉encflow。**

以最老样本的保守边界计算：

`D_total ≈ B + max(L_feat, L_pitch) + L_ef + R_dec + D_front/grid + D_compute + D_IO + D_queue`。

这里B、L均已换为ms；`D_front/grid`负责真实CNN完整帧、mel中心/帧原点等未被L定义涵盖的额外时间（必须按时间戳去重，不能重复计CNN delay）。若每个阶段独立等到下一个block tick，额外调度量化也要加到D_queue；仅同一个tick按依赖推进、对齐各阶段水位时才能避免多收几个B。

| 配置（设计候选，非实测承诺） | B | L_feat | L_pitch | L_ef | R_dec | 纯块+前瞻 | 加front/grid 5–25、IO 10–20、compute20的估计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q：较完整decoder | 40 | 60 | 30 | 20 | 120 | 240 | 275–305 ms，超目标 |
| M：历史近似配置 | 60 | 60 | 20 | 20 | 80 | 220 | 255–285 ms；若compute约65则更高，且RTF>1 |
| L：低延迟候选 | 40 | 20 | 20 | 0 | 60 | 120 | 155–185 ms；decoder非严格parity、feature失真较大 |
| L+：稍多语义上下文 | 40 | 40 | 20 | 0 | 60 | 140 | 175–205 ms；预算非常紧 |
| E：严格flow基线 | 40 | 60 | 30 | enc_p20 + flow240 | 120 | 580 | 615–645 ms，供正确性对照 |

上述compute20 ms是**需要实现的优化目标**，不是既有split图性能。当前约63 ms/block结果下，40 ms候选RTF>1，队列无界增长；此时写出一次“低延迟”数值没有意义。必须先使p95计算耗时小于B，最好p95 `<0.5B`，再讨论总延迟。先做provider partition profiling、固定shape/IO binding、去主机往返与decoder融合，然后才评估有状态层级缓存。

块等待的平均可接近B/2，但验收 ≤200 ms应用p95/最坏预算和物理测量，不用平均值包装。输出预填、调度抖动、WASAPI buffer必须单列，不能重复加入已在B里统计的积攒时间。

## 8. 计算量：按阶段和单位时间比较

旧路径每B帧对约118帧做contentvec/encflow，并对 `B+12+6` 左右的decoder片段渲染（按当前120ms布局为30帧）。因此“旧decoder每次整1.18s全算”的假设不符合当前`skip_head`路径。

固定B、相同窗口下，模式A粗略比例：

| 阶段 | 新/旧每块工作量（估算） | 备注 |
|---|---|---|
| contentvec | 约1 | 仍要跑全窗口全局attention；仅冻结输出不节省其计算 |
| retrieval | 约 `(B/2)/F_cv`，B=12/F_cv≈58时约.10 | 只查新增50fps帧；独立帧可缓存 |
| encflow | 约1 | 固定118帧整窗；不是只算新帧 |
| decoder | `(B+L+R)/(B+12+6)` | B=12,L=12,R=12时36/30=1.20；R=8时32/30=1.07 |
| crossfade/SOLA | 0或很小 | 去掉不能补偿attention/GPU图开销 |

若B从120ms减到40ms，contentvec/encflow调用频率×3；即使每块略便宜，每秒总成本仍会变大。准确相对值应以各阶段实测成本加权：`C_new/C_old=(C_cv+C_pitch+C_retr_new+C_ef+C_dec_new)/(C_cv+C_pitch+C_retr_old+C_ef+C_dec_old)`；无法在未知成本权重时给出可信的统一百分比。

后续带状态decoder图能减少halo重算：每层缓存实际中间激活，并用明确的lookahead delay line给非因果层提供未来输入。cache key包括epoch、绝对sample/该层stride相位。缓存任意旧激活而不保证其全部输入成熟，会把window-dependent错误固定下来。它不会消除固定非因果网络需要的真实未来信息。

## 9. 全局注意力与残余误差量化

contentvec及enc_p全局双向attention不能直接套causal KV cache：新输入既改变新query，也改变历史位置的隐藏状态；多层传播使旧K/V失效。给固定权重模型改chunked/causal mask是一项改变推理分布的近似，需要独立质量验证；不能将只改mask称为精确cache。

保持window长度、起点网格、参数epoch固定；成熟帧只发布一次。对每个L、W、B测试：

- feature/m_p/logs_p/z：同一绝对帧跨移位窗口的cosine、相对L2、p10/median/p95，按距左右边界分桶；低能量帧不直接拿cosine解释音质。
- retrieval：top8集合重合率与混合后feature漂移，防止近邻切换放大差异。
- f0：voiced/unvoiced错误率、voiced cents误差，尤其低频和音节边缘。
- audio：相同冻结流的整段decoder vs chunked是纯边界误差；全冻结链路vs全上下文是模型近似误差；这两项分别记录。
- 接缝：边界±5ms的sample jump、导数/高频能量、短时RMS、mel谱差与离边界分布；再测块频4–25Hz附近的包络调制。
- 听测：盲比120/250ms旧路径与候选，固定响度/声线/pitch，包含plosive、低音长元音、耳语、静音切换。cosine/SNR高不能替代听测。

## 10. NSF绝对相位、随机数和实时变调

latent Gaussian keyed为 `(session_seed, voice_epoch, LATENT_STREAM, absolute_frame, channel)`；source Gaussian keyed为 `(session_seed, voice_epoch, SOURCE_STREAM, absolute_40k_sample, harmonic)`。用稳定counter/hash PRNG与Box–Muller，或缓存已冻结噪声；禁止依赖每次调用的数组长度和消费次数。现有 [noise.rs](../engine/src/dsp/noise.rs) 可复用，但多维key组合要防止不同channel/time冲突。

为每个冻结f0帧存相位checkpoint：

`P[t+1] = frac(P[t] + 400*f0_shifted[t]/40000)`。

decoder从frame s开始的`phase0=P[s]`，本图首采样采用 `(r+1)*f0/40000` 的约定，与现有导出保持一致；不能改成r起算而漏一采样。累加用f64并逐帧模1，再转换图所需f32；有多个harmonics时同时保持初始harmonic phase。当前checkpoint harmonic_num=0，其他模型不能沿用这个假设。

f0=0时phase是否保持，遵从现有source的语义；恢复voicing不能随块重置phase。长流定期phase checkpoint避免巨大浮点数；分块/整段测试须统一相位约定与数值路径。

用户调pitch时取 `t_effect=max(next_uncommitted_pitch,next_z_affected)` 的安全边界。epoch事件只修改尚未冻结的后续raw f0变调、coarse pitch和source相位积分；已冻结/入队区间继续原参数，UI显示生效延迟。可对未来f0在20–40ms内平滑，但coarse/continuous f0须来自同一条平滑曲线。切声线/换模型为新epoch，清理模型状态并在音频输出端做一次受控切换，不能把旧voice的z当新voice上下文。

## 11. 风险、回退和落地顺序

**若严格decoder需要>10帧右上下文，诚实选择延迟或近似，不能靠冻结消除物理依赖。**

短前瞻回退可选R=4–8帧，保持L=12–16，在相邻块额外保留2–5ms边缘输出做极短淡化。该方案会故意重复提交候选渲染区域、只混合发布一次；它不再具有严格overlap-save parity，仍要测边界、功率与听感。低f0周期可能10–15ms，2ms淡化不能保证谐波相位问题消失；只在误差已局限边缘时采用。若需要大幅SOLA移动才听起来正常，应回到latent/phase/时间网格定位，而不是继续叠淡化。

1. 保存现有路径和配置；不动Applio、不改系统默认设备、不自动打开CABLE。
2. CPU decoder parity矩阵 → 锁定可信halo；split parity核对speaker/noise/phase/scale。
3. 离线生成冻结cache，分别定位feature、pitch、encflow、decoder误差；同窗按块与canonical逐帧分别比较。
4. 对固定shape的真实split图做GPU profiling；仅在游戏未运行时做重负载测量，记录provider、驱动、shape与p95/max。
5. 先让块计算deadline过关，再用脉冲回环测物理延迟（原独立 rt-proto 原型已删除，源码见 commit b603bd6），随后在主引擎受控集成。
6. 验收：数值基线误差`<1e-4`；近似实时模式单列；持续运行无队列增长、无underrun/overrun；盲听相对旧250ms路径可接受；往返/单程延迟分别标明，目标端到端p95≤200ms。

目前已有证据不足以宣称所有目标同时满足。最有价值的下一步是**严格区分全局窗口近似、卷积边界误差和DirectML图性能**，逐项收敛，而不是直接用更短块替换现有实现。
