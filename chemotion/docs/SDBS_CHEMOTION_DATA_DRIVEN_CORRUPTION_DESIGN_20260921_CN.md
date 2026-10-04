# SDBS 与 Chemotion 启发的模态扰动设计

日期：2026-09-21  
用途：为 learned router 构造更接近真实采集过程的 source corruption。  
当前状态：已完成 1:1:1 smoke/基线 pilot；medium-dominant 2K pilot 已在 GPU3 完成。

## 1. 先说结论

真实数据中的变化不是“每个 token 独立加一点白噪声”这么简单，而是以下几类因素叠加：

1. 峰提取方式变化：一个 multiplet 可能被拆成多个 singleton，也可能漏峰或多出杂峰。
2. 采集协议变化：MS 的离子源、能量和峰列表定义会改变；IR 的表示可能是 transmittance 或
   absorbance，覆盖范围也可能不同。
3. 坐标与基线变化：NMR 有 reference drift，IR 有横轴 warp、baseline、broadening 和区域
   gain。
4. 背景和动态范围变化：MS 会出现大量低强度背景峰，IR 的幅度范围和光滑程度会变化。
5. tokenizer/预处理契约变化：真实输入的 token 数、patch 可见性、数值尺度不一定和模拟 source
   一致。

因此下一版扰动不应该把四种模态都压成同一个随机 mask 概率，也不应该把 Chemotion 的原始峰数
直接平均进 SDBS。建议采用：

- **SDBS 同分子配对 residual**：负责决定扰动的概率和幅度；
- **Chemotion 的处理契约和边际分布**：负责补充单位、覆盖范围、峰提取和格式变化；
- **四模态始终存在且四模态都受扰**：避免 router 学到模态 availability shortcut；
- **同一分子内变换**：不做跨分子置换，不把别的分子的谱拼进当前样本；
- **不使用 PMGFA、target 结构标签或固定 C bias**：让 action reward 自己决定是否偏向 C。

这里的“平等”是四个模态都经过扰动、每个模态都有机会出现轻/中/重变化；并不是把每个模态删除
相同数量的 token。因为四种模态的物理测量误差不同，应该按各自观测到的 residual 分布采样。

## 2. 数据来源和可用边界

### 2.1 SDBS：定量估计同分子变化

SDBS 审计只使用 train/validation 输入，不读取 test 结构标签：

- source rows：794,403；
- SDBS train/validation target rows：2,669；
- exact same-molecule pairs：1,059；
- 四种模态均有 1,059 对可比较记录。

同分子配对可以回答“source 到真实谱之间发生了什么”，因此用于拟合 missing、extra、shift、
split 和背景峰数量的经验分布。它不能被写成严格 source-only calibration；但没有使用 SDBS
test 的正确结构或 Top-1 标签。

### 2.2 Chemotion：补充真实表示契约，不当作 residual 真值

Chemotion 当前可用的处理产物 manifest 为：

```text
/hpc2hdd/home/aimslab/ChengtangZhan/Dataset/spectra_tokenizer_project/data/transformer/chemotion_payload_multimodal_simulation_matched_v3_20260911/opennmt/T_1H_13C_IR_MS/manifest.json
```

manifest 统计：

| 项目 | 数量 |
|---|---:|
| manifest records | 1,995 |
| reaction_zips | 1,793 |
| sample_zips | 202 |
| H 原始 observation rows | 43,035 |
| C 原始 observation rows | 56,514 |
| IR 原始 observation rows | 8,938 |
| MS 原始 observation rows | 11,090 |
| incomplete/invalid record events | 4,180 |
| missing/invalid H/C/IR/MS | 1,114 / 1,532 / 2,535 / 3,426 |

Chemotion 不是与 SDBS 相同的原始采集协议，也不是同一分子的 source/real 配对。它已经经过
observation 选择、单位转换、过滤和 tokenizer 统一。因此它适合回答“另一套真实数据处理链会
产生哪些输入契约变化”，不适合直接估计某个 source 峰的真实 missing probability。

## 3. SDBS 配对审计得到的扰动分布

下面的 recall、precision 和 extra 均在同分子配对上计算：

- `source recall = matched / source peaks`；
- `real precision = matched / real peaks`；
- `missing fraction = 1 - source recall`；
- `extra/source = (real peaks - matched) / source peaks`。

### 3.1 HNMR：峰拆分是主效应

| 指标 | 数值 |
|---|---:|
| source peak count mean | 5.07 |
| real matched-visible peak count mean | 6.96 |
| matched peak count mean | 3.73 |
| source recall | 0.7465 |
| real precision | 0.6180 |
| missing fraction | 0.2535 |
| extra peaks / source peak | 0.6435 |
| local shift error | 0.0388 |
| raw real peak count mean | 25.98 |
| singleton split factor | 4.36 |

含义：HNMR 的真实差异不是单纯把 source 峰随机删掉。multiplet 被拆成多个紧邻 singleton，
同时有约四分之一的 source 峰没有被可靠观察到，并伴随额外峰和小的 reference drift。

### 3.2 CNMR：仍然要扰动，但 residual 明显更小

| 指标 | 数值 |
|---|---:|
| source peak count mean | 8.19 |
| real peak count mean | 7.75 |
| matched peak count mean | 7.03 |
| source recall | 0.8977 |
| real precision | 0.9192 |
| missing fraction | 0.1023 |
| extra peaks / source peak | 0.0820 |
| local shift error | 0.249 ppm（审计坐标） |
| global shift median | 约 -0.062 ppm |

含义：CNMR 不能被固定成完全干净，但它应主要受到 reference shift、局部 jitter、少量漏峰和
少量溶剂/杂峰影响。若把 C 也按 H/MS 的强度删除，得到的不是 SDBS，而是人为改变了真实的
可靠性排序。

### 3.3 MS：相关峰保留，但背景峰长尾很重

| 指标 | 数值 |
|---|---:|
| source peak count mean | 18.82 |
| real peak count mean | 91.78 |
| matched peak count mean | 17.44 |
| source recall | 0.9301 |
| real precision | 0.1996 |
| missing fraction | 0.0699 |
| extra peaks / source peak | 4.90 |
| real peak count q05 / median / q95 | 45 / 82 / 175 |

含义：MS 的主要变化不是删掉分子相关 fragment，而是多出大量 protocol/background peaks。当前
source 是多通道 ESI-MS/MS，而 SDBS real 是单路 EI 75 eV；两者不能用同一套简单 mask 解释。

### 3.4 IR：谱形更平滑、动态范围更大

| 指标 | mean | median |
|---|---:|---:|
| real std / simulated std | 2.385 | 2.075 |
| real roughness / simulated roughness | 0.467 | 0.450 |
| best-orientation shape correlation | 0.194 | 0.163 |

IR 的形状相关性很低，但不能据此加入大量独立白噪声。更符合观测的变化是 smooth axis warp、
broadening、baseline、regional gain、幅度缩放和少量宽背景带。

## 4. Chemotion 能补充什么

对 1,995 条最终 `src.txt` 的 model-visible token 统计：

| 模态 | 可确认的统计 | 解释 |
|---|---|---|
| HNMR peak count | mean 26.48；median 17；q95 65 | observation 选择和 singleton 表示导致密度变大 |
| CNMR token count | mean 112.16；median 100；不少记录达到256-token cap | 受统一 token contract 和重复/密集峰影响，不能直接等同 SDBS 峰数 |
| IR | 固定400点；400–4000 cm^-1；值域0–100；median value 34 | 已统一坐标和强度 contract |
| MS peak-pair count | mean 38.21；median 34；q95 88.3 | 单路 peak list |
| MS m/z | median 226.2 | 与 SDBS EI 的低 m/z 分布不同 |
| MS intensity | median 6.9；q95 65.6 | 低强度背景明显 |

Chemotion IR 的原始表示也不统一：

| 输入类型 | 行数 | 处理方式 |
|---|---:|---|
| TRANSMITTANCE | 8,891 | fraction 用 `1 - T`，percent 用 `100 - T` |
| ABSORBANCE | 24 | 保留 native absorbance |
| ARBITRARY UNITS | 18 | 排除主数据集 |
| UNKNOWN | 2 | 排除主数据集 |
| REFLECTANCE | 3 | 排除主数据集 |

因此 Chemotion 最值得借鉴的不是“把 C 加到 100 个 token”或“把 MS 峰数改成 38”，而是以下
真实输入变化：

- 同一记录可能有多条 observation，需要选择一条合法、信息量足够的谱；
- IR 的 transmittance/absorbance 单位和幅值尺度不统一；
- 部分记录覆盖范围不足或无效，需要做边界处理；
- 预处理后的数据仍然可能出现 OOV、长度截断和密集重复峰。

## 5. 推荐的四模态扰动算法

### 5.1 总体采样规则

每个 source molecule 生成一个 episode，Formula 保持不变，四个光谱都保留且都至少接受一种
扰动。每个模态独立采样 residual profile，不再使用同一个 shared severity：

```text
z_m ~ Bootstrap(P_m)
```

其中 `m` 为 H、C、MS、IR，`P_m` 是该模态在 SDBS 同分子配对审计中的 residual profile
集合。一次 bootstrap 应抽取一条完整 profile，例如 H 同时取得该记录的 missing fraction、
extra/source、global shift、local error 和 split factor；不要分别从五个边际分布独立抽值，否则
可能合成真实数据中从未出现过的极端组合。

为保证四种模态在训练中公平暴露，可先把每个 `P_m` 按该模态内部的经验严重度排序，分成
light/medium/heavy 三档，再对四个模态使用相同的档位采样概率（例如各约 1/3）。档内仍直接
bootstrap 真实 profile，而不是给四种模态设置相同的删除率。建议的经验参照点为：

| 模态与 residual | light（q25） | typical（q50） | heavy（q75；括号内为q95） |
|---|---:|---:|---:|
| H missing fraction | 0.000 | 0.250 | 0.400（0.603） |
| H extra/source | 0.174 | 0.500 | 1.000（1.800） |
| H local shift / ppm | 0.0178 | 0.0346 | 0.0560（0.0867） |
| H singleton split factor | 2.125 | 3.200 | 5.333（11.507） |
| C missing fraction | 0.000 | 0.000 | 0.167（0.429） |
| C extra/source | 0.000 | 0.000 | 0.143（0.333） |
| C local shift / ppm | 0.097 | 0.179 | 0.350（0.706） |
| MS missing fraction | 0.000 | 0.038 | 0.111（0.250） |
| MS extra/source | 2.811 | 3.962 | 5.667（11.436） |
| IR std ratio real/sim | 1.604 | 2.075 | 2.776（4.763） |
| IR roughness ratio real/sim | 0.332 | 0.450 | 0.577（0.817） |

表中的 quantile 是验收参照，不是要求把参数固定为这几个离散值。特别是 C 的 light/typical
profile 即使没有漏峰和杂峰，仍会通过 global/local shift 受到扰动，因此不会出现人为 clean C。

所有随机操作都按有效 token、peak group、peak pair 或 IR patch 的数量进行 Bernoulli/Poisson
采样，而不是每条样本固定删除相同数量：

```text
drop_count_m ~ Binomial(valid_count_m, q_m(z_m))
extra_count_m ~ Poisson(lambda_m(z_m) * valid_count_m)
```

这样 H/MS/IR 原本更长时会自然产生更多受扰位置，但不会因为长度长而被额外施加一个固定的
模态偏置。

### 5.2 HNMR

建议顺序如下：

1. **multiplet split**：把一个 group 拆成 `1..K` 个紧邻 singleton，`K` 从 SDBS 配对得到的
   split-factor 分布采样；Chemotion 只用于核对最终长度和 tokenizer cap；
2. **source peak missing**：以约 0.25 的中心概率按 group 独立删除，light/medium/heavy 用
   配对审计的 residual quantile；
3. **reference drift + local jitter**：先对整条谱加 global drift，再对保留峰加小幅 local
   shift；
4. **extra peak**：按 source peak 数采样少量到中等数量的额外峰，位置来自局部邻域、溶剂区和
   低强度背景，而不是跨分子复制峰；
5. **token budget**：允许 singleton 增长，但按项目 tokenizer 的合法上限截断，截断规则按峰
   强度/位置保留，不随机截掉所有尾部。

### 5.3 CNMR

CNMR 也必须被扰动，但使用 SDBS 估计的较小 residual：

1. 按约 0.10 中心概率独立 mask 有效 peak group；
2. 采样 global reference shift，叠加小幅 local jitter；
3. 按约 0.08 倍 source 峰数采样少量 solvent/impurity peaks；
4. 对局部拥挤峰允许合并或轻微 broadening；
5. 不强行把 CNMR 峰数变成 Chemotion 的 100-token 中位数，因为那是不同的 processed contract。

这仍然是“有噪声的 C”，不是 clean-C rule；C 最终是否被 router 选择，必须由四个 action 的
同一 source reward 比较决定。

### 5.4 MSMS

建议把 MS 的扰动拆成“分子相关峰”和“采集背景峰”两部分：

1. 保留约 0.93 的 source-related peak pairs，并按 severity 采样少量漏峰；
2. 对 m/z 加 protocol-specific shift，避免把 ESI-MS/MS 的 header/token 当成 EI 峰；
3. 对 intensity 使用 log-domain scale、低强度压缩和少量随机响应变化；
4. 采样额外背景峰，`extra_count/source_count` 从 SDBS 的长尾分布（中心约4.9）抽样；
5. 额外峰的 m/z 优先覆盖低质量区，并使用长尾强度分布；
6. 如果加入 Chemotion profile，作为独立 `chemotion_contract` episode，不能和 SDBS EI profile
   取平均后生成一个不存在的中间协议。

### 5.5 IR

IR 使用连续信号扰动，避免独立白噪声主导：

1. 对 400–4000 cm^-1 坐标施加低频 smooth warp；
2. 使用随机卷积核模拟 broadening，核宽从真实谱的宽峰分布采样；
3. 加二阶以内 baseline、slope 和 curvature；
4. 按低频区域分段施加 gain/scale，匹配 real/sim std ratio；
5. 以低概率加入宽背景 band 和少量 spike；
6. 若模拟覆盖不足，只对边界数值做合理 extrapolation；不要把 patch 设成不可见，除非 target
   loader 也真的使用同一 patch-mask 契约；
7. 重新计算 mean/std/roughness，使模型看到的 tensor 统计与真实输入契约一致。

### 5.6 Chemotion contract branch

Chemotion 可作为小比例、独立标记的 contract branch，而不是 residual branch。首轮 smoke 可在
source episode 中加入约 10–20% 的 contract stress rows；该比例只是起始超参数，应只通过
source validation 和输入统计验收，不按 target test accuracy 调整：

- H：多 observation 选择后再做 singleton density/长度变化；
- C：模拟重复峰、密集 token 和合法截断，但不直接复制 Chemotion 的分子峰列表；
- IR：在 fraction/percent/absorbance 三种合法尺度中采样，再统一回项目 0–100 contract；
- MS：使用单路 peak list、低强度背景和不同 m/z 边际，但不混用 SDBS EI 的参数；
- 所有 rows 仍然保持当前 source 分子对应关系，不引入 target 结构。

该 branch 的作用是测试 router 对预处理契约变化的稳健性，不是告诉 router “Chemotion 里的某个
模态一定更差”。

## 6. 为什么不采用几种看似简单的方案

### 6.1 四个模态相同的 mask 概率

相同概率不等于公平。H 的真实问题是 split/missing/extra，MS 的真实问题是背景长尾，IR 的真实
问题是 smooth shape，C 的真实问题主要是 shift。统一 mask 会制造错误的 action reward。

### 6.2 把 Chemotion 与 SDBS 统计直接平均

Chemotion 的 C token 中位数约100，而 SDBS 配对 C 峰数约7；Chemotion 的 MS m/z 中位数约226，
SDBS EI 约82。平均后得到的分布既不是 SDBS，也不是 Chemotion。

### 6.3 跨分子置换

跨分子置换会让 router 学到“内容明显不一致”的 shortcut，而不是学习真实采集误差。若未来
专门研究 negative transfer，应单独作为 counterfactual 对照，不应混进主 acquisition corruption。

### 6.4 只保留 clean C 或给 C 加 reward bonus

这会让结果看起来稳定选 C，但不能证明 router 从 sim-to-real gap 学到了 C 的可靠性。当前正式
版本不采用 C guard、C bonus、PMGFA threshold 或 target-label rule。

## 7. 训练前的验收指标

新 corruption 在启动大规模训练前，先对 model-visible tensor 做 smoke audit：

| 层级 | 必查指标 |
|---|---|
| HNMR | singleton split factor、missing fraction、extra/source、shift quantile、有效 token 数 |
| CNMR | recall、precision、global/local shift、extra/source、有效 token 数 |
| MSMS | related-peak recall、extra/source、m/z 分位数、强度分位数、有效 pair 数 |
| IR | coverage、std ratio、roughness ratio、warp/broadening、patch visibility |
| Router | action reward 分布、C vs Full 胜率、entropy、15 action structural coverage |

验收时必须同时看 raw 输入和 encoder 真正看到的 tensor；v1 IR 的失败说明原始谱统计接近并不
代表 router feature 接口接近。

建议的停止条件：

1. 任一模态的 post-corruption 统计落在 SDBS/Chemotion 目标区间之外，先修契约，不跑正式训练；
2. 80% 以上 rows 的 15 个 action reward 全部无效或全部相同，说明破坏过强；
3. source validation 只剩一个 action 且 reward margin 异常大，先检查是否出现 availability 或
   tokenizer shortcut；
4. 只有通过上述检查后，才比较 learned router 与 fixed C/Full。

## 8. 建议的实现顺序

1. 从当前 `real_acquisition_paired_v5` 复制一份 `v6_profiled`，保留旧版本可复现性；
2. 增加 SDBS paired residual profile 文件，按 train/validation molecule 拟合 quantiles；
3. 把 shared severity 改成每模态独立的 profile sampling；
4. 先实现 H/C/MS/IR 的 native residual branch，不加入 Chemotion branch；
5. 做 128–256 条 smoke，检查 raw 和 model-visible metrics；
6. 再加入 10–20% 独立 Chemotion contract branch，重新 smoke；
7. 只有 smoke 通过后，启动 2K source router；
8. 用 source validation reward、regret、action coverage 和 target frozen-teacher route-only 一起
   决定是否替换现有 checkpoint。

## 9. 最终定位

本方案的目标不是预先规定“router 必须选 C”，而是让训练数据中出现真实的：

- H 的拆峰、漏峰、杂峰和漂移；
- C 的小幅但非零缺陷；
- MS 的相关峰保留与背景长尾；
- IR 的连续谱形、基线、轴和动态范围变化；
- Chemotion 所体现的 observation/单位/coverage/tokenization 契约变化。

如果在这些同分子、无 PMGFA、无 target label 的扰动下，source reward 仍然不支持 C，那么应如实
记录为“source corruption 与 target utility 仍不一致”，而不是继续加大随机噪声或把 C 结果硬编码
进 router。

## 10. 三档训练采样比例

`weak/medium/strong` 是 episode 的共同 curriculum 标签，但不是四个模态共用的物理噪声强度。
初始 smoke 使用 `1:1:1`，用途仅是检查三档实现、reward 覆盖率和训练通路，不能作为最终数据
分布。由于 strong 档仍有较高的 all-actions-nonpositive 比例，而 medium 档更接近真实采集
residual，正式 2K pilot 改用 medium-dominant 配置：

```text
source episode 生成：weak : medium : strong = 1 : 2 : 1
uniform router：在上述 1:2:1 数据集上按自然比例训练
curriculum 起始：2 : 6 : 1
curriculum 结束：1 : 6 : 2
```

因此 medium 在 curriculum 中始终约占三分之二；训练早期保留较多 weak 样本建立稳定的
reward 排序，训练后期增加 strong 样本检验困难扰动下的泛化，但不让 strong 的无效 reward
主导梯度。验证集固定按 `1:2:1` 的数据集分层抽取，不随 epoch 改变。

这组比例不是对 CNMR 的手工奖励或保底规则。router 仍然只根据 15 个 action 的 source reward
学习，真实数据推理时也不调用 PMGFA 或 fixed-C fallback。

## 11. 审计与实现位置

```text
SDBS paired audit:
/hpc2hdd/home/aimslab/ChengtangZhan/ttt/paper_multitask_ms_snapshot_20260914/runs/paired_real_sim_corruption_audit_sdbs_trainval_20260920.json

Chemotion manifest:
/hpc2hdd/home/aimslab/ChengtangZhan/Dataset/spectra_tokenizer_project/data/transformer/chemotion_payload_multimodal_simulation_matched_v3_20260911/opennmt/T_1H_13C_IR_MS/manifest.json

Chemotion model-visible input:
/hpc2hdd/home/aimslab/ChengtangZhan/Dataset/spectra_tokenizer_project/data/transformer/chemotion_payload_multimodal_simulation_matched_v3_20260911/opennmt/T_1H_13C_IR_MS/data/src.txt

Current corruption implementation:
/hpc2hdd/home/aimslab/ChengtangZhan/ttt/paper_multitask_ms_snapshot_20260914/code/tools/build_modality_router_dataset.py
```

## 12. Medium-dominant 2K pilot 结果

运行目录：

```text
/hpc2hdd/home/aimslab/ChengtangZhan/ttt/paper_multitask_ms_snapshot_20260914/runs/router_source_reference_free_real_acquisition_profiled_v6_2048_beam1_gpu3_20260921_medium2x
```

source 数据集实际分层为 `weak=512`、`medium=1024`、`strong=512`，即严格 `1:2:1`。分层
reward 审计如下：

| tier | all-actions nonpositive | fixed-C reward | fixed-Full reward |
|---|---:|---:|---:|
| weak | 28.7% | 0.759 | 0.031 |
| medium | 51.3% | 0.499 | -0.058 |
| strong | 67.6% | 0.301 | -0.095 |

两种训练器在固定 source validation（405 rows）上的结果相同：oracle action accuracy 为
65.2%，learned reward 为 0.492，mean regret 为 0.132；两者均选择 CNMR 405/405，未加入
PMGFA、CNMR bonus 或 fallback。

| router | SDBS 1K Top-1 | invalid Top-1 | action 选择频率 |
|---|---:|---:|---|
| uniform (`1:2:1`) | 465/1000 = 46.5% | 21 | CNMR 994，HNMR 6 |
| curriculum (`2:6:1 -> 1:6:2`) | 455/1000 = 45.5% | 21 | CNMR 968，HNMR 27，MSMS 3，IR 2 |

这次 pilot 的结论是：medium-dominant 采样降低了 strong 样本对训练的支配，但 curriculum
尚未改善真实 SDBS Top-1，反而比 uniform 低 1.0 个百分点。两者都自然收敛到以 CNMR 为主的
策略，因此后续若要扩大到全量数据，应优先检查 source features 对非 CNMR action 的可分性，
而不是继续单纯调整 weak/medium/strong 比例。

## 13. 纯 easy/medium router 对照

为区分“某一难度本身是否可学”与“混合难度采样是否有效”，从同一个 2K 数据集直接抽取
单独的 weak（这里称 easy）和 medium 子集，分别训练 100 epoch。两次训练都使用相同的
source-reference-free 特征、soft-KL reward loss、beam=1，并且真实数据评估只调用 learned
router；没有 PMGFA、CNMR bonus 或 fallback。

| 训练子集 | source rows（train/val） | source learned reward | source oracle accuracy | SDBS 1K Top-1 | 选择频率 |
|---|---:|---:|---:|---:|---|
| easy/weak | 512（424/88） | 0.718 | 62.5% | 463/1000 = 46.3% | CNMR 958，CNMR+IR 37，其余5 |
| medium | 1024（810/214） | 0.475 | 67.3% | 455/1000 = 45.5% | CNMR 955，HNMR 36，其余9 |

纯 easy 的 source reward 明显更高，但真实 SDBS Top-1 并没有超过 mixed uniform 的 46.5%；
这说明 source 难度越低不等于 sim-to-real 映射越好。纯 medium 的 oracle action accuracy
更高，说明 action 排序监督更有区分度，但它仍主要学习选择 CNMR，真实 Top-1 与三阶段之前的
medium-dominant curriculum 相同（45.5%）。

## 14. 下一轮推荐的三阶段 curriculum

应该保留课程学习，但逐步增加的是 **medium 的占比**，而不是大幅增加 strong。下一轮建议：

```text
阶段 1：weak : medium : strong = 4 : 5 : 1  （40.0% : 50.0% : 10.0%）
阶段 2：weak : medium : strong = 3 : 6 : 1  （30.0% : 60.0% : 10.0%）
阶段 3：weak : medium : strong = 2 : 6 : 1  （22.2% : 66.7% : 11.1%）
```

阶段之间使用 piecewise-linear interpolation；验证集固定为数据集自然比例。这样训练难度会
逐步提高：weak 样本减少、medium 样本增加，strong 只从 10% 小幅增加到约 11%，不会让
all-actions-nonpositive 的 strong rows 主导梯度。

之前的 `2:6:1 -> 1:6:2` 并不是一个合适的主 curriculum，因为 medium 保持不变而 strong
翻倍；真实 SDBS 上观察到的 Top-1 下降，正是由错误的非-C 偏离增加造成的。上述三阶段配置
是下一轮待验证方案，不把它当作已经测得的最优比例。
