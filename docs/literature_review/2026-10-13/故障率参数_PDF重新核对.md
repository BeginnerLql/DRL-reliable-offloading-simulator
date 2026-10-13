# 可靠性文献故障率：重新核对PDF实验设置

> 参数快照说明：本文引用的仓库基础故障率0.001～0.005 s⁻¹及对应算例属于调整前版本。当前配置已改为0.1～0.5 s⁻¹，状态尺度为0.5 s⁻¹，β仍为0.5；历史文献核对和原算例保留。


核查日期：2026-10-13。针对前两轮单副本/执行冗余候选及恢复类中的25篇本地PDF，重新读取实验设置、模型定义、参数表及敏感性图。没有PDF的4篇冗余摘要候选不在这次数值核对范围内。PDF页码为附件物理页，不是期刊连续页码。

## 关键结论

- **TORE-NET**的Table II明确印为10^-3 fault/ms，等于1 s^-1；不能写成10^-3 s^-1。λ0是最高归一化频率下的基础值，实际λi还随频率放大。
- **RTPO**明确印为{0.04,0.06,0.08} s^-1；这是本轮较清楚的秒制故障强度设置。
- **RARE-DTR**调度实验10^-6 faults/μs也等于1 s^-1。**DieHard**给per hour，换算后约10^-9–10^-10 s^-1，故障对象为网络/供电域，不能与CPU软件瞬态故障直接类比。
- 多篇实验所谓failure rate实际上是无量纲概率或失效节点比例。明确标明“推定”的单位不是作者在实验表里写出的单位。
- **RASOUL与QoE论文**虽然有指数故障可靠性模型，实验参数表没有列出基础故障强度赋值，不能凭相关论文自行补齐。

有些PDF的参数表是文字轮廓而非可提取文本。对RASOUL、QoE、ReLIEF、RTPO、UDQF的表格进行了矢量轮廓核读，避免把“文字提取没读到表格”误认为“原文没写”。表格确实没给故障参数的条目，仍明确标记为未报告。

## 逐篇核对

| 文献 | 实验数值 | 单位/参数性质 | 秒制换算或限制 | PDF位置 |
|---|---|---|---|---|


| TORE-NET (`D7D2ZEC2`) | λ0=10^-3 fault/ms | fault/ms，参数表明确；基础故障强度 | 1 s^-1 | PDF p.10 Table II；pp.6–7式(16)–(18) |

| RASOUL (`ETI9KJD8`) | 未在实验参数表列出λ0、ω的数值 | 故障强度单位亦未明确；指数CPU故障强度；实验数值未公布 | 不能数值换算 | PDF p.5式(13)(14)；p.8 Table III（矢量表格核读） |

| HRTO (`WV7TGTXZ`) | 默认执行α∈[0,0.3]、链路β∈[0,0.1]；执行敏感性[0,0.1]至[0,0.5] | 参数段未标/s；结果段称每period失败概率；正文Poisson强度与实验周期概率描述不一致 | 不能唯一换算；若输入指数且Tc用s，则应为s^-1 | PDF p.4式(3)–(5)；p.8设置；p.10 Section V-C4 |

| Reliability-aware task replication (`ZB2BYGHC`) | 没有统一CPU故障强度；拟合LTE/5G接入时延、排队时间 | 不适用；期限内完成概率 | 不能换算CPU λ | PDF pp.5–7，Sections V–VI |

| DCR/DAA (`TRC42RBI`) | 没有统一CPU故障强度；无线服务参数v_nm | v_nm为完成时间分布的服务参数；job截止违约概率 | 不能作为CPU故障λ | PDF pp.4–5，式(2)–(13) |

| FTRATO (`B76XMRDS`) | λ=0.010 | Table I未标时间单位；Poisson MEC服务器故障系数 | 若T用s，则0.010 s^-1（推定，非表格明示） | PDF p.7 Table I；p.3式(16)(17) |

| TMDRL-RARTO (`QIXLEZBS`) | ES γm∈[5%,10%]；UAV γu∈[10%,20%]；敏感性统一5%–30% | 百分比，无时间单位；节点硬件失败概率 | 无法无条件换成s^-1 | PDF p.11 Table 3；p.13 Section 6.5；p.5式(12) |

| DT-MARL (`6GVHFETT`) | 节点失败参数均匀[0,0.1] | 实验段无单位；模型直接取概率乘积；子任务执行期间失败概率 | 不能直接换成s^-1 | PDF p.10实验设置；pp.4–5式(1)–(3) |

| ReLIEF (`YNX4H396`) | Fig.6(b) λ刻度0.0001、0.001、0.01、0.1、0.5；Fig.6(d) λ=0.0001 | 图中未标/s；时间轴和计算模型支持秒制推定；Poisson计算及链路故障强度 | 同数值s^-1仅属量纲推定 | PDF p.5式(7)–(9)；p.9 Fig.6/Table III（图与表核读） |

| DTFR (`3EE7NX8V`) | DEVICE/SITE/LDNS真实故障轨迹；DBN估计条件/联合失败概率 | 概率、停机时间比例；非统一λ；轨迹学习的可用性与联合失效 | 没有统一秒制常数 | PDF pp.5、7–8；Sections 4、6 |

| DieHard (`9M5ZCA4V`) | 网络节点λ∈[0.000022,0.000032]/hour；PDU λ=0.4×10^-6/hour | per hour，正文明确；指数故障强度 | 网络[6.1111e-9,8.8889e-9] s^-1；PDU 1.1111e-10 s^-1 | PDF p.3 Section III-A |

| KPath (`FFUIT44G`) | node issue rate从5%到45% | 概率，正文明确；请求显著变慢/节点问题概率 | 不适用CPU故障强度换算 | PDF p.10 Section VI-A1 |

| RARE-DTR (`KITIKZMY`) | 调度实验λ0=10^-6 faults/microsecond；另考察高至10^-4；检测coverage=0.98 | faults/μs，正文明确；基础瞬态故障强度＋检测覆盖率 | 默认1 s^-1；10^-4/μs对应100 s^-1 | PDF pp.13–15；Section VII-A/B |

| 多目标分散计算冗余 (`RQ9KM5ZT`) | 模型0<ϕj<1，实验Table II未列ϕj的具体赋值 | 无量纲失败概率；具体实验数值未列；IoTD单次失败概率 | 不能换算 | PDF p.5式(1)；p.11 Table II |

| DIST (`BQVHSWSJ`) | 真实VM实验失败比例0%、10%、20% | 百分比；实验称VM完全丢失；节点/VM故障注入比例 | 不适用每秒故障强度 | PDF pp.13–14 Fig.12及分析；p.5式(7) |

| FT-DSAC (`KX9FH3L7`) | 默认ES[0,0.1]、中继[0,0.03]；ES敏感性[0,0.1]→[0,0.2]；中继[0,0.05]→[0,0.3] | 实验段未标时间单位；正文Poisson强度 | 若正文计算/传输时间为s，则数值单位s^-1（推定） | PDF p.5式(6)(7)；p.10设置；pp.11–12敏感性实验 |

| ALC-LSMOEA-KT (`DZ68WBXY`) | base probability=10^-3/epoch；epoch Δ=200ms，另启用shock场景 | 每epoch概率，Table 2明确；离散窗故障概率 | 若独立窗且匹配指数生存率，等效λ=-ln(0.999)/0.2≈0.0050025 s^-1 | PDF p.10 Table 2 |

| 车辆edge-cloud容错策略DDPG (`S3R4ZGMY`) | Edge低[0.00894,0.01553]、高[0.02689,0.08254]；Cloud低[0.00776,0.00799]、高[0.00821,0.00827] | Table 4未标失败参数单位；节点延迟为s；正文指数失败参数 | 同数值s^-1可从秒制执行时间推定，未直接印在表中 | PDF p.25 Table 4；p.18指数失败模型 |

| 图强化学习容错调度 (`ZCQ23XJC`) | Cloud φ=0.02，edge φ=0.05，user φ=0.1 | 无量纲概率；节点失败/超期触发概率 | 不能直接换成s^-1 | PDF pp.7、11–12 Section 5.1 |

| LeDRL (`HY4YPKGL`) | 拓扑node failure rate=0.01、node appearance probability=0.1；故障敏感性图5：执行和传输概率0.1、0.15、0.2、0.25、0.3 | 实验图明确failure probability；默认node failure rate未给时间单位；拓扑失效/任务故障概率；正文另为Poisson模型 | 无法无条件归为s^-1 | PDF p.3式(3)等；p.6设置；pp.7–8 Fig.5 |

| TRO (`XPM57MLT`) | Fig.4(c) UAV failure rate parameter刻度0.01、0.03、0.05、0.07、0.09、0.11；未见完整默认α/β列表 | 图未标/s；正文计算/传输时间为秒；Poisson计算与链路故障强度 | 该扫描数值s^-1为量纲推定，非图轴明示 | PDF pp.5–6式(7)(8)；p.10设置；p.11 Fig.4(c)（矢量轴核读） |

| QoE/Reliability-Aware Scheduling (`KQ4FD48B`) | 模型λ(f)νi，νi∈[0,1]；参数表未列λ(f)数值/函数赋值与γ数值 | 基础λ数值未报告；ν为无量纲；指数软错误强度×脆弱性 | 未能数值换算 | PDF p.4式(6)(7)；p.9 Tables I/II（两表核读） |

| ESFMTO (`SIK5DA7T`) | 硬件θ=0.02–0.05；容器λ=0.03–0.09；恢复强度硬件2、容器3 | Table 2未标时间单位；Poisson故障与指数恢复强度 | 按秒制时间推定s^-1，作者表格未明示 | PDF p.4式(5)(6)等；p.8 Table 2 |

| RTPO (`KVZXU9ZL`) | 故障λ∈{0.04,0.06,0.08}；修复θ∈{0.1,0.3,0.8} | s^-1，Table II明确；故障/修复强度 | 原值即s^-1 | PDF p.10 Table II（矢量表格核读）；pp.3–4模型 |

| UDQF/checkpoint卸载 (`FJBEC772`) | 本次未从实验设置核得明确的故障Poisson参数μ数值 | 未公布统一数值/单位；正文Poisson故障计数；实验输出失败率 | 不能换算 | PDF p.3式(7)；p.7 Table II（矢量表格核读）；p.10 Fig.11 |

## 核对说明与原始PDF

### TORE-NET

λi还按归一化频率放大，λ0不是全部ES最终强度；r_i=[0.3,0.98]为长期可用性，不能当CPU故障率。按原表保留单位，不能自行改成fault/s。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/WBTK2YQA/Rasouli 等 - 2026 - TORE-NET A task replication and offloading strategy in reliable and energy-efficient mobile edge co.pdf>)；PDF p.10 Table II；pp.6–7式(16)–(18)。

### RASOUL

模型有λi=λ0·10^[ω(1-fnorm)/(1-fmin)]；参数表仅有可用性r_i等，不能继承TORE-NET的λ0。秒制执行时间要求λi是s^-1，属于量纲推断。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/ZEN8WTAP/Rasouli 等 - 2025 - RASOUL A Reliability-Aware Task Allocation Strategy to Improve Success Rate and Energy Saving in Mo.pdf>)；PDF p.5式(13)(14)；p.8 Table III（矢量表格核读）。

### HRTO

时隙0.5s。不能把周期概率p直接当λ；若p确为0.5s内故障概率，则λ=-ln(1-p)/0.5。λ=0.95是任务到达概率，不是节点故障率。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/VNXIBXJH/Guo 等 - 2025 - Hybrid redundancy for reliable task offloading in collaborative edge computing.pdf>)；PDF p.4式(3)–(5)；p.8设置；p.10 Section V-C4。

### Reliability-aware task replication

pi,j是deadline内接入与处理成功概率；文中的λ用于任务/请求到达，μ用于服务，勿误认作CPU故障率。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/CSPHAIV5/Yang 等 - 2024 - Reliability-aware task replication for mobile edge computing.pdf>)；PDF pp.5–7，Sections V–VI。

### DCR/DAA

P(Ynm≤t)=1-exp[-vnm(t-dn/fm)]为传输完成分布；不是CPU无故障执行率。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/WCQBI4P7/Dai 等 - 2024 - Distributed Convex Relaxation for Heterogeneous Task Replication in Mobile Edge Computing.pdf>)；PDF pp.4–5，式(2)–(13)。

### FTRATO

需结合实现确认T的尺度；不是明确的1%单次失败概率。模型P(无故障)=exp(-λT)。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/HT4IAF76/Zhang 等 - 2025 - Fault-tolerant resource allocation and task offloading scheme for UAV NOMA-MEC networks.pdf>)；PDF p.7 Table I；p.3式(16)(17)。

### TMDRL-RARTO

乘积直接使用γm/γu；不依赖C/f。不能把0.05写成0.05 s^-1。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/US4YLEBS/Liu 等 - 2026 - Reinforcement learning for strategy-proof reliability-aware redundancy task offloading in UAV-enable.pdf>)；PDF p.11 Table 3；p.13 Section 6.5；p.5式(12)。

### DT-MARL

时隙1s不自动证明每个子任务失败概率是每秒强度；缺少统一暴露时长。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/5JBIQ2VX/Chen 等 - 2026 - Decentralized task offloading in collaborative edge computing A digital twin assisted multi-agent r.pdf>)；PDF p.10实验设置；pp.4–5式(1)–(3)。

### ReLIEF

正文区分节点λ与链路μ，实验没有同时给出链路μ完整设置；Table III未列λ。不能将横轴数值解释为单任务概率。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/2NCJS8U6/Siyadatzadeh 等 - 2023 - ReLIEF A reinforcement-learning-based real-time task assignment strategy in emerging fault-tolerant.pdf>)；PDF p.5式(7)–(9)；p.9 Fig.6/Table III（图与表核读）。

### DTFR

SITE探测间隔30min；LDNS平均1h探测不是故障强度。链路失效为downtime/total time。不要将轨迹采样频率当节点λ。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/2Y37P5CA/Aral和Brandić - 2021 - Learning spatiotemporal failure dependencies for resilient edge computing services.pdf>)；PDF pp.5、7–8；Sections 4、6。

### DieHard

参数对象为网络/供电故障域，与任务软件瞬态故障不是同一类型；原文也假设机器相同硬件失败概率，但此段不是所有机器λ的赋值。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/U6I7C655/Sedaghat 等 - 2016 - DieHard Reliable scheduling to survive correlated failures in cloud data centers.pdf>)；PDF p.3 Section III-A。

### KPath

含节点失效、未知负载和资源竞争导致的slowdown，不仅物理故障。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/UTI8A7BZ/Shi 等 - 2025 - KPath Dependent and redundant task execution path planning in edge service networks.pdf>)；PDF p.10 Section VI-A1。

### RARE-DTR

p.14副本数量/执行时间分布示例另用λ0=0.1，没有与μs实验同样明确的单位，不把两种实验混用。检测coverage是概率而非故障强度。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/R9RCKXXZ/Hosseini 等 - 2025 - Dynamic task replication with imperfect fault detection in multicore cyber-physical systems.pdf>)；PDF pp.13–15；Section VII-A/B。

### 多目标分散计算冗余

Table II中的λj为单位能耗收费，不是故障强度。不要把reference value 10^-3误认成ϕj。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/XPMST24M/Huang 等 - 2026 - Joint latency and charge cost minimization for reliable task offloading in dispersed computing A mu.pdf>)；PDF p.5式(1)；p.11 Table II。

### DIST

正文exp(-λt)没有使实验0/10/20%自动成为λ；失败比例与故障发生强度应分开。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/LGLMU6MY/Oustad 等 - 2025 - DIST Distributed learning-based energy-efficient and reliable task scheduling and resource allocati.pdf>)；PDF pp.13–14 Fig.12及分析；p.5式(7)。

### FT-DSAC

不能把无单位数值表述成作者明确给出每秒强度；需核验原实现时间单位。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/M2DT9REX/Long 等 - 2025 - Fault-Tolerant Aware Task Offloading Based on Reinforcement Learning in Mobile Edge Computing.pdf>)；PDF p.5式(6)(7)；p.10设置；pp.11–12敏感性实验。

### ALC-LSMOEA-KT

这是附加假设下的等效连续强度，不是原文Poisson λ；p/Δ≈0.005仅是小概率近似。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/BA4TBH4K/Dong 等 - 2026 - A fault-tolerant task offloading framework via large-scale multi-objective evolutionary optimization.pdf>)；PDF p.10 Table 2。

### 车辆edge-cloud容错策略DDPG

保留Cloud高/低区间的原文数值，不按常见范围自行替换。本次正文实验p.22等明确DDPG，先前简称只称DRL。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/HGJVYJ8D/Babaiyan和Bushehrian - 2025 - A deep-reinforcement-learning-based strategy selection approach for fault-tolerant offloading of del.pdf>)；PDF p.25 Table 4；p.18指数失败模型。

### 图强化学习容错调度

执行/通信波动ω=0.1/0.15/0.25是另一项参数，勿混为φ。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/UIHNSIKV/Zhang 等 - 2024 - Fault-Tolerant Scheduling Mechanism for Dynamic Edge Computing Scenarios Based on Graph Reinforcemen.pdf>)；PDF pp.7、11–12 Section 5.1。

### LeDRL

正文α为软件、γ为硬件、β为链路强度；γ不是“环境故障率”。默认0.01与敏感性概率不是同一明示参数。实测YOLO阈值0.4/0.5也不是故障强度。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/E2W8EZ42/Hao Guo 等 - 2026 - LLM-enhanced deep reinforcement learning for task offloading in collaborative edge computing.pdf>)；PDF p.3式(3)等；p.6设置；pp.7–8 Fig.5。

### TRO

这里只是UAV故障参数扫描，不能假定UE、cloud和所有链路都设为相同范围。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/DQVFCHRP/Hao 等 - 2025 - Reliability-aware optimization of task offloading for UAV-assisted edge computing.pdf>)；PDF pp.5–6式(7)(8)；p.10设置；p.11 Fig.4(c)（矢量轴核读）。

### QoE/Reliability-Aware Scheduling

ν∈[0,1]不是故障率范围。γ称bit error rate却进入exp(-γTtx)，需单独核对量纲，不能直接当每秒故障强度。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/76BJLVYA/Zhou 等 - 2025 - Quality of experience and reliability-aware task offloading and scheduling for multi-user mobile-edg.pdf>)；PDF p.4式(6)(7)；p.9 Tables I/II（两表核读）。

### ESFMTO

0.02–0.05与0.03–0.09是两种故障，不应合并成服务器单一λ；2/3是恢复率，非故障率。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/M2XBRLUD/Yang 等 - 2025 - ESFMTO A reliable task offloading strategy based on edge server failure model in IIoT.pdf>)；PDF p.4式(5)(6)等；p.8 Table 2。

### RTPO

这篇提供明确秒制参数，可直接作量纲参照，但原系统包含失败后修复与重执行。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/LFLAP7IX/Liang 等 - 2023 - Reliability-aware task processing and offloading for data-intensive applications in edge computing.pdf>)；PDF p.10 Table II（矢量表格核读）；pp.3–4模型。

### UDQF/checkpoint卸载

输出的average failure rate不是实验输入λ；不能从降低百分比反推故障强度。

[本地PDF](</mnt/d/Users/l2354/Documents/Zotero/storage/IEU66UEI/Long 等 - 2025 - Fault-tolerant mobile service offloading in mobile edge computing.pdf>)；PDF p.3式(7)；p.7 Table II（矢量表格核读）；p.10 Fig.11。

## HRTO与LeDRL的口径不一致

HRTO正文写Poisson故障并使用exp(-αTc)，因此若Tc以秒计，α需为s^-1。默认实验段却给[0,0.3]但不印单位，敏感性段称“一个period内失败的概率”，时隙为0.5s。不能在这两种描述间随意选择；若它实际是0.5s内概率p=0.3，则对应恒定强度为-ln(0.7)/0.5≈0.71335 s^-1，而不是0.3 s^-1。

LeDRL同样存在正文Poisson强度与实验图“failure probability”的不同表述。图5横轴的0.1–0.3要按概率列出，不能假设就是α/β每秒值。正文γ为硬件故障强度；此前分析写成“环境”项不准确，已更正本轮报告的说明。

## 对当前repo的解释

本次读取的repo基础λ范围是0.001–0.005 s^-1，故障强度用于R=exp(-λeff·C/f)。与TORE-NET表中基础1 s^-1比较，repo低200–1000倍；这只是基础强度比较，双方频率/空间风险映射后不能还用这一固定倍数。

1秒CPU执行的失败概率为1-exp(-λ)：repo基础值约0.09995%–0.49875%；TORE-NET表中基础1 s^-1约63.212%。这并非报告任务级实验失败比例，尚未计具体执行时间、频率放大、可用性与副本。

参数换算必须满足λt无量纲。10^-3/ms×1000=1/s；10^-6/μs×10^6=1/s；per hour除以3600得到per second。不能只复制数字，不复制时间单位。

若只有固定暴露时长Δ下失败概率p，在恒定指数生存、独立暴露窗等假设下，可以定义等效λ=-ln(1-p)/Δ；没有Δ或这些假设时，不能唯一转换。节点失败比例、任务截止违约概率、检测coverage与长期可用性也不能直接换成CPU故障强度。

本次没有修改仿真参数，没有把原文可能有歧义的单位静默“修正”为更常见的设置。要忠实复现，应优先采用原文明确的参数单位，并对未明示/不一致处另查作者代码。
