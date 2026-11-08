"""Build the revised proposal with geographic correlation, load-driven hazard, and existing figures."""
from pathlib import Path
import csv, hashlib, shutil
# Reuse the established template and measured native-shape layout helpers.
helper=Path(__file__).with_name('build_proposal.py')
exec(compile(helper.read_text(encoding='utf8').split('# 01 cover')[0],str(helper),'exec'),globals())
BASE=helper.parent/'revision_geography_load';BASE.mkdir(exist_ok=True)
ROOT=helper.parents[2]
TITLE='考虑空间相关与负载动态故障的边缘计算可靠任务卸载方法研究'
OUT=BASE/'卫琳-开题报告（空间相关与负载动态故障·含实验）.pptx'
TOTAL=26
for _ in range(6):
 s=p.slides.add_slide(originals[11].slide_layout)
 for sh in list(s.shapes):sh._element.getparent().remove(sh._element)
originals=list(p.slides)
order=list(range(12))+list(range(20,26))+list(range(12,20))
ids=list(p.slides._sldIdLst)
for el in ids:p.slides._sldIdLst.remove(el)
for i in order:p.slides._sldIdLst.append(ids[i])
originals=[originals[i] for i in order]
figure_records=[]
SENS=ROOT/'results/five_method_sensitivity_20261104_073956_983864'
DIST=ROOT/'results/reliability_distribution_20261104_082959_793544'
with (SENS/'summary.csv').open() as f:sens=list(csv.DictReader(f))
with (DIST/'summary.csv').open() as f:dist=list(csv.DictReader(f))

def footer(s,i):tx(s,f'{i+1}/{TOTAL}',12.1,7.12,.88,.22,11,'muted',align='right')
def pic(s,path,x,y,w,h):
 path=Path(path);im=Image.open(path).convert('RGB');iw,ih=im.size
 factor=min(w/iw,h/ih);dw,dh=iw*factor,ih*factor;xx=x+(w-dw)/2;yy=y+(h-dh)/2
 s.shapes.add_picture(str(path),Inches(xx),Inches(yy),width=Inches(dw),height=Inches(dh))
 current.paste(im.resize((px(dw),px(dh)),Image.Resampling.LANCZOS),(px(xx),px(yy)))
 figure_records.append({'slide':len(previews),'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'embedded_unchanged':True})
 texts[-1].append('实验图：'+path.name)
 notes[-1]+='\n图源：'+str(path)+'。图像直接嵌入；源图与源统计未修改。'

def stage_note(s,text='现有阶段结果：固定基础故障率 + 空间相关风险；队列驱动动态故障率尚未验证。'):
 tx(s,text,.52,6.95,11.50,.23,11,'muted')
def flow(s,labels,bodies,y=2.1):
 n=len(labels);width=(12.2-(n-1)*.30)/n
 for k,(title,body) in enumerate(zip(labels,bodies)):
  x=.52+k*(width+.30);rect(s,x,y,width,1.76,'pale','line');tx(s,title,x+.13,y+.20,width-.26,.42,21,'blue',True,'center')
  tx(s,body,x+.13,y+.91,width-.26,.72,17,align='center')
  if k<n-1:line(s,x+width+.03,y+.87,x+width+.25,y+.87,'blue',1.7,True)

# 1 cover
s=begin(0,'根据用户最新要求，研究内容一为外部地理邻近引起的相关故障，研究内容二为内部队列负载驱动的动态故障率；第二项函数及参数尚未确定，按拟研究方法呈现。已纳入2026-11-04的两组近期实验，不再沿用早先跳过实验结果的要求。')
tx(s,'考虑空间相关与负载动态故障的\n边缘计算可靠任务卸载方法研究',.95,1.52,11.43,1.65,34,'blue',True,'center',lh=1.42)
tx(s,'开 题 报 告',4.75,3.45,3.83,.47,24,'blue',True,'center')
tx(s,'答 辩 人：卫琳\n指导教师：待定（职称待定）\n专       业：待定\n学       院：待定',4.35,4.35,5.1,1.75,20,'blue',lh=1.5)
tx(s,'答辩时间：待定',4.35,6.24,5.1,.32,15,'muted')
# 2 contents
s=begin(1,'沿用原模板五部分结构，研究内容与方法部分包含阶段实验；不安排系统模型概述页。')
tx(s,'目 录',.67,.12,1.60,.52,29,'white',True);tx(s,'CONTENTS',.67,.72,1.60,.30,15,'white',True)
for k,t in enumerate(['研究背景与意义','国内外研究现状','研究内容与方法','研究进度安排','参考文献']):
 y=1.78+k*.93;tx(s,f'{k+1}.',4.12,y,.7,.6,32,'blue',True);tx(s,t,4.96,y+.10,7.3,.5,27,'navy',True)
chapter(2,1,'研究背景与意义','从现有性能与资源调度研究引出故障风险描述的两个不足：空间上的独立化简化，时间上的静态化简化；二者分别对应外部环境共同风险和内部队列负载变化。')
# 4 background
s=begin(3,'地理邻近仅作为共同环境风险的代理变量，不代表距离必然造成故障。队列积压作为内部运行压力的候选代理，和物理故障率的关系需标定验证。两项研究共同改进固定节点风险参数的简化，但空间相关模型本身不等于时变故障率；真正的时间动态性由研究内容二引入。')
head(s,3,'研究背景与意义','从任务处理性能优化到故障风险精细刻画')
tx(s,'现有边缘卸载研究通过节点选择、资源分配和任务复制优化时延、能耗与服务质量。\n当节点风险被简化为固定参数时，外部共同环境与内部负载变化的影响难以充分体现。',.50,1.74,12.2,1.03,21,lh=1.40)
for x,title,body,chain in [(.52,'外部因素：地理邻近','邻近节点可能共享供电、网络或环境风险。\n副本部署在不同节点，也可能同时受损。','地理邻近 → 共同风险 → 相关故障'),(6.85,'内部因素：队列负载','任务持续到达使队列积压与运行压力变化。\n节点风险可能随负载状态演化。','负载变化 → 运行压力 → 动态故障率')]:
 rect(s,x,3.08,5.95,2.53,'pale');tx(s,title,x+.22,3.31,5.50,.47,23,'blue',True)
 tx(s,body,x+.22,4.00,5.50,.96,20,lh=1.40);tx(s,chain,x+.22,5.03,5.50,.40,18,'orange',True)
band(s,'共同目标：从固定、独立的风险参数，走向兼顾外部关联与内部负载变化的卸载决策。',6.35,18)
# 5 significance
s=begin(4,'按原模板“现状—不足—研究问题”逻辑组织。静态故障率不足有文献[3]的明确常数假设支持；[1][2]用于空间独立性不足。已有[5][6]分别研究负载感知故障率与时空相关失效，不能将所有现有研究笼统描述为故障率恒定。动态故障率为用户确认的拟研究内容。')
head(s,4,'研究背景与意义','研究意义')
rows=[('1）','现有研究通过任务复制与节点协作提高服务保障，但部分方法仍用相互独立的节点故障参数评估副本收益。对于地理位置邻近、可能受共同环境因素影响的服务器，这种简化难以反映副本间的联合失效风险，可能高估冗余执行的保障能力。','因此，如何引入外部地理关联，刻画共同风险下的相关故障，是改进静态、独立风险描述的重要研究问题。'),('2）','现有方法已将队列负载纳入时延和资源调度，但采用恒定故障率的模型无法反映负载变化对节点风险的影响。副本增加会加重队列压力，若仍以固定参数估计故障风险，就难以描述卸载决策、负载演化与可靠性之间的反馈关系。','因此，如何建立内部队列负载驱动的动态故障率，并用于在线卸载决策，是突破静态故障率假设的关键问题。')]
for k,(num,a,b) in enumerate(rows):
 y=1.79+k*2.25;tx(s,num,.45,y,.65,.47,23,'blue',True);tx(s,a,1.08,y,11.70,1.17,20,lh=1.35);tx(s,b,1.08,y+1.24,11.70,.88,20,'blue',True,lh=1.35)
band(s,'研究主线：外部地理关联引起的相关故障 + 内部队列负载驱动的动态故障率。',6.50,19)
chapter(5,2,'国内外研究现状','研究现状按外部空间共同风险与内部负载时变故障率分组。核心文献[1]—[5]集中在2024—2026年；[6]为相关故障基础研究，[7][8]为方法来源，[9]为位置数据来源。')
# 7 external literature
s=begin(6,'文献[1]第2.3节式(9)使用失效概率乘积聚合，表明独立失效形式；其停车概率随累计停车时间变化，不能将该文误写为所有概率都恒定。文献[2]第2.4节明确采用代表性的独立失效概率，并说明未纳入局部停电和网络中断导致的相关退出。二者支持地理共同风险的切入点。[6]已研究故障时空依赖，故本课题不声称首次考虑空间相关故障。')
head(s,6,'国内外研究现状','外部地理邻近与相关故障')
for k,(a,b,c) in enumerate([
('文献[1] 陈明秋等（2024）','以任务复制应对停驻车辆中止服务，设计贪婪和增强遗传算法，\n在满足服务可靠需求的条件下优化副本服务时延。','不足：副本可靠度采用独立失效乘积形式，未显式刻画邻近节点的共同风险。'),
('文献[2] Lee 等（2026）','面向志愿节点随机退出，对关键任务选择性复制，联合配置任务放置、\nCPU 频率和副本数量，协调费用、能耗与实时性。','不足：采用代表性独立失效概率，未纳入局部停电、网络中断等相关失效。')]):
 y=1.78+k*2.12;tx(s,a,.50,y,12.20,.43,22,'blue',True);tx(s,b,.50,y+.57,12.20,.90,20,lh=1.34);tx(s,c,.50,y+1.45,12.20,.65,19,'orange',True,lh=1.30)
rect(s,.43,6.32,12.42,.67,'pale');tx(s,'对应研究一：将地理邻近所代理的共同环境风险引入联合可靠性评估，\n改进仅按节点固定参数、独立组合计算副本风险的做法。',.64,6.40,11.98,.56,18,'blue',True,lh=1.22)
# 8 internal literature
s=begin(7,'[3]Liu等，2024，FGCS原文第3.4节明确将lambda_j定义为constant failure rate，式(9)可靠性为exp(-lambda_j*execution_time)，支持恒定故障率的不足。[4]Tang等，2024，正式出版题名与DOI已核对；可访问作者预印本第3.2节和第5.5节将lambda_j作为节点参数，实验从N(0.005,10^-6)生成异质参数，而队列进入Lyapunov控制，未给出lambda_j随Q_j(t)更新的映射。这一判断基于作者公开版本，与正式版本细节可能有差异。[5]赵庶旭等，2026，全文已建立CPU使用率相关故障率和最大压力控制，不能称其故障率恒定。本文与其的研究对象区别在于：该文按DAG就绪层部署并限制资源使用率；本课题拟将CPU队列积压轨迹、外部空间共同风险和事件驱动双副本策略联合考虑。该差异是课题切入点，创新性及效果仍需进一步验证。')
head(s,7,'国内外研究现状','内部队列负载与动态故障率')
items=[
('文献[3] Liu 等（2024）','在能耗约束下联合优化调度时间与可靠性，并利用任务副本提高容错能力。','不足：明确采用恒定 CPU 故障率，未反映队列负载变化对故障强度的影响。'),
('文献[4] Tang 等（2024）','结合主从复制与 Lyapunov 控制，协调队列负载、能耗和冗余开销。','不足：故障率作为节点参数设置，缺少由队列状态驱动故障率更新的机制。'),
('文献[5] 赵庶旭等（2026）','已建立负载感知故障率模型，通过限制服务器最大资源使用率进行任务部署。','研究边界：侧重资源使用率与部署；拟进一步联合队列演化、空间共同风险和副本选择。')]
for k,(a,b,c) in enumerate(items):
 y=1.76+k*1.40;tx(s,a,.50,y,12.20,.36,20,'blue',True);tx(s,b,.50,y+.46,12.20,.38,18);tx(s,c,.50,y+.91,12.20,.43,17.5,'orange',True)
rect(s,.43,6.28,12.42,.70,'pale');tx(s,'对应研究二：建立队列负载—故障率的动态映射，研究负载、失效风险与卸载动作的反馈。\n改进固定风险参数假设，刻画外部共同风险与内部队列负载共同作用下的卸载决策。',.64,6.37,11.98,.57,18,'blue',True,lh=1.22)
chapter(8,3,'研究内容与方法','本部分直接描述两项研究方法及阶段实验，不设置系统模型介绍页。现有实验仅覆盖固定基础故障率叠加空间相关风险，第二项动态故障率方法尚未实现或验证。')
# 10 overview
s=begin(9,'两项研究均改进风险刻画：第一项为外部空间关联，已有实现与阶段实验；第二项为内部负载驱动的时间动态性，按用户确认的拟研究方案描述。第一项并不自动意味着故障率随时间变化。')
head(s,9,'研究内容与方法','两项研究内容：由外部关联到内部动态')
for x,num,title,body,steps in [(.50,'研究内容一 · 已有原型','外部地理关联导致的相关故障','研究邻近节点受共同环境风险影响时，\n如何评估副本联合失效并选择卸载位置。',['地理距离 → 潜在风险关联','联合可靠性积分 → 允许动作','相关感知策略 → 阶段实验']),(6.88,'研究内容二 · 拟研究','内部队列负载驱动的动态故障率','研究负载演化改变节点风险时，\n如何动态评估可靠性并调整卸载策略。',['队列积压 → 故障强度映射','时变风险积分 → 在线更新','空间与负载联动 → 对照验证'])]:
 rect(s,x,1.83,5.91,3.85,'pale');tx(s,num,x+.22,2.01,5.45,.36,17,'muted',True);tx(s,title,x+.22,2.58,5.45,.82,22,'blue',True)
 tx(s,body,x+.22,3.55,5.45,.90,19)
 for j,t in enumerate(steps):tx(s,'• '+t,x+.22,4.62+j*.32,5.47,.30,16.5)
band(s,'共同目标：改进固定风险参数的简化，在任务可靠性要求下优化在线卸载时延。',6.35,19)
# 11 spatial method
s=begin(10,'研究方法与core/spatial_risk.py一致。距离核刻画潜在高斯风险的相关性，不是故障事件Pearson相关系数；风险强度为lambda0_j*exp(beta*Z_j-beta²/2)。现阶段将潜在风险积分掉，不抽样逐时刻风险场，不能将其表述为已经实现时间动态故障率。GH积分只使用CPU执行暴露时间。')
head(s,10,'研究内容与方法','研究一：从地理邻近到联合失效评估')
flow(s,['地理距离表征','潜在风险关联','条件故障强度','联合可靠性'],['用距离代理共同\n环境影响程度','以距离核构建\n节点风险依赖','在基础故障率上\n引入环境风险','积分计算副本\n共同失效概率'],2.00)
tx(s,'ρⱼₖ = exp(−dⱼₖ / ℓ)',.67,4.20,5.95,.55,26,'navy',True)
tx(s,'λⱼ(Zⱼ) = λ₀ⱼ exp(βZⱼ − β²/2)',6.60,4.20,6.10,.55,24,'navy',True)
tx(s,'Rᵢ(j,k) = 1 − E[(1 − rᵢⱼ(Zⱼ))(1 − rᵢₖ(Zₖ))]',.67,5.18,12.0,.60,26,'navy',True)
tx(s,'采用 Gauss–Hermite 数值积分；独立对照保持相同边缘风险，只去除节点间关联。',.58,6.16,12.1,.50,20)
tx(s,'现阶段定位：对外部空间关联建模；基础故障率尚不随内部队列负载更新。',.58,6.85,12.0,.28,13,'muted')
# 12 spatial strategy
s=begin(11,'已有策略实现为可靠性允许动作筛选结合PPO。可行集为空时回退至最大可靠性服务器对，不保证达标。此处只描述技术路径，不介绍服务器、任务或执行架构等系统模型。观测为当前可得队列信息，并非完整Markov状态。')
head(s,11,'研究内容与方法','研究一：相关感知的卸载决策与验证路径')
flow(s,['估计联合风险','筛选允许动作','学习卸载策略','记录任务反馈'],['计算服务器对\n相关可靠性','按逐任务要求\n形成可行集合','结合当前队列\n以 PPO 选择副本','以完成时延反馈\n更新策略'],1.94)
tx(s,'可靠性约束',.60,4.26,5.70,.42,22,'blue',True);tx(s,'Aᵢ = {(j,k): Rᵢ(j,k) ≥ rᵢ}',.60,4.95,5.70,.55,25,'navy',True)
tx(s,'时延优化与对照',6.88,4.26,5.70,.42,22,'blue',True);tx(s,'相关 / 独立 × PPO / Greedy\n另设单副本 RASOUL-Delay 参考。',6.88,4.95,5.70,.92,20,'ink')
band(s,'验证重点：独立假设下的风险偏差，以及可靠性满足率与平均时延之间的取舍。',6.40,18)
# 13 existing training and evaluation settings
s=begin(12,'来源为five_method_sensitivity_20261104_073956_983864。8个EUA站点，可靠性要求U[0.92,0.999]；Corr-PPO和Ind-PPO同种子各训练8000轮、每轮200任务；每点评估500轮，模型及优化器冻结；评估任务、到达序列和服务器在方法间共享。训练过程统计不能替代独立评估。仅一个训练种子，区间不覆盖训练种子差异。可靠性满足率按解析可靠性是否达到任务要求计算，不抽样实际失败；当前仅CPU执行期暴露风险。')
head(s,12,'研究内容与方法','研究一阶段实验：训练过程与评估口径')
tx(s,'8 个站点 · 两种 PPO 各训练 8000 轮 · 每个评估点 500 轮 · 五种方法使用配对输入',.56,1.70,12.1,.40,18,'muted')
pic(s,SENS/'figures/training_curves.png',.57,2.13,12.15,3.62)
with (SENS/'training_summary.csv').open() as f:train=list(csv.DictReader(f))
tr={r['method']:r for r in train}
tx(s,f"Corr-PPO：前 500 轮 {float(tr['ppo']['first_500_mean_delay']):.2f} s → 后 500 轮 {float(tr['ppo']['last_500_mean_delay']):.2f} s。",.60,5.96,12.0,.42,20,'blue',True)
tx(s,'正式结果使用冻结模型与独立评估集；训练曲线用于说明当前策略学习的实现基础。',.60,6.47,12.0,.36,18)
stage_note(s)
# 14-16 all six recent linear figures, one factor per page
blocks=[
('arrival_rate','任务到达率', '到达率增加会加重排队；基准 Corr-PPO 时延 6.64 s，Corr-Greedy 为 14.60 s。\n相关方法满足率维持 99.50%，与当前故障率不随队列负载更新的设定一致。'),
('demand_multiplier','任务计算需求','计算需求从 0.5 倍增至 3 倍时，Corr-PPO 满足率由 100% 降至 84%。\n计算需求同时增加执行暴露与排队压力；当前结果不能单独归因于动态故障率。'),
('task_count','每轮任务数量','任务数从 200 增至 600，Corr-PPO 时延约由 6.64 s 增至 8.39 s。\n这是每轮排空的有限任务实验；不能将其解释为系统稳定态或动态故障率验证。')]
for j,(key,label,conclusion) in enumerate(blocks):
 i=13+j;s=begin(i,'本页为2026-11-04敏感性实验已有图。对应CSV为'+str(SENS/'summary.csv')+'。每次只改变一个因素，所有方法使用共享输入。双副本按相关解析可靠性、单副本按边际可靠性评估满足率；图中“任务成功率”指模型可靠性约束满足比例。当前基础故障率不随Q(t)更新。'+conclusion)
 head(s,i,'研究内容与方法','研究一阶段实验：'+label+'的影响')
 pic(s,SENS/'figures_linear'/f'{key}_mean_delay_seconds.png',.40,1.72,6.16,4.18)
 pic(s,SENS/'figures_linear'/f'{key}_success_rate.png',6.78,1.72,6.16,4.18)
 tx(s,conclusion,.57,6.04,12.15,.74,18,'blue',True,lh=1.30)
 stage_note(s)
# 17 latest reliability distribution: delay
s=begin(16,'三组可靠性分布实验位于reliability_distribution_20261104_082959_793544，各组均重新训练两种PPO模型；除阈值之外任务属性、服务器、到达时刻配对。500轮评估，每种PPO只有一个训练种子。时延包含所有任务和回退任务。严格阈值组的相关双副本可行性上限5.5%，94.5%任务回退，高时延主要反映高回退下的队列积压，不应据此比较可靠性达标策略的优劣。')
head(s,16,'研究内容与方法','研究一阶段实验：不同可靠性需求下的时延')
pic(s,DIST/'figures/mean_delay_seconds_bar.png',.48,1.73,12.35,4.90)
tx(s,'严格要求组中 Corr-PPO 回退率为 94.5%，平均时延约 145.57 s；应结合可行性解释。',.57,6.64,12.1,.34,17,'blue',True)
stage_note(s)
# 18 latest reliability distribution: RSR
s=begin(17,'与上一页为同一组新实验。相关双副本可行性上限随三组阈值分别为100%、94%、5.5%；Corr-PPO达到这些上限。中间组Corr-PPO与Ind-PPO实际相关口径满足率为94%和86.201%，差约7.799个百分点。达到模型可行性上限不代表时延最优，也不是实测任务无故障成功率。高阈值下固定双副本与当前服务器参数可能不可满足要求，需要正视不可行性。')
head(s,17,'研究内容与方法','研究一阶段实验：不同可靠性需求下的达标比例')
pic(s,DIST/'figures/success_rate_bar.png',.48,1.73,12.35,4.90)
tx(s,'U[0.99, 0.999]：Corr-PPO 为 94.00%，Ind-PPO 为 86.20%；更严格组可行上限仅 5.50%。',.57,6.64,12.1,.34,17,'blue',True)
stage_note(s)
# 19 load hazard method, no system-model overview
s=begin(18,'用户已确认动态故障率函数和依据尚未确定，按拟研究方法表述。拟将运行剩余工作和排队服务工作量等作为内部压力代理，建立lambda_j(t)对负载的映射；不预设经验公式、系数或单调关系已获实测证实。可考虑使用利用率、温度、错误日志等辅助标定，区分物理执行失效、截止期超时和队列溢出。可靠性使用CPU执行窗口内的时变危险率积分；未来队列不可直接观测，须使用当前可得信息和短期预测，避免前视。')
head(s,18,'研究内容与方法','研究二：内部队列负载驱动的动态故障率（拟研究）')
flow(s,['提取负载压力','建立动态映射','累计风险暴露'],['运行剩余工作\n排队服务工作量','由负载及风险参数\n更新节点故障强度','对执行窗口内的\n时变强度积分'],1.95)
tx(s,'λⱼ(t) = F(λ₀ⱼ, qⱼ(t), Zⱼ; θ)',.68,4.10,11.8,.60,27,'navy',True)
tx(s,'rᵢⱼ = E[ exp(−∫ λⱼ(u) du) ]',.68,5.03,11.8,.60,27,'navy',True)
tx(s,'F 为拟研究的负载—故障率映射；积分窗口采用 CPU 执行区间。\n具体函数、参数与短期负载预测方法需结合文献和数据标定。',.68,5.97,11.8,.85,20,lh=1.30)
# 20 load learning and test plan
s=begin(19,'拟研究流程在研究一空间共同风险基础上加入队列负载引起的时变故障强度，并动态更新可行动作。使用当前可得队列及预测信息，不能调用未来实际轨迹。由于失效暴露与负载在动作后会演化，需评估基于短期预测/保守近似的动作筛选误差。拟进行2x2对照：独立/相关 × 恒定/负载驱动；保持边缘参数、任务、到达及种子可比。需校准失效事件及积分估计，验证策略是否缓解拥堵—失效反馈。尚无动态故障率结果，不展示伪造动态故障曲线。')
head(s,19,'研究内容与方法','研究二：风险更新与卸载策略联动（拟研究）')
flow(s,['感知当前队列','预测风险变化','更新允许动作','决策并反馈'],['观察负载与\n服务工作量','融合外部共同风险\n与内部负载映射','重估副本联合可靠性\n及任务约束','PPO 选择卸载位置\n按事件更新观测'],1.90)
tx(s,'拟回答的问题',.57,4.17,5.65,.43,22,'blue',True)
tx(s,'卸载动作如何改变队列及后续风险？\n固定故障率会在什么负载下低估风险？\n风险预测误差如何影响可行集合？',.57,4.82,5.73,1.50,20,lh=1.48)
tx(s,'拟采用的对照设计',6.88,4.17,5.65,.43,22,'blue',True)
tx(s,'空间独立 / 空间相关\n×\n恒定故障率 / 负载驱动动态故障率',6.88,4.82,5.73,1.50,21,'navy',True,lh=1.40)
band(s,'验证目标：在相同输入下区分空间关联、负载动态及二者联动对策略的影响。',6.55,18)
# 21 research characteristics and progress
s=begin(20,'阶段划分保持证据边界：研究一已有固定基础故障率下的空间相关风险、GH与PPO实现，近期图用于阶段展示；研究二尚在模型与验证方案设计阶段。已有文献[5]研究负载感知，[6]研究时空依赖，所以创新论证聚焦具体结合、推导和验证，不宣称首次提出负载动态故障率或相关故障。近期结果为单训练种子、有限任务批次，不能推出普遍最优或长期稳定性。')
head(s,20,'研究内容与方法','拟形成的研究特点与当前进展')
for x,a,b,c,d in [(.52,'外部空间关联','以地理邻近刻画共同环境风险，\n改进独立失效假设下的副本风险估计。','当前进展','已实现联合可靠性积分与卸载原型，\n已完成训练及多组配对评估。'),(6.87,'内部负载动态','以队列负载更新故障强度，研究卸载、\n负载演化与风险变化之间的反馈。','后续重点','明确负载映射与标定依据，完善时变\n风险估计、策略联动和独立对照。')]:
 rect(s,x,1.83,5.94,4.45,'pale');tx(s,a,x+.22,2.10,5.50,.43,23,'blue',True);tx(s,b,x+.22,2.82,5.50,.98,20)
 tx(s,c,x+.22,4.16,5.50,.43,22,'blue',True);tx(s,d,x+.22,4.88,5.50,.98,20)
band(s,'创新论证聚焦：外部共同风险与内部队列动态共同作用下的可靠任务卸载。',6.55,18)
chapter(21,4,'研究进度安排','研究进度按已有研究一原型与阶段实验，以及尚未完成的研究二映射/算法/验证安排；具体日期待用户毕业节点确认。')
# 23 schedule
s=begin(22,'建议以相对月份编排，用户未提供毕业时间。研究一已有阶段结果，后续需风险参数依据、多训练种子与稳健性验证；研究二需要先明确映射和标定，再实现策略及对照实验。')
head(s,22,'研究进度安排','围绕两项研究内容的后续安排')
cols=[.48,2.62,9.12];widths=[2.13,6.49,3.72]
for x,w,t in zip(cols,widths,['计划阶段','主要工作','阶段产出']):rect(s,x,1.77,w,.55,'blue');tx(s,t,x+.14,1.92,w-.28,.29,18,'white',True)
rows=[('第 1—2 个月','完善空间风险参数依据，补充多训练种子与稳健性分析。','研究一完善稿'),('第 3—4 个月','确定队列负载—动态故障率映射，明确标定及风险暴露。','研究二模型与推导'),('第 5—6 个月','实现动态可靠性估计、允许动作更新及策略联动。','动态风险卸载原型'),('第 7—8 个月','开展相关/独立与恒定/动态交叉对照、消融和敏感性实验。','研究二评估与分析'),('第 9—10 个月','整合两项研究，完成论文写作、修改与答辩准备。','论文与答辩材料')]
for k,row in enumerate(rows):
 y=2.32+k*.81
 for x,w,t in zip(cols,widths,row):rect(s,x,y,w,.81,'pale' if k%2==0 else 'white','line');tx(s,t,x+.14,y+.15,w-.28,.60,17 if x!=cols[0] else 18)
tx(s,'具体起止年月及毕业节点待定；当前阶段实验已单独展示，动态故障率仍为拟研究。',.52,6.73,12.0,.38,17,'muted')
chapter(23,5,'参考文献','核心文献[1]—[5]与两项研究内容对应，基础文献和数据源为[6]—[9]。近期结果来源记录在对应页面备注和图源清单中。')
# 25 references
s=begin(24,'[1][2]用于外部共同风险；[3]原文明确恒定故障率；[4]正式出版页与作者预印本联合核验，队列动态与故障率参数区分；[5]已考虑CPU使用率相关故障率，作为需要比较的近邻研究，不能标注为静态故障率。[6]为时空依赖基础，[7][8]为PPO与GAE，[9]为地理位置来源。')
head(s,24,'参考文献','近年研究文献及方法、数据来源')
refs=[
('[1] 陈明秋, 黄家乐, 武继刚. 面向停驻车辆辅助边缘计算的服务可靠增强算法. 广东工业大学学报, 2024, 41(4): 80–88.','https://doi.org/10.12052/gdutxb.230104'),
('[2] Lee J, Kwon G, Bahn H. Real-Time Task Offloading with Replication Under Node Churn in Volunteer Edge Computing. Mathematics, 2026, 14(17): 3092.','https://doi.org/10.3390/math14173092'),
('[3] Liu R, Wu W, Guo X, et al. Replica fault-tolerant scheduling with time guarantee under energy constraint in fog computing. Future Generation Computer Systems, 2024, 159: 567–579.','https://doi.org/10.1016/j.future.2024.05.014'),
('[4] Tang W, Gao D, Yu S, et al. Reliable and adaptive computation offload strategy with load and cost coordination for edge computing. Pervasive and Mobile Computing, 2024, 102: 101932.','https://doi.org/10.1016/j.pmcj.2024.101932'),
('[5] 赵庶旭, 朱祺, 王小龙. 面向移动边缘计算的可靠性增强任务部署方法. 浙江大学学报(工学版), 2026, 60(8): 1697–1708.','https://doi.org/10.3785/j.issn.1008-973X.2026.08.009'),
('[6] Aral A, Brandić I. Learning Spatiotemporal Failure Dependencies for Resilient Edge Computing Services. IEEE Transactions on Parallel and Distributed Systems, 2021, 32(7): 1578–1590.','https://doi.org/10.1109/TPDS.2020.3046188'),
('[7] Schulman J, Wolski F, Dhariwal P, et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.','https://arxiv.org/abs/1707.06347'),
('[8] Schulman J, Moritz P, Levine S, et al. High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv:1506.02438, 2015.','https://arxiv.org/abs/1506.02438'),
('[9] Lai P, et al. EUA Datasets: Edge server and user datasets for Edge Computing research. 数据集仓库，访问日期：2026-11-04.','https://github.com/PhuLai/eua-dataset')]
y=1.70
for t,url in refs:
 n=len(wrap(t,12.3,13.2));hh=n*13.2*1.23/72+.06;tx(s,t,.47,y,12.36,hh,13.2,link=url);y+=hh+.09
reference_bottom=y
# 26 closing
s=begin(25,'报告人卫琳。研究一为外部地理相关风险，研究二为内部队列动态故障率；用户个人信息待补充，动态映射待明确。')
tx(s,'请各位专家批评指正！',1.1,2.28,11.13,1.08,45,'blue',True,'center');tx(s,'答辩人：卫琳',4.77,4.58,3.8,.44,23,'blue',align='center')
tx(s,'外部地理关联 · 内部队列负载 · 可靠任务卸载',1.13,5.55,11.07,.46,20,'muted',align='center')
# Clean notes and metadata.
for i,s in enumerate(originals):
 s.notes_slide.notes_text_frame.text=notes[i]
 for el in list(s._element.xpath('.//p:custDataLst')):el.getparent().remove(el)
p.core_properties.author='卫琳';p.core_properties.last_modified_by='卫琳';p.core_properties.title=TITLE
p.core_properties.subject='开题报告：两项研究内容与阶段实验';p.core_properties.keywords='地理相关故障,队列负载,动态故障率,可靠卸载,PPO'
p.core_properties.comments='基于用户模板改写；实验图直接引用已有输出；动态故障率按拟研究方案表述。';p.core_properties.revision=1
p.core_properties.created=datetime.now(timezone.utc);p.core_properties.modified=datetime.now(timezone.utc);p.save(str(OUT))
with zipfile.ZipFile(OUT) as z:parts={n:z.read(n) for n in z.namelist()}
root=etree.fromstring(parts['ppt/presentation.xml'])
for el in root.xpath('//*[local-name()="sectionLst"]'):el.getparent().remove(el)
parts['ppt/presentation.xml']=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
app_ns='http://schemas.openxmlformats.org/officeDocument/2006/extended-properties';app=etree.Element('{'+app_ns+'}Properties',nsmap={None:app_ns})
for key,val in [('Application','python-pptx'),('PresentationFormat','宽屏'),('Slides',str(TOTAL)),('Notes',str(TOTAL)),('HiddenSlides','0')]:etree.SubElement(app,'{'+app_ns+'}'+key).text=val
parts['docProps/app.xml']=etree.tostring(app,xml_declaration=True,encoding='UTF-8',standalone=True)
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in parts.items():z.writestr(n,b)
proof=BASE/'layout_proof';proof.mkdir(exist_ok=True)
for i,im in enumerate(previews):im.save(proof/f'{i+1:02d}.png')
thumbw,thumbh,gap=480,270,18;rows=math.ceil(TOTAL/4)
contact=Image.new('RGB',(4*thumbw+5*gap,rows*(thumbh+34)+(rows+1)*gap),'#E5EAF1');d=ImageDraw.Draw(contact)
for i,im in enumerate(previews):
 x=gap+(i%4)*(thumbw+gap);y=gap+(i//4)*(thumbh+34+gap);contact.paste(im.resize((thumbw,thumbh),Image.Resampling.LANCZOS),(x,y));d.text((x,y+thumbh+5),f'{i+1:02d}',font=font(10),fill='#1C4885')
contact.save(BASE/'卫琳-开题报告-版面总览.png')
md=['# 卫琳｜开题报告内容稿','',f'> 暂拟题目：{TITLE}。共{TOTAL}页，包含9张已有实验图。研究二为拟研究方法，函数与参数待标定。导师、专业、学院及日期待定。','']
for i,items in enumerate(texts):
 md.extend([f'## 第 {i+1} 页','']);md.extend([t+'\n' for t in items if not re.fullmatch(r'\d+/\d+',t)]);md.extend(['讲述提示：'+notes[i],''])
md.extend(['## 文献来源','']+[f'- [{t}]({url})' for t,url in refs])
(BASE/'卫琳-开题报告-内容稿.md').write_text('\n'.join(md),encoding='utf8')
provenance={'figures':figure_records,'source_summaries':[{'file':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}for f in [SENS/'summary.csv',SENS/'training_summary.csv',DIST/'summary.csv']],'dynamic_failure_rate_status':'proposed; not implemented or validated by the included figures','validation_scope':'OOXML, measured layout proof; not an Office render'}
(BASE/'实验图源清单.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'pptx':str(OUT),'slides':len(p.slides),'embedded_figures':len(figure_records),'size_bytes':OUT.stat().st_size,'text_layout_overflow':violations,'reference_bottom_inch':reference_bottom},ensure_ascii=False,indent=2))
if violations:raise SystemExit('Text overflow needs correction')
