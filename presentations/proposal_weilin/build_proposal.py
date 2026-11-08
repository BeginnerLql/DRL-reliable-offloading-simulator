from pathlib import Path
from copy import deepcopy
from io import BytesIO
import json, math, re, zipfile
from datetime import datetime, timezone
from functools import lru_cache
from lxml import etree
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.oxml.xmlchemy import OxmlElement

BASE=Path(__file__).resolve().parent
TEMPLATE=Path('/mnt/d/Users/l2354/Documents/xwechat_files/wxid_meywboco9xrl22_4c14/msg/file/2026-10/贾中海-开题报告.pptx')
OUT=BASE/'卫琳-开题报告（研究现状优化版）.pptx'
FONT='/mnt/c/Windows/Fonts/msyh.ttc'
BOLD='/mnt/c/Windows/Fonts/msyhbd.ttc'
MATH_FONT='/mnt/c/Windows/Fonts/cambria.ttc'
SUBSCRIPT_CHARS=set('ᵢⱼₖ₀')
C={'blue':'1C4885','navy':'002872','ink':'252D3A','muted':'607087','pale':'F1F5FB','line':'D1DCEB','white':'FFFFFF','orange':'C96920','warm':'FFF4E8'}
p=Presentation(str(TEMPLATE)); originals=list(p.slides)
W,H=1600,900; SCALE=120
previews=[]; texts=[]; notes=[]; current=None; violations=[]

# Keep the template's cover border, contents background, and chapter artwork.
keep={0:{5,6,7,8,9},1:{54,6,47},2:{3,10},5:{3,10},8:{3,10},15:{3,10},17:{3,10},19:{5,6,7,8,9}}
for i,s in enumerate(originals):
 for shape in list(s.shapes):
  if shape.shape_id not in keep.get(i,set()):
   el=shape._element; el.getparent().remove(el)
 for shape in s.shapes:
  if shape.has_text_frame: shape.text_frame.clear()
 # Old speaker notes, comments, tags, embedded objects and unused images are not carried forward.
 used=set(s._element.xpath('//@r:embed'))
 for rel in list(s.part.rels.values()):
  typ=rel.reltype.rsplit('/',1)[-1]
  if typ in {'notesSlide','tags','comments','commentAuthors','oleObject','chart'} or (typ=='image' and rel.rId not in used):
   s.part.drop_rel(rel.rId)
 for tag in ['p:timing','p:transition','p:extLst']:
  for el in s._element.findall(tag, s._element.nsmap):s._element.remove(el)


def rgb(c):return RGBColor.from_string(C.get(c,c))
@lru_cache(maxsize=128)
def font(sz,b=False):return ImageFont.truetype(BOLD if b else FONT,round(sz*SCALE/72))
@lru_cache(maxsize=128)
def math_font(sz):return ImageFont.truetype(MATH_FONT,round(sz*SCALE/72))
def font_runs(text,sz,b=False):
 chunks=[]
 for ch in text:
  kind=ch in SUBSCRIPT_CHARS
  if chunks and chunks[-1][0]==kind:chunks[-1]=(kind,chunks[-1][1]+ch)
  else:chunks.append((kind,ch))
 return [(kind,t,math_font(sz) if kind else font(sz,b)) for kind,t in chunks]
def text_width(text,sz,b=False):return sum(f.getlength(t) for _,t,f in font_runs(text,sz,b))
def px(x):return round(x*SCALE)
def wrap(text,w,sz,b=False):
 f=font(sz,b); lines=[]
 for raw in str(text).split('\n'):
  if not raw: lines.append('');continue
  line=''
  for ch in raw:
   if text_width(line+ch,sz,b)>w*SCALE-6 and line:
    # Keep short English words intact when a word crosses the line.
    m=re.search(r'([A-Za-z][A-Za-z0-9\-]*)$',line)
    if m and m.start()>0 and ch.isascii() and ch.isalnum():
     lines.append(line[:m.start()].rstrip());line=m.group(1)+ch
    else:lines.append(line);line=ch
   else:line+=ch
  lines.append(line)
 return lines

def tx(s,text,x,y,w,h,sz=20,color='ink',bold=False,align='left',fill=None,lh=1.23,link=None):
 text=str(text);lines=wrap(text,w,sz,bold)
 linept=sz*lh
 needed=(len(lines)-1)*linept+sz*1.12
 if needed>h*72+2: violations.append((len(previews),text[:50],round(needed,1),round(h*72,1)))
 sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame
 tf.clear();tf.word_wrap=False;tf.auto_size=MSO_AUTO_SIZE.NONE
 tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
 tf.vertical_anchor=MSO_ANCHOR.TOP
 for j,line in enumerate(lines):
  q=tf.paragraphs[0] if j==0 else tf.add_paragraph()
  q.alignment={'left':PP_ALIGN.LEFT,'center':PP_ALIGN.CENTER,'right':PP_ALIGN.RIGHT}[align]
  q.line_spacing=Pt(linept);q.space_after=Pt(0);q.space_before=Pt(0)
  for kind,chunk,_ in font_runs(line,sz,bold):
   r=q.add_run();r.text=chunk;r.font.name='Cambria' if kind else 'Microsoft YaHei';r.font.size=Pt(sz);r.font.bold=bold;r.font.color.rgb=rgb(color)
   rpr=r._r.get_or_add_rPr();ea=OxmlElement('a:ea');ea.set('typeface','Cambria' if kind else '微软雅黑');rpr.append(ea)
   if link:r.hyperlink.address=link
 texts[-1].append(text)
 # Draw from the same measured lines for a layout proof (not an Office render).
 draw=ImageDraw.Draw(current);f=font(sz,bold)
 for j,line in enumerate(lines):
  xx=x*SCALE
  if align=='center':xx+=(w*SCALE-text_width(line,sz,bold))/2
  elif align=='right':xx+=w*SCALE-text_width(line,sz,bold)
  baseline=y*SCALE+j*linept*SCALE/72+font(sz,bold).getmetrics()[0]
  for kind,chunk,cf in font_runs(line,sz,bold):
   draw.text((xx,baseline),chunk,font=cf,fill='#'+C.get(color,color),anchor='ls')
   xx+=cf.getlength(chunk)
 return sh

def rect(s,x,y,w,h,fill='pale',line=None,r=0):
 sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if r else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
 sh.fill.solid();sh.fill.fore_color.rgb=rgb(fill)
 if line:sh.line.color.rgb=rgb(line);sh.line.width=Pt(.8)
 else:sh.line.fill.background()
 if r:
  try:sh.adjustments[0]=.08
  except Exception:pass
 d=ImageDraw.Draw(current);box=(px(x),px(y),px(x+w),px(y+h))
 if r:d.rounded_rectangle(box,radius=px(r),fill='#'+C.get(fill,fill),outline='#'+C.get(line,line) if line else None,width=1)
 else:d.rectangle(box,fill='#'+C.get(fill,fill),outline='#'+C.get(line,line) if line else None,width=1)
 return sh

def line(s,x1,y1,x2,y2,color='line',width=1,arrow=False):
 sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2))
 sh.line.color.rgb=rgb(color);sh.line.width=Pt(width)
 if arrow:
  el=OxmlElement('a:tailEnd');el.set('type','triangle');el.set('w','sm');el.set('len','sm');sh.line._get_or_add_ln().append(el)
 d=ImageDraw.Draw(current);col='#'+C.get(color,color);d.line((px(x1),px(y1),px(x2),px(y2)),fill=col,width=max(1,round(width*SCALE/72)))
 if arrow:
  theta=math.atan2(y2-y1,x2-x1);a=10
  v=(px(x2),px(y2));d.polygon([v,(v[0]-a*math.cos(theta-.45),v[1]-a*math.sin(theta-.45)),(v[0]-a*math.cos(theta+.45),v[1]-a*math.sin(theta+.45))],fill=col)
 return sh

def retain_preview(s):
 # Draw the unchanged template decorations into the layout proof.
 for sh in s.shapes:
  box=(px(sh.left/914400),px(sh.top/914400),px((sh.left+sh.width)/914400),px((sh.top+sh.height)/914400))
  if sh.shape_type==13:
   im=Image.open(BytesIO(sh.image.blob)).convert('RGBA');im=im.resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS)
   current.paste(im,(box[0],box[1]),im)
  elif sh.shape_type in (1,14):
   try:
    fc=sh.fill.fore_color
    if fc.type==1:col=str(fc.rgb)
    else:
     el=sh._element.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}solidFill')
     col='FFFFFF' if el is not None else None
    if col:
     d=ImageDraw.Draw(current)
     if sh.auto_shape_type==MSO_SHAPE.OVAL:d.ellipse(box,fill='#'+col)
     elif sh.auto_shape_type==MSO_SHAPE.ROUNDED_RECTANGLE:d.rounded_rectangle(box,radius=22,fill='#'+col)
     else:d.rectangle(box,fill='#'+col)
   except Exception:pass

def begin(i,note):
 global current
 current=Image.new('RGB',(W,H),'white');previews.append(current);texts.append([]);notes.append(note)
 s=originals[i];retain_preview(s)
 return s

def footer(s,i):tx(s,f'{i+1}/20',12.1,7.08,.88,.22,11,'muted',align='right')
def head(s,i,section,subtitle):
 tx(s,section,.31,.10,12.5,.60,28,'ink',True)
 line(s,.35,.835,13.05,.835,'navy',1)
 rect(s,.35,.795,1.8,.08,'navy')
 tx(s,'• '+subtitle,.37,1.02,12.4,.48,23,'navy',True)
 footer(s,i)
def band(s,text,y=6.48,sz=19):
 rect(s,.38,y,12.55,.51,'pale');rect(s,.38,y,.055,.51,'blue')
 tx(s,text,.58,y+.09,12.1,.34,sz,'blue',True)
def bullets(s,items,x,y,w,sz=20,gap=.28):
 for label,body in items:
  tx(s,label,x,y,w,.42,sz,'blue',True)
  ls=wrap(body,w-.05,sz-1)
  hh=max(.42,len(ls)*(sz-1)*1.23/72+.06)
  tx(s,body,x,y+.49,w,hh,sz-1)
  y+=.49+hh+gap
 return y

def chapter(i,n,title,note):
 s=begin(i,note)
 # The number circle remains a template shape; add fresh text to avoid retained authorship.
 tx(s,str(n),3.35,3.25,1.16,.95,50,'blue',True,'center')
 tx(s,title,4.62,3.33,8.1,.80,38,'navy',True)
 footer(s,i)

# 01 cover
s=begin(0,'开题报告题目为暂拟题目，依据当前可靠任务卸载项目编写。报告人卫琳，用户未提供的身份信息均保留待定。实验结果及原作者成果不纳入本稿。')
tx(s,'考虑空间相关故障的边缘计算\n可靠任务卸载方法研究',.95,1.52,11.43,1.65,36,'blue',True,'center',lh=1.42)
tx(s,'开 题 报 告',4.75,3.45,3.83,.47,24,'blue',True,'center')
tx(s,'答 辩 人：卫琳\n指导教师：待定（职称待定）\n专       业：待定\n学       院：待定',4.35,4.35,5.1,1.75,20,'blue',lh=1.5)
tx(s,'答辩时间：待定',4.35,6.24,5.1,.32,15,'muted')
# 02 contents
s=begin(1,'沿用原模板五部分结构。本版不包含实验结果、投稿状态或已完成性能结论。')
tx(s,'目 录',.67,.12,1.60,.52,29,'white',True)
tx(s,'CONTENTS',.67,.72,1.60,.30,15,'white',True)
for k,t in enumerate(['研究背景与意义','国内外研究现状','研究内容与方法','研究进度安排','参考文献']):
 y=1.78+k*.93;tx(s,f'{k+1}.',4.12,y,.7,.6,32,'blue',True);tx(s,t,4.96,y+.10,7.3,.5,27,'navy',True)
chapter(2,1,'研究背景与意义','介绍边缘任务卸载的应用需求与相关故障带来的可靠性问题。')
# 04 background
s=begin(3,'文献[1]讨论停驻车辆提供边缘计算与任务副本卸载，文献[5]讨论时空相关故障。此处供电与环境风险为相关失效的动机示例。当前实现仅对CPU执行阶段计算可靠性，未显式模拟真实故障、故障恢复、链路失效或用户移动性。')
head(s,3,'研究背景与意义','从低时延卸载走向可靠卸载')
tx(s,'边缘计算将任务卸载至邻近服务器，有助于满足终端的计算与时延需求。[1]\n当任务采用双副本并行执行时，副本位置、服务器负载与共同风险将共同影响决策。',.42,1.70,12.3,1.0,21,lh=1.45)
rect(s,.55,3.15,2.30,1.63,'pale','line');tx(s,'终端任务',.78,3.38,1.84,.38,22,'blue',True,'center');tx(s,'数据量 / 计算量\n可靠性要求',.73,3.93,1.94,.70,17,align='center')
rect(s,3.66,3.15,2.65,1.34,'blue');tx(s,'卸载调度器',3.90,3.39,2.17,.4,22,'white',True,'center');tx(s,'选择两个不同服务器',3.83,4.00,2.30,.35,16,'white',align='center')
line(s,2.9,3.82,3.57,3.82,'blue',2,True)
rect(s,7.17,3.00,2.47,1.62,'warm','orange',.08);tx(s,'共同风险区域',7.38,3.15,2.05,.35,18,'orange',True,'center')
for x,t in [(7.42,'服务器 A'),(8.52,'服务器 B')]:
 rect(s,x,3.80,.92,.57,'white','orange');tx(s,t,x+.03,3.97,.86,.26,12,'orange',True,'center')
rect(s,10.38,3.51,2.25,.82,'pale','line');tx(s,'服务器 C',10.60,3.75,1.81,.38,19,'blue',True,'center')
line(s,6.35,3.79,7.08,3.79,'blue',2,True);line(s,6.72,3.80,6.72,4.98,'blue',1.5);line(s,6.72,4.98,11.45,4.98,'blue',1.5);line(s,11.45,4.98,11.45,4.45,'blue',1.5,True)
tx(s,'副本 A、B 可能受相同环境因素影响',7.25,5.28,5.45,.43,18,'orange',True)
band(s,'关键问题：如何在考虑故障关联与排队竞争的条件下，选择可靠且低时延的卸载位置？',6.25,18)
# 05 significance
s=begin(4,'参照用户提供模板的行文逻辑，按“现有研究方向—限定条件下的不足—需要解决的问题”组织两段论述，分别引出空间相关风险建模与可靠性约束下的队列感知在线卸载。第一段由性能和资源优化研究切入，讨论采用独立失效假设时可能高估副本容错收益的局限；第二段由节点选择和负载分配切入，讨论副本决策对当前任务和后续队列的耦合影响。近年代表性研究见文献[1]—[4]，共同风险与独立假设局限见文献[2][5]；动态队列与副本耦合为本课题系统模型中的问题分析。已有文献[5]已经研究相关故障，因此不声称现有研究均未考虑相关性，也不声称本课题首次提出。本文研究特点与效果仍需验证。')
head(s,4,'研究背景与意义','研究意义')
rows=[
 ('1）','现有边缘计算任务卸载研究主要围绕降低时延、减少能耗和提高资源利用效率展开，并通过任务复制增强服务保障。然而，采用节点独立失效假设的副本配置方法，难以反映邻近服务器受共同环境风险影响的情况，可能高估副本带来的容错收益。','因此，如何刻画服务器间的空间风险关联，准确评估双副本任务的联合可靠性，是制定合理卸载决策的重要基础。'),
 ('2）','现有卸载策略通过节点选择与负载分配提高任务处理效率。但在任务持续到达的场景下，副本配置还会改变服务器队列及后续任务的资源竞争；若故障风险评估与动态调度分开处理，就难以兼顾任务要求与长期时延表现。','因此，在满足任务可靠性要求的条件下，如何联合考虑空间相关风险与动态队列负载，设计低时延的在线卸载策略，是本课题需要解决的关键问题。')
]
for k,(num,status,question) in enumerate(rows):
 y=1.79+k*2.25
 tx(s,num,.45,y,.65,.47,23,'blue',True)
 tx(s,status,1.08,y,11.70,1.17,20,lh=1.35)
 tx(s,question,1.08,y+1.24,11.70,.88,20,'blue',True,lh=1.35)
band(s,'本课题切入点：空间相关风险建模 + 可靠性约束与队列感知的在线卸载',6.50,19)
chapter(5,2,'国内外研究现状','核心研究现状选取2024—2026年发表的4篇文献，按“文献方法—具体局限—对应研究切入点”的原模板逻辑展开；相关故障基础研究[5]及算法基础[6][7]另列参考，不将PPO或动作掩码本身作为新方法。')
# 07 literature
s=begin(6,'[1]陈明秋等，2024，原文第2.2节以条件停车概率表示单车服务可靠度，第2.3节式(9)以1减各副本失效概率乘积计算服务可靠度。该乘积聚合形式体现独立失效假设，因此“未显式刻画节点间共同风险”为依据公式作出的模型范围判断，不是作者直接自述；第2.4节使用传输与计算时延并按副本平均分配CPU，不能误写为完全未考虑资源竞争。[2]Lee等，2026，2026-08-28正式发表。第2.4节式(10)(11)采用独立节点失效模型，作者明确说明局部网络中断、停电等相关churn未显式纳入。本页的归纳只针对这两项工作的模型，不泛化为全部已有研究；[5]已经研究时空依赖。对应研究内容一为构建距离相关的潜在风险模型，计算双副本联合可靠性，并在相同边缘风险下与独立假设比较。')
head(s,6,'国内外研究现状','针对副本配置与故障风险建模')
items=[
 ('文献[1] 陈明秋等（2024）','针对停驻车辆服务可能中止的问题，引入任务复制，设计贪婪与增强遗传算法，\n在满足服务可靠需求的条件下，降低任务副本的平均服务时延。','不足：采用独立失效的乘积形式聚合副本可靠度，未显式刻画节点间共同风险。'),
 ('文献[2] Lee 等（2026）','针对志愿边缘节点随机退出的问题，对关键任务选择性复制，并联合优化任务放置、\nCPU 频率与副本数量，协调能耗、费用和实时性要求。','不足：明确假设节点独立失效，未纳入局部停电、网络中断等引起的相关失效。')
]
for k,(title,method,gap) in enumerate(items):
 y=1.78+k*2.12
 tx(s,title,.50,y,12.20,.43,22,'blue',True)
 tx(s,method,.50,y+.57,12.20,.90,20,lh=1.34)
 tx(s,gap,.50,y+1.45,12.20,.65,19,'orange',True,lh=1.30)
rect(s,.43,6.32,12.42,.67,'pale');rect(s,.43,6.32,.055,.67,'blue')
tx(s,'研究切入点一：建立空间相关风险模型，评估双副本联合可靠性，\n避免在共同风险场景中直接沿用独立失效假设。',.64,6.40,11.98,.56,18,'blue',True,lh=1.22)
# 08 research gap
s=begin(7,'[3]Jiang等，2024，EMPPO原文第2.1节为多UE、单MEC服务器，队列位于UE侧，按固定长度时帧决策；第3节按有限码长传输错误概率筛选用户选择动作，已有动作掩码与动态任务到达，不能声称该文不考虑队列、动态性或可靠性约束。相对本课题的范围局限是单服务器通信调度，未解决多个CPU队列之间的双副本服务器对选择与空间共同风险。[4]Yang等，2025，ESFMTO依据出版商摘要核对：硬件与容器故障及恢复交互模型，SAC-BNN估计完成时延分布与CVaR，并通过状态扰动提高稳健性。正文未取得，因此仅比较其摘要明确的“服务器内部故障交互”建模范围，不推断其未考虑CPU排队、未设置硬约束或未具备在线能力。由内部故障交互模型不能直接替代跨服务器共同风险模型，是本文的研究范围比较，不是作者自述缺陷。对应研究内容二为将服务器对联合可靠性约束构造成允许动作集合，并结合动态CPU队列及事件间隔学习卸载策略；不将PPO、负时延奖励或动作掩码本身声称为首次提出。')
head(s,7,'国内外研究现状','针对约束驱动的学习卸载决策')
items=[
 ('文献[3] Jiang 等（2024，EMPPO）','针对动态任务与随机衰落信道，在传输错误概率约束下采用动作掩码 PPO，\n联合进行用户选择与码长分配，优化任务用于决策时的信息时效。','不足：面向单服务器传输调度，未覆盖多服务器双副本选择及 CPU 队列之间的竞争。'),
 ('文献[4] Yang 等（2025，ESFMTO）','建立硬件与容器故障及恢复的交互模型，结合 SAC-BNN 估计任务完成时延分布，\n以条件风险价值（CVaR）评估尾部风险，提高故障条件下的卸载稳健性。','局限：其服务器内部故障交互模型，不能直接替代跨节点共同风险与副本联合失效建模。')
]
for k,(title,method,gap) in enumerate(items):
 y=1.78+k*2.12
 tx(s,title,.50,y,12.20,.43,22,'blue',True)
 tx(s,method,.50,y+.57,12.20,.90,20,lh=1.34)
 tx(s,gap,.50,y+1.45,12.20,.65,19,'orange',True,lh=1.30)
rect(s,.43,6.32,12.42,.67,'pale');rect(s,.43,6.32,.055,.67,'blue')
tx(s,'研究切入点二：以服务器对联合可靠性筛选动作，结合动态 CPU 队列与事件驱动 PPO，\n研究满足任务要求的低时延在线副本选择。',.64,6.40,11.98,.56,18,'blue',True,lh=1.22)
chapter(8,3,'研究内容与方法','本部分使用当前项目实现作为技术基础，按拟研究方案表述，实验结果已依照用户要求省略。')
# 10 research contents
s=begin(9,'本课题分为互相支撑的两个研究内容：相关可靠性建模与约束策略学习。项目已经包含模型和PPO原型，但算法效果、最优性和适用边界仍需要后续实验验证。')
head(s,9,'研究内容与方法','研究内容与总体思路')
for x,num,title,desc,steps in [(.50,'研究内容一','空间相关可靠性建模','针对独立假设难以反映共同风险的问题，\n构建适用于双副本卸载的联合可靠性模型。',['地理距离与潜在风险相关核','条件失效强度与副本可靠性','Gauss–Hermite 数值积分']),(6.88,'研究内容二','可靠性约束的学习卸载','针对排队状态随任务到达动态变化的问题，\n研究动作筛选与 PPO 结合的在线卸载策略。',['任务—服务器观测与动作设计','可行集合与无可行动作回退','事件驱动 PPO 策略优化'])]:
 rect(s,x,1.83,5.91,3.85,'pale');tx(s,num,x+.22,2.01,5.45,.36,17,'muted',True)
 tx(s,title,x+.22,2.58,5.45,.45,24,'blue',True);tx(s,desc,x+.22,3.25,5.45,1.01,19)
 for j,t in enumerate(steps):tx(s,'• '+t,x+.22,4.47+j*.35,5.47,.31,17)
line(s,6.42,3.68,6.81,3.68,'blue',2,True)
band(s,'技术路线：系统建模 → 联合可靠性估计 → 可靠动作筛选 → 学习卸载策略 → 后续验证',6.25,18)
# 11 system
s=begin(10,'模型依据core/task.py、core/env_state.py与core/main_loop.py。每个任务选择两个不同服务器，两个完整副本分别上传并竞争CPU。任务时延按最先完成副本定义；另一个副本继续执行，不进行副本取消。上传与CPU处理可重叠；CPU为单服务台排队。风险只作用于CPU执行阶段。时延为仿真完成时间，可靠性以解析积分单独评估，不采样失败事件。')
head(s,10,'研究内容与方法','系统模型与问题边界')
tx(s,'任务与服务器',.50,1.78,4.85,.45,23,'blue',True)
tx(s,'任务 i：数据量 Dᵢ、计算量 Cᵢ、\n可靠性要求 rᵢ。\n服务器 j：传输速率 vⱼ、CPU 频率 fⱼ、\n基础失效强度 λ₀ⱼ、地理位置与队列。',.50,2.42,5.03,1.86,20,lh=1.48)
tx(s,'动作 aᵢ = {j, k}，j ≠ k\n选择两个不同服务器执行完整副本。',.50,4.67,5.15,1.18,20,'blue',True,lh=1.45)
tx(s,'双副本并行执行',6.12,1.78,6.54,.45,23,'blue',True)
for yy,t in [(2.62,'副本 A'),(3.98,'副本 B')]:
 tx(s,t,6.12,yy+.14,1.04,.34,17,'blue',True)
 for xx,ww,lab,fill in [(7.35,1.33,'上传','pale'),(9.02,1.33,'排队','pale'),(10.69,1.51,'CPU 执行','blue')]:
  rect(s,xx,yy,ww,.68,fill,'line' if fill=='pale' else None);tx(s,lab,xx+.05,yy+.21,ww-.1,.30,17,'white' if fill=='blue' else 'blue',True,'center')
 line(s,8.70,yy+.34,8.98,yy+.34,'blue',1.3,True);line(s,10.36,yy+.34,10.64,yy+.34,'blue',1.3,True)
tx(s,'任务完成时间取最先完成的副本；\n另一副本继续占用资源直至执行完成。',6.12,5.12,6.4,.94,20,'orange',True)
band(s,'任务时延 Tᵢ = min{副本 A 完成时刻，副本 B 完成时刻} − 任务到达时刻',6.40,19)
# 12 correlation
s=begin(11,'公式与core/spatial_risk.py一致。Z为零均值单位方差高斯潜在风险场，相关矩阵由指数距离核给出；ρ为潜在风险相关性，不是故障事件Pearson相关系数。归一化因子exp(-β²/2)使失效强度均值仍为λ₀。给定潜在场，两个副本条件独立。Gauss–Hermite积分计算未知潜在风险下的边缘与联合可靠性，无需抽样某次场实现。独立基线设置ρ=0并保持λ₀、β与边缘可靠性一致。当前实现仅对CPU执行时间Cᵢ/fⱼ暴露风险，未将上传和排队时间计入失效暴露；β与相关长度等需要标定。')
head(s,11,'研究内容与方法','研究内容一：空间相关风险与双副本可靠性')
for x,title,formula,desc in [(.50,'空间相关结构','ρⱼₖ = exp(−dⱼₖ / ℓ)','dⱼₖ：服务器地理距离\nℓ：潜在风险相关长度'),(4.80,'条件失效强度','λⱼ(Zⱼ) = λ₀ⱼ ×\nexp(βZⱼ − β²/2)','Z：标准化高斯潜在风险\nβ：环境风险扰动强度'),(9.10,'条件执行可靠性','rᵢⱼ(Zⱼ) =\nexp[−λⱼ(Zⱼ) Cᵢ/fⱼ]','仅对 CPU 执行阶段计入风险\n给定潜在场后副本条件独立')]:
 rect(s,x,1.83,3.73,2.30,'pale');tx(s,title,x+.16,2.03,3.40,.42,21,'blue',True)
 tx(s,formula,x+.16,2.67,3.40,.74,18,'navy',True)
 tx(s,desc,x+.16,3.42,3.40,.63,16)
tx(s,'双副本联合可靠性',.53,4.50,5.7,.44,23,'blue',True)
tx(s,'Rᵢ(j,k) = 1 − E[(1 − rᵢⱼ(Zⱼ))(1 − rᵢₖ(Zₖ))]',.65,5.15,12.0,.56,26,'navy',True)
tx(s,'采用 Gauss–Hermite 积分计算期望；独立对照令 ρⱼₖ = 0，并保持相同边缘可靠性。',.55,5.99,12.2,.45,19)
tx(s,'模型说明：ρ 描述潜在风险的相关程度；故障强度、风险扰动与相关长度仍需后续标定。',.55,6.67,12.2,.38,17,'muted')
# 13 constrained problem
s=begin(12,'优化目标为可靠性约束下的平均任务完成时延。训练使用负时延和时间折扣作为策略优化替代目标，不能直接声称具有平均时延全局最优保证。可靠性可行集按每个任务的要求计算。可行集为空时，当前实现保留可靠性最大的全部并列动作，以best-effort方式继续处理；该行为并不保证满足原约束，必须单独计入无可行动作与回退率。RSR定义为模型估计可靠性不小于要求的任务比例，并非抽样任务成功率。')
head(s,12,'研究内容与方法','研究内容二：可靠性约束与动作筛选')
rect(s,.50,1.77,12.3,1.31,'pale')
tx(s,'优化目标',.74,1.97,1.70,.40,22,'blue',True)
tx(s,'min  E[ (1/K) ∑ᵢ Tᵢ ]',2.73,1.95,5.55,.48,26,'navy',True)
tx(s,'约束：Rᵢ(j,k) ≥ rᵢ，j ≠ k',7.97,2.03,4.50,.43,22,'navy',True)
steps=[(.50,'01  估计可靠性','计算所有服务器对的\n联合可靠性 Rᵢ(j,k)。'),(4.80,'02  构建可行集合','Aᵢ = {(j,k): Rᵢ(j,k) ≥ rᵢ}\n屏蔽不满足要求的动作。'),(9.10,'03  学习低时延选择','在允许的服务器对中，\n由 PPO 选择卸载动作。')]
for x,t,body in steps:
 rect(s,x,3.62,3.73,1.70,'pale','line');tx(s,t,x+.17,3.81,3.40,.39,21,'blue',True);tx(s,body,x+.17,4.46,3.40,.74,18)
line(s,4.27,4.43,4.73,4.43,'blue',1.8,True);line(s,8.57,4.43,9.03,4.43,'blue',1.8,True)
rect(s,.50,5.70,12.30,1.13,'warm');tx(s,'无可行动作时的处理',.73,5.88,3.65,.39,21,'orange',True)
tx(s,'保留可靠性最大的服务器对，记录回退率与未达标任务；\n回退操作不代表满足可靠性约束。',4.43,5.88,8.08,.85,18,'orange')
# 14 PPO
s=begin(13,'实现依据agents/ppo_agent.py、core/main_loop.py与core/env_state.py。观测使用服务器基础参数、运行剩余服务时间、排队工作量与当前任务数据量/计算量，可靠性要求通过动作筛选起作用。原型观测不是完整系统状态，不包含在途上传任务。时间不等间隔的决策转移采用每秒折扣基数γ及γ^Δt，完成时延产生后反馈负时延奖励。GAE在按事件顺序的轨迹上计算；当前策略比率、采样和熵使用相同掩码分布。本文只陈述方案与实现基础，不声称收敛或性能优越。PPO与GAE参考[6][7]。')
head(s,13,'研究内容与方法','事件驱动的 PPO 卸载技术路线 [6][7]')
xs=[.50,3.02,5.54,8.06,10.58]
for k,(x,title,body) in enumerate(zip(xs,['观测构建','可靠性估计','动作筛选','PPO 决策','执行与反馈'],['任务属性\n服务器参数与队列','潜在风险模型\n联合可靠性积分','逐任务可靠性要求\n形成允许动作集合','采样服务器对\n双副本并行卸载','记录完成时延\n生成奖励与转移'])):
 rect(s,x,1.94,2.25,1.87,'blue' if k==3 else 'pale','line' if k!=3 else None)
 tx(s,title,x+.10,2.19,2.05,.40,20,'white' if k==3 else 'blue',True,'center');tx(s,body,x+.10,2.97,2.05,.68,16,'white' if k==3 else 'ink',align='center')
 if k<4:line(s,x+2.29,2.90,x+2.48,2.90,'blue',1.7,True)
line(s,11.70,4.02,11.70,4.35,'blue',1.2);line(s,11.70,4.35,9.18,4.35,'blue',1.2);line(s,9.18,4.35,9.18,3.93,'blue',1.2,True)
for x,title,formula,body in [(.52,'时延驱动的奖励','uᵢ = −Tᵢ','用任务完成时延反馈引导选择，\n学习当前副本配置对后续负载的影响。'),(6.87,'不等间隔的决策转移','γₖ = γ ^ Δtₖ','按事件间隔计算折扣并估计优势，\n通过 PPO 裁剪目标更新策略。')]:
 tx(s,title,x,4.73,5.75,.42,22,'blue',True);tx(s,formula,x,5.36,5.75,.50,26,'navy',True);tx(s,body,x,6.03,5.75,.83,19)
# 15 innovation feasibility
s=begin(14,'以下为拟形成的研究特点，不作为创新性已经成立或实验结果已验证的声明。可行性依据当前代码：core/spatial_risk.py相关风险与GH积分，core/task.py离散事件双副本任务，agents/ppo_agent.py策略原型，agents/corr_delay_greedy_agent.py相关感知贪心，Ind-PPO/Indep-delayGreedy作为独立对照，data中EUA服务器位置。EUA只提供地理位置来源，不提供故障实测数据[8]。原型仿真完成不代表实验参数已经定稿；风险参数标定、不同负载/要求下的验证以及策略效果分析属于后续工作。本版不包含结果。')
head(s,14,'研究内容与方法','拟形成的研究特点与可行性基础')
tx(s,'拟形成的研究特点',.52,1.82,5.75,.43,23,'blue',True)
bullets(s,[('相关感知的联合可靠性估计','区分边缘风险与副本间依赖，在相同边缘分布下分析独立假设的影响。'),('可靠筛选与队列感知策略结合','将逐任务可靠性要求嵌入动作选择，结合队列负载学习卸载策略。')],.52,2.56,5.73,20,.37)
tx(s,'已有实现与后续工作',6.87,1.82,5.75,.43,23,'blue',True)
bullets(s,[('已有：仿真与算法原型','已具备双副本离散事件仿真、相关可靠性积分和 PPO 原型；位置取自 EUA。[8]'),('后续：标定与系统验证','补充风险参数依据，完善独立/相关对照与消融，检验适用条件和计算开销。')],6.87,2.56,5.75,20,.37)
band(s,'当前阶段：具备实现基础；方法效果与适用范围仍需通过后续实验验证。',6.42,19)
chapter(15,4,'研究进度安排','用户尚未提供开题及毕业日期，计划采用开题后的相对月份，后续可替换为具体年月，不沿用原模板过期计划。')
# 17 schedule
s=begin(16,'以下为建议进度，采用相对月份，未推定用户的毕业日期、已完成阶段或论文发表计划。用户确认学校要求和毕业时间后可改为日历时间。研究基础中的代码原型并不代表所有阶段均已完成。')
head(s,16,'研究进度安排','建议进度：按开题后的相对月份编排')
cols=[.48,2.62,9.12];widths=[2.13,6.49,3.72]
for x,w,t in zip(cols,widths,['计划阶段','主要工作','阶段产出']):rect(s,x,1.77,w,.55,'blue');tx(s,t,x+.14,1.92,w-.28,.29,18,'white',True)
rows=[('第 1—2 个月','完善文献调研、系统假设与问题定义；补充风险参数依据。','文献综述与模型说明'),('第 3—4 个月','完善空间相关可靠性模型、积分精度检查及独立对照。','模型推导与验证方案'),('第 5—6 个月','完善动作筛选、队列观测与事件驱动 PPO 实现。','可复现的算法原型'),('第 7—8 个月','开展对比、消融与敏感性实验，分析有效条件和计算开销。','实验分析与论文主体'),('第 9—10 个月','整理论文初稿，补充验证，完成修改与答辩准备。','学位论文与答辩材料')]
for k,row in enumerate(rows):
 y=2.32+k*.81
 for x,w,t in zip(cols,widths,row):rect(s,x,y,w,.81,'pale' if k%2==0 else 'white','line');tx(s,t,x+.14,y+.15,w-.28,.60,17 if x!=cols[0] else 18)
tx(s,'说明：具体起止年月与毕业节点待定；此表为建议安排，可按培养计划调整。',.52,6.73,12.0,.38,17,'muted')
chapter(17,5,'参考文献','文献条目已按作者原稿、出版页或机构数据库核对，参考来源同时在备注及内容稿中提供。')
# 19 refs
s=begin(18,'核心研究现状使用文献[1]—[4]，发表年份为2024、2026、2024、2025；文献[5]为相关故障基础研究，文献[6][7]为PPO与GAE方法来源，文献[8]为位置数据来源，不将基础文献冒充近年研究成果。每个条目均附可点击来源。文献[4]仅取得出版商摘要，未据此断言未考虑排队或未设置硬约束。访问日期2026-11-04。')
head(s,18,'参考文献','近年研究文献及方法、数据来源')
refs=[
('[1] 陈明秋, 黄家乐, 武继刚. 面向停驻车辆辅助边缘计算的服务可靠增强算法. 广东工业大学学报, 2024, 41(4): 80–88.','https://doi.org/10.12052/gdutxb.230104'),
('[2] Lee J, Kwon G, Bahn H. Real-Time Task Offloading with Replication Under Node Churn in Volunteer Edge Computing. Mathematics, 2026, 14(17): 3092.','https://doi.org/10.3390/math14173092'),
('[3] Jiang Z, Yang J, Gao X. Minimizing Task Age upon Decision for Low-Latency MEC Networks Task Offloading with Action-Masked Deep Reinforcement Learning. Sensors, 2024, 24(9): 2812.','https://doi.org/10.3390/s24092812'),
('[4] Yang Y, Chen Y, Chen N, Chen J. ESFMTO: A reliable task offloading strategy based on edge server failure model in IIoT. Ad Hoc Networks, 2025: 103887.','https://doi.org/10.1016/j.adhoc.2025.103887'),
('[5] Aral A, Brandić I. Learning Spatiotemporal Failure Dependencies for Resilient Edge Computing Services. IEEE Transactions on Parallel and Distributed Systems, 2021, 32(7): 1578–1590.','https://doi.org/10.1109/TPDS.2020.3046188'),
('[6] Schulman J, Wolski F, Dhariwal P, et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.','https://arxiv.org/abs/1707.06347'),
('[7] Schulman J, Moritz P, Levine S, et al. High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv:1506.02438, 2015.','https://arxiv.org/abs/1506.02438'),
('[8] Lai P, et al. EUA Datasets: Edge server and user datasets for Edge Computing research. 数据集仓库，访问日期：2026-11-04.','https://github.com/PhuLai/eua-dataset')]
y=1.70
for t,url in refs:
 n=len(wrap(t,12.3,14.2));hh=n*14.2*1.23/72+.06
 tx(s,t,.47,y,12.36,hh,14.2,link=url);y+=hh+.12
reference_bottom=y
# 20 closing
s=begin(19,'报告人卫琳。其余封面资料待用户补充。实验结果页暂未纳入。')
tx(s,'请各位专家批评指正！',1.1,2.28,11.13,1.08,45,'blue',True,'center')
tx(s,'答辩人：卫琳',4.77,4.58,3.8,.44,23,'blue',align='center')
tx(s,'考虑空间相关故障的边缘计算可靠任务卸载方法研究',1.13,5.55,11.07,.46,19,'muted',align='center')

# Add clean speaker notes and source provenance; no inherited remarks remain.
for i,s in enumerate(originals):
 s.notes_slide.notes_text_frame.text=notes[i]
 # Empty template decorations may have legacy metadata; remove all shape tags.
 for el in list(s._element.xpath('.//p:custDataLst')):
  el.getparent().remove(el)
p.core_properties.author='卫琳';p.core_properties.last_modified_by='卫琳'
p.core_properties.title='考虑空间相关故障的边缘计算可靠任务卸载方法研究'
p.core_properties.subject='开题报告（不含实验结果）'
p.core_properties.keywords='边缘计算,空间相关故障,可靠任务卸载,PPO,开题报告'
p.core_properties.comments='基于用户提供的开题报告模板改写。'
p.core_properties.revision=1
p.core_properties.created=datetime.now(timezone.utc)
p.core_properties.modified=datetime.now(timezone.utc)
p.save(str(OUT))
# Remove presentation sections/custom properties that could retain template names.
with zipfile.ZipFile(OUT) as z:parts={n:z.read(n) for n in z.namelist()}
ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
for path in ['ppt/presentation.xml']:
 root=etree.fromstring(parts[path])
 for el in root.xpath('//*[local-name()="sectionLst"]'):
  el.getparent().remove(el)
 parts[path]=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
app_ns='http://schemas.openxmlformats.org/officeDocument/2006/extended-properties'
app=etree.Element('{'+app_ns+'}Properties',nsmap={None:app_ns})
for key,val in [('Application','python-pptx'),('PresentationFormat','宽屏'),('Slides','20'),('Notes','20'),('HiddenSlides','0')]:
 etree.SubElement(app,'{'+app_ns+'}'+key).text=val
parts['docProps/app.xml']=etree.tostring(app,xml_declaration=True,encoding='UTF-8',standalone=True)
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in parts.items():z.writestr(n,b)

# Review artifacts. Proof images share the measured text layout used in the PPTX.
proof=BASE/'layout_proof';proof.mkdir(exist_ok=True)
for i,im in enumerate(previews):im.save(proof/f'{i+1:02d}.png')
thumbw,thumbh=480,270;gap=18
contact=Image.new('RGB',(4*thumbw+5*gap,5*(thumbh+34)+6*gap),'#E5EAF1');d=ImageDraw.Draw(contact)
for i,im in enumerate(previews):
 x=gap+(i%4)*(thumbw+gap);y=gap+(i//4)*(thumbh+34+gap)
 contact.paste(im.resize((thumbw,thumbh),Image.Resampling.LANCZOS),(x,y));d.text((x,y+thumbh+5),f'{i+1:02d}',font=font(10),fill='#1C4885')
contact.save(BASE/'卫琳-开题报告-版面总览.png')
md=['# 卫琳｜开题报告内容稿','', '> 暂拟题目：考虑空间相关故障的边缘计算可靠任务卸载方法研究。导师及职称、专业、学院、答辩时间和毕业时间待定。本版沿用所提供模板的五部分结构，共20页，不包含实验结果。','']
for i,items in enumerate(texts):
 md.extend([f'## 第 {i+1} 页',''])
 md.extend([t+'\n' for t in items if not re.fullmatch(r'\d+/20',t)])
 md.extend(['讲述提示：'+notes[i],''])
md.extend(['## 参考来源','']+[f'- [{t}]({url})' for t,url in refs])
(BASE/'卫琳-开题报告-内容稿.md').write_text('\n'.join(md),encoding='utf-8')
(BASE/'待补充信息.txt').write_text('已确认：报告人卫琳；基于当前项目；题目由助手暂拟；实验结果先跳过。\n待补充：导师姓名与职称、专业、学院、开题/答辩时间、毕业时间、正式题目（如需调整）。\n计划使用相对月份，未沿用模板的旧时间。\nPPT正文、图示、公式和表格均可编辑；版面总览为根据相同几何与字体生成的排版校样，并非Office软件渲染。\n',encoding='utf-8')
print(json.dumps({'pptx':str(OUT),'slides':len(p.slides),'size_bytes':OUT.stat().st_size,'text_layout_overflow':violations,'reference_bottom_inch':reference_bottom},ensure_ascii=False,indent=2))
if violations: raise SystemExit('Text layout overflow needs correction')
