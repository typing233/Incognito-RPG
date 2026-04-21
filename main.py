from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uvicorn

app = FastAPI()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class NoteType:
    MATH = "math"
    ENGLISH = "english"
    PHYSICS = "physics"
    HISTORY = "history"

NOTE_CONTENTS = {
    NoteType.MATH: """第一章 函数与极限
1.1 映射与函数
一、集合
1. 集合的概念
具有某种特定性质的事物的总体称为集合，组成这个集合的事物称为该集合的元素。

2. 集合的表示法
列举法：把集合的全体元素一一列举出来，例如 A = {a, b, c, d}
描述法：若集合 M 是由具有某种性质 P 的元素 x 的全体所组成，则 M 可表示为
M = {x | x 具有性质 P}

3. 常用数集
N：自然数集（非负整数）
Z：整数集
Q：有理数集
R：实数集

二、映射
1. 映射的概念
设 X、Y 是两个非空集合，如果存在一个法则 f，使得对 X 中的每个元素 x，按法则 f，在 Y 中有唯一确定的元素 y 与之对应，则称 f 为从 X 到 Y 的映射。

1.2 数列的极限
一、数列极限的定义
设 {x_n} 为一数列，如果存在常数 a，对于任意给定的正数 ε（不论它多么小），总存在正整数 N，使得当 n > N 时，不等式 |x_n - a| < ε 都成立，那么就称常数 a 是数列 {x_n} 的极限，或者称数列 {x_n} 收敛于 a。

记作：lim(n→∞) x_n = a 或 x_n → a (n → ∞)

1.3 函数的极限
一、自变量趋于有限值时函数的极限
设函数 f(x) 在点 x_0 的某一去心邻域内有定义。如果存在常数 A，对于任意给定的正数 ε（不论它多么小），总存在正数 δ，使得当 x 满足不等式 0 < |x - x_0| < δ 时，对应的函数值 f(x) 都满足不等式 |f(x) - A| < ε，那么常数 A 就叫做函数 f(x) 当 x → x_0 时的极限。

记作：lim(x→x0) f(x) = A 或 f(x) → A (当 x → x0)

第二章 导数与微分
2.1 导数概念
一、引例
1. 直线运动的速度
设某点沿直线运动。在直线上引入原点和单位点，使直线成为数轴。此外，再取定一个时刻作为测量时间的零点。设动点于时刻 t 在直线上的位置的坐标为 s（简称位置 s）。这样，运动完全由某个函数 s = s(t) 所确定。

从时刻 t0 到时刻 t0 + Δt 的时间间隔内，动点从位置 s(t0) 移动到 s(t0 + Δt)，则在这段时间间隔内的平均速度为：
v̄ = [s(t0 + Δt) - s(t0)] / Δt

二、导数的定义
1. 函数在一点处的导数与导函数
定义：设函数 y = f(x) 在点 x0 的某个邻域内有定义，当自变量 x 在 x0 处取得增量 Δx（点 x0 + Δx 仍在该邻域内）时，相应地，因变量取得增量 Δy = f(x0 + Δx) - f(x0)；如果 Δy 与 Δx 之比当 Δx → 0 时的极限存在，则称函数 y = f(x) 在点 x0 处可导，并称这个极限为函数 y = f(x) 在点 x0 处的导数，记为 f'(x0)。

即：f'(x0) = lim(Δx→0) [f(x0 + Δx) - f(x0)] / Δx

2.2 函数的求导法则
一、函数的和、差、积、商的求导法则
定理1：如果函数 u = u(x) 及 v = v(x) 都在点 x 具有导数，那么它们的和、差、积、商（除分母为零的点外）都在点 x 具有导数，且：

(1) [u(x) ± v(x)]' = u'(x) ± v'(x)
(2) [u(x)v(x)]' = u'(x)v(x) + u(x)v'(x)
(3) [u(x)/v(x)]' = [u'(x)v(x) - u(x)v'(x)] / v(x)²  (v(x) ≠ 0)

第三章 微分中值定理与导数的应用
3.1 微分中值定理
一、罗尔定理
定理1：如果函数 f(x) 满足：
(1) 在闭区间 [a, b] 上连续；
(2) 在开区间 (a, b) 内可导；
(3) 在区间端点的函数值相等，即 f(a) = f(b)，

那么在 (a, b) 内至少有一点 ξ (a < ξ < b)，使得 f'(ξ) = 0。

二、拉格朗日中值定理
定理2：如果函数 f(x) 满足：
(1) 在闭区间 [a, b] 上连续；
(2) 在开区间 (a, b) 内可导，

那么在 (a, b) 内至少有一点 ξ (a < ξ < b)，使等式 f(b) - f(a) = f'(ξ)(b - a) 成立。

第四章 不定积分
4.1 不定积分的概念与性质
一、原函数与不定积分的概念
定义1：如果在区间 I 上，可导函数 F(x) 的导函数为 f(x)，即对任一 x ∈ I，都有 F'(x) = f(x) 或 dF(x) = f(x)dx，那么函数 F(x) 就称为 f(x)（或 f(x)dx）在区间 I 上的一个原函数。

定义2：在区间 I 上，函数 f(x) 的带有任意常数项的原函数称为 f(x)（或 f(x)dx）在区间 I 上的不定积分，记作 ∫f(x)dx。

其中记号 ∫ 称为积分号，f(x) 称为被积函数，f(x)dx 称为被积表达式，x 称为积分变量。

第五章 定积分
5.1 定积分的概念与性质
一、定积分问题举例
1. 曲边梯形的面积
设 y = f(x) 在区间 [a, b] 上非负、连续。由直线 x = a、x = b、y = 0 及曲线 y = f(x) 所围成的图形称为曲边梯形，其中曲线弧称为曲边。

二、定积分的定义
定义：设函数 f(x) 在 [a, b] 上有界，在 [a, b] 中任意插入若干个分点

a = x0 < x1 < x2 < ... < xn-1 < xn = b

把区间 [a, b] 分成 n 个小区间，各个小区间的长度依次为
Δx1 = x1 - x0, Δx2 = x2 - x1, ..., Δxn = xn - xn-1

记 λ = max{Δx1, Δx2, ..., Δxn}，如果不论对 [a, b] 怎样划分，也不论在小区间 [xi-1, xi] 上点 ξi 怎样选取，只要当 λ → 0 时，和 S 总趋于确定的极限 I，那么称这个极限 I 为函数 f(x) 在区间 [a, b] 上的定积分（简称积分），记作 ∫(a到b) f(x)dx。
""",
    NoteType.ENGLISH: """Unit 1: How to Improve Your Study Habits
Text A

Maybe you are an average student with an average intellect. You pass most of your subjects. You occasionally get good grades, but they are usually just average. You think you will never be a top student. But you can be better if you want to. Yes, even students with average intellect can be top students without additional work. Here's how:

1. Plan your time carefully. When you plan your week, you should make a list of things that you have to do. After making this list, you should make a schedule of your time. First, fill in the time for eating, sleeping, dressing, etc. Then decide on a good, regular time for studying. Don't forget to set aside enough time for entertainment. This weekly schedule may not solve all of your problems, but it will make you more aware of how you spend your time. Furthermore, it will enable you to plan your activities so that you have adequate time for both work and play.

2. Find a good place to study. Look around the house for a good study area. Keep this space, which may be a desk or simply a corner of your room, free of everything but study materials. No games, radios, or television! When you sit down to work, concentrate on the subject.

3. Make good use of your time in class. Listening to what the teacher says in class means less work later. Sit where you can see and hear well. Take notes to help you remember what the teacher says.

4. Study regularly. Go over your notes as soon as you can after class. Review important points mentioned in class as well as points you remain confused about. If you know what the teacher will discuss the next day, skim and read that material too. This will help you understand the next class. If you review your notes and textbook regularly, the material will become more meaningful and you will remember it longer.

5. Develop a good attitude about tests. The purpose of a test is to show what you have learned about a subject. The world won't end if you don't pass a test, so don't worry excessively about a single test. Tests provide grades, but they also let you know what you need to spend more time studying, and they help make your new knowledge permanent.

There are other techniques that might help you with your studying. Only a few have been mentioned here. You will probably discover many others after you have tried these. Talk with your classmates about their study techniques. Share with them some of the techniques you have found to be helpful. Improving your study habits will improve your grades.

Vocabulary:
- average (adj.): 平均的，普通的
- intellect (n.): 智力，才智
- additional (adj.): 额外的，附加的
- schedule (n.): 时间表，计划表
- entertainment (n.): 娱乐
- adequate (adj.): 充足的，适当的
- concentrate (v.): 集中，专心
- regularly (adv.): 定期地，有规律地
- review (v.): 复习，回顾
- confused (adj.): 困惑的，混乱的
- skim (v.): 浏览，略读
- attitude (n.): 态度
- excessively (adv.): 过度地，过分地
- permanent (adj.): 永久的，持久的
- technique (n.): 技巧，技术

Grammar: Conditional Sentences
1. Zero conditional (if + present simple, present simple)
- Used for general truths or habits
Example: If you heat water to 100 degrees Celsius, it boils.

2. First conditional (if + present simple, will + base verb)
- Used for possible future situations
Example: If you study hard, you will pass the exam.

3. Second conditional (if + past simple, would + base verb)
- Used for hypothetical or unlikely situations
Example: If I won the lottery, I would travel around the world.

4. Third conditional (if + past perfect, would have + past participle)
- Used for regrets or hypothetical past situations
Example: If I had studied harder, I would have passed the exam.

Unit 2: The Dinner Party
Text A

The country is India. A colonial official and his wife are giving a large dinner party. They are seated with their guests - officers and their wives, and a visiting American naturalist - in their dining room, which has a bare marble floor, open rafters and wide glass doors opening onto a veranda.

A spirited discussion springs up between a young girl who says that women have outgrown the jumping-on-a-chair-at-the-sight-of-a-mouse era and a major who says that they haven't.

"A woman's reaction in any crisis," the major says, "is to scream. And while a man may feel like it, he has that ounce more of control than a woman has. And that last ounce is what really counts."

The American does not join in the argument but watches the other guests. As he looks, he sees a strange expression come over the face of the hostess. She is staring straight ahead, her muscles contracting slightly. She motions to the native boy standing behind her chair and whispers something to him. The boy's eyes widen: he quickly leaves the room.

Of the guests, none except the American notices this or sees the boy place a bowl of milk on the veranda just outside the open doors.

The American comes to with a start. In India, milk in a bowl means only one thing - bait for a snake. He realizes there must be a cobra in the room.
""",
    NoteType.PHYSICS: """第一章 质点运动学
1.1 参考系与坐标系
一、参考系
自然界中所有的物体都在不停地运动着，绝对静止的物体是不存在的。在观察一个物体的位置及位置的变化时，总是要选取其他物体作为参考，被选作参考的物体称为参考系。

参考系的选择是任意的，但在讨论问题时，参考系一经选定，就不能再随意变动。选择不同的参考系，对同一物体运动的描述是不同的。例如，在地面上观察行驶的汽车，汽车是运动的；但在汽车中的乘客看来，汽车是静止的，而窗外的树木在向后运动。

二、坐标系
为了定量地描述物体的位置及位置的变化，需要在参考系上建立适当的坐标系。坐标系的选择也是任意的，主要取决于问题的性质和研究的方便。常用的坐标系有：

1. 直角坐标系（笛卡尔坐标系）
在参考系上取一固定点 O 作为原点，过 O 点作三条互相垂直的坐标轴，分别称为 x 轴、y 轴、z 轴。空间中任意一点 P 的位置可用三个坐标 (x, y, z) 来确定。

2. 极坐标系
在平面上，取固定点 O 为极点，从 O 出发引一条射线 Ox 作为极轴。平面上任意一点 P 的位置可用极径 r（P 到 O 的距离）和极角 θ（从 Ox 到 OP 的角度）来确定。

3. 自然坐标系
当质点沿已知轨迹运动时，可在轨迹上任选一点 O 作为原点，规定沿轨迹的某一方向为正方向。质点的位置可用从 O 到质点位置的弧长 s 来确定，s 称为弧坐标。

1.2 质点的位置矢量与位移
一、位置矢量
为了表示质点在空间的位置，我们从坐标系的原点 O 向质点所在位置 P 引一矢量 r，这个矢量就叫做质点的位置矢量，简称位矢。

在直角坐标系中，位矢 r 可表示为：
r = x(t)i + y(t)j + z(t)k

其中 i、j、k 分别是沿 x、y、z 轴正方向的单位矢量。

位矢的大小为：
|r| = √(x² + y² + z²)

位矢的方向可用方向余弦表示：
cosα = x/r, cosβ = y/r, cosγ = z/r

其中 α、β、γ 分别是位矢与 x、y、z 轴正方向的夹角。

二、运动方程
质点运动时，其位置随时间而变化，即位矢 r 是时间的函数：
r = r(t) = x(t)i + y(t)j + z(t)k

这个方程叫做质点的运动方程。

运动方程的分量形式为：
x = x(t)
y = y(t)
z = z(t)

1.3 速度
一、平均速度
质点在 Δt 时间内的位移 Δr 与时间 Δt 的比值，叫做质点在这段时间内的平均速度，用 v̄ 表示：

v̄ = Δr / Δt

平均速度是矢量，其方向与位移 Δr 的方向相同。

平均速度的分量形式为：
v̄_x = Δx/Δt
v̄_y = Δy/Δt
v̄_z = Δz/Δt

二、瞬时速度
为了精确地描述质点在某一时刻 t 的运动快慢和方向，我们将 Δt 趋近于零，取平均速度的极限，这个极限就叫做质点在时刻 t 的瞬时速度，简称速度，用 v 表示：

v = lim(Δt→0) Δr/Δt = dr/dt

速度是矢量，其大小称为速率，方向沿轨迹在该点的切线方向。

在直角坐标系中，速度的分量形式为：
v = v_xi + v_yj + v_zk

其中：
v_x = dx/dt
v_y = dy/dt
v_z = dz/dt

速度的大小为：
v = √(v_x² + v_y² + v_z²)

1.4 加速度
一、平均加速度
设质点在时刻 t 的速度为 v，在时刻 t + Δt 的速度为 v'，则速度的增量 Δv = v' - v 与时间 Δt 的比值，叫做质点在这段时间内的平均加速度，用 ā 表示：

ā = Δv/Δt

平均加速度是矢量，其方向与速度增量 Δv 的方向相同。

二、瞬时加速度
为了精确地描述质点在某一时刻 t 的速度变化情况，我们将 Δt 趋近于零，取平均加速度的极限，这个极限就叫做质点在时刻 t 的瞬时加速度，简称加速度，用 a 表示：

a = lim(Δt→0) Δv/Δt = dv/dt = d²r/dt²

加速度是矢量，其方向为速度增量 Δv 的极限方向。加速度的方向一般与速度的方向不同。

在直角坐标系中，加速度的分量形式为：
a = a_xi + a_yj + a_zk

其中：
a_x = dv_x/dt = d²x/dt²
a_y = dv_y/dt = d²y/dt²
a_z = dv_z/dt = d²z/dt²

加速度的大小为：
a = √(a_x² + a_y² + a_z²)

第二章 牛顿运动定律
2.1 牛顿第一定律
一、牛顿第一定律
任何物体都保持静止或匀速直线运动的状态，直到其他物体的作用迫使它改变这种状态为止。

牛顿第一定律包含两个重要概念：
1. 惯性：物体保持原有运动状态不变的性质，称为惯性。因此，牛顿第一定律又称为惯性定律。
2. 力：力是使物体运动状态发生变化的原因，即力是产生加速度的原因。

二、惯性系
牛顿第一定律定义了一种特殊的参考系，在这种参考系中，一个不受力作用的物体将保持静止或匀速直线运动状态。这种参考系称为惯性参考系，简称惯性系。

实验表明，在一般精度范围内，地球是一个近似的惯性系。相对于地球作匀速直线运动的参考系也是惯性系，而相对于地球作加速运动的参考系则不是惯性系。

2.2 牛顿第二定律
一、牛顿第二定律
物体受到外力作用时，所获得的加速度的大小与合外力的大小成正比，与物体的质量成反比；加速度的方向与合外力的方向相同。

数学表达式：F = ma

在国际单位制（SI）中，力的单位是牛顿（N），质量的单位是千克（kg），加速度的单位是米/秒²（m/s²）。

牛顿第二定律的分量形式：
在直角坐标系中：
F_x = ma_x
F_y = ma_y
F_z = ma_z

在自然坐标系中：
F_t = ma_t = m dv/dt
F_n = ma_n = m v²/R
""",
    NoteType.HISTORY: """第一章 反对外国侵略的斗争
第一节 资本-帝国主义对中国的侵略
一、军事侵略
资本-帝国主义列强对中国的侵略，首先和主要的是进行军事侵略。它们依仗先进的武器和军事技术，或者进行武力威胁，或者发动侵略战争，或者武装干涉中国的内政，直至出兵镇压中国革命。这种军事侵略是逐步升级的，从骚扰、蚕食中国沿海、边疆，到割占中国大片领土，直至企图瓜分中国。

（一）发动侵略战争，屠杀中国人民
从1840年鸦片战争以来，资本-帝国主义列强发动了一次又一次的侵华战争。在历次侵华战争中，外国侵略者屠杀了大批中国人民，制造了一系列骇人听闻的惨案。

1894年11月，日军制造了旅顺大屠杀惨案，4天之内连续屠杀中国居民2万余人。
1900年，俄国入侵中国东北时，制造了江东六十四屯惨案，数千中国居民遇难。
1900年，八国联军在侵华战争中，烧杀抢掠，无恶不作，仅在庄王府一处，就烧死和杀死义和团团民与平民1700多人。

（二）侵占中国领土，划分势力范围
资本-帝国主义列强通过侵略战争和进行武力威胁等，割占中国大片领土，强占中国租界，强租中国港湾，并且在中国划分势力范围，严重破坏了中国的领土主权完整。

1842年，英国强迫清政府签订《南京条约》，割让香港岛。
1860年，英国通过《北京条约》，割让九龙半岛南端和昂船洲。
1849年，葡萄牙武力强占澳门半岛。
1887年，签订《中葡和好通商条约》，允许葡萄牙"永居管理澳门"。

俄国利用英、法发动第二次鸦片战争之机，于1858年胁迫黑龙江将军奕山签订《瑷珲条约》，割去黑龙江以北60万平方公里领土。
1860年，通过签订中俄《北京条约》，割去乌苏里江以东40万平方公里领土。
1864年，强迫清政府签订《勘分西北界约记》，割去中国西北44万平方公里领土。
1881年，通过《改订伊犁条约》和5个勘界议定书，割去中国西北7万多平方公里领土。

通过这一系列不平等条约，俄国共侵占中国领土150多万平方公里。

1895年，日本强迫清政府签订《马关条约》，割去中国台湾全岛及所有附属各岛屿和澎湖列岛。

1898年，德国强租山东的胶州湾，把山东划为其势力范围。
沙俄强租辽东半岛的旅顺口、大连湾及其附近海面，以长城以北为其势力范围。
英国强租山东的威海卫和香港岛对岸的九龙半岛界限街以北、深圳河以南及附近的岛屿（新界），以长江流域为其势力范围。
1899年，法国强租广东的广州湾及其附近水面，把广东、广西、云南作为其势力范围。
日本也声明把福建作为其势力范围。

（三）勒索赔款，抢掠财富
资本-帝国主义列强发动战争来侵略中国、屠杀中国人民，却要中国人民加倍地承担其战争费用。它们向中国勒索巨额赔款，造成中国严重的财政危机，直接破坏和阻碍中国的经济发展。

鸦片战争期间，英国侵略者强迫清朝地方政府缴纳广州赎城费600万元（银元）。
《南京条约》规定赔款2100万元。
《北京条约》规定赔偿英法军费各800万两白银。
《马关条约》规定赔偿日本军费2亿两白银，加上赎辽费3000万两，共计2.3亿两白银。
《辛丑条约》规定赔款4.5亿两白银，分39年还清，本息合计达9.8亿两白银。

列强还在侵华战争中公开抢劫中国的财富，肆意破坏中国的文物和古迹。
1860年，英法联军火烧圆明园，将园内珍宝抢劫一空后纵火焚烧，使这座世界名园化为一片废墟。
八国联军侵华期间，对颐和园等皇家园林也进行了洗劫。

二、政治控制
为了统治中国，资本-帝国主义列强在政治上采取的主要方式，是控制中国政府，操纵中国的内政、外交，把中国当权者变成自己的代理人和驯服工具。

（一）控制中国的内政、外交
资本-帝国主义列强对中国政治的控制是逐步实现的。

在鸦片战争时期，外国侵略者还只是通过中国内部的妥协派贵族大臣如琦善等人来对清政府施加压力和影响。清王朝的许多大权仍掌握在主张抵抗的贵族大臣手中。

第二次鸦片战争期间，英法联军先后攻陷大沽、天津，1860年10月攻进北京，火烧圆明园，清政府被迫签订《天津条约》、《北京条约》。通过这些不平等条约，外国公使可以常驻北京，而外国公使在北京是以战胜者的姿态出现的，他们不是普通的外交官，而是清政府的"太上皇"。

资本-帝国主义列强在中国还享有领事裁判权。1843年中英《五口通商章程》规定，在通商口岸，中国人如与英侨"遇有交涉诉讼"，英国领事有"查察"、"听诉"之权，"其英人如何科罪，由英国议定章程、法律，发给管事官照办"。

（二）镇压中国人民的反抗
资本-帝国主义列强还勾结清政府镇压中国人民的反侵略反封建斗争。

为了镇压太平天国农民起义，他们不但向清政府供应军火、船只，而且派外国军官组织并指挥"洋枪队"，甚至直接动用陆海军，对太平军作战。

1870年发生天津教案后，法、英、美、俄、德、比、西等国联合向清政府提出抗议，并调遣军舰到天津海口及烟台一带示威。清政府不得不把爱国官吏和群众杀的杀，判的判，以讨好外国侵略者。

第二节 抵御外国武装侵略、争取民族独立的斗争
一、反抗外来侵略的斗争历程
资本-帝国主义侵略、压迫中国人民的过程，同时也是中国人民反抗它们的侵略、压迫的过程。救亡图存，成了一代又一代中国人面临的神圣使命。为了捍卫民族生存的权利，他们在长时间里进行了不屈不挠、再接再厉的英勇斗争。

（一）人民群众的反侵略斗争
鸦片战争时期，中国人民便掀起了反对外来侵略的斗争。

1841年5月，英军在广州郊区三元里一带的淫掠暴行，激起当地乡民的义愤，"不呼而集者数万人"，与英军展开激烈战斗。三元里人民的抗英斗争，是中国近代史上中国人民第一次大规模的反侵略武装斗争，显示了中国人民不甘屈服和敢于斗争的英雄气概。

太平天国农民战争后期，太平军曾多次重创英、法侵略军和外国侵略者指挥的洋枪队"常胜军"、"常捷军"。

（二）爱国官兵的反侵略斗争
在历次反抗外国侵略的战争中，爱国官兵表现了英勇顽强的战斗精神，并在一些战役中取得了胜利。

鸦片战争期间，1841年2月，广东水师提督关天培率部在虎门与英国侵略者激战，壮烈牺牲。
1842年6月，江南提督陈化成率部坚守吴淞炮台，与英国侵略军激战数小时，身负重伤仍坚持指挥战斗，直至壮烈牺牲。
1842年7月，副都统海龄率部在镇江与英国侵略军殊死战斗，城陷后自杀殉国。

中法战争期间，1885年3月，年近70的老将冯子材率部在镇南关（今友谊关）大败法军，取得镇南关大捷。这是中国近代史上反侵略战争中一次重大的胜利。

中日甲午战争期间，爱国将士浴血奋战，表现了崇高的爱国主义精神。
1894年9月，黄海海战中，北洋舰队致远舰管带邓世昌在军舰受伤、弹药用尽的情况下，下令开足马力撞击日舰吉野，不幸被鱼雷击中，邓世昌与全舰官兵250余人壮烈殉国。
"""
}

class StoryNode:
    def __init__(self, node_id: int, description: str, progress: int, 
                 is_choice: bool = False, options: List[Dict] = None,
                 next_node: int = None, is_ending: bool = False,
                 ending_type: str = None):
        self.node_id = node_id
        self.description = description
        self.progress = progress
        self.is_choice = is_choice
        self.options = options or []
        self.next_node = next_node
        self.is_ending = is_ending
        self.ending_type = ending_type

STORY_NODES = {
    0: StoryNode(
        node_id=0,
        description="你正在数学课上，老师在讲台上滔滔不绝地讲着微积分。你看了看窗外，今天的天气真好，可你却被困在这无聊的教室里。你的目光转向了教室的后门...",
        progress=0,
        next_node=1
    ),
    1: StoryNode(
        node_id=1,
        description="你假装认真地记着笔记，心里却在盘算着如何溜出去。离下课还有30分钟，但你已经迫不及待了。你注意到老师正在黑板上写板书，这可能是个机会...",
        progress=5,
        next_node=2
    ),
    2: StoryNode(
        node_id=2,
        description="你正在思考对策，突然听到老师说：'接下来请一位同学上来做这道题。'你的心猛地一跳，老师的目光似乎在向你这边扫来...",
        progress=10,
        is_choice=True,
        options=[
            {"key": "1", "text": "假装捡笔，弯腰躲避老师的目光", "next_node": 3},
            {"key": "2", "text": "举手上厕所，趁机离开教室", "next_node": 4}
        ]
    ),
    3: StoryNode(
        node_id=3,
        description="你迅速弯腰，假装在地上捡笔。老师的目光从你身上掠过，点了坐在你前面的同学。你松了一口气，但知道这招不能用太多次...",
        progress=15,
        next_node=5
    ),
    4: StoryNode(
        node_id=4,
        description="你果断举手：'老师，我想去厕所。'老师皱了皱眉，但还是点了点头。你强装镇定地走出教室，感觉自由就在眼前！",
        progress=20,
        next_node=6
    ),
    5: StoryNode(
        node_id=5,
        description="你继续假装记笔记，但心中更加急切了。那位同学在黑板上做题，老师正在耐心指导。后门似乎没有关紧，一道微风吹过，吹动了窗帘...",
        progress=18,
        is_choice=True,
        options=[
            {"key": "1", "text": "趁老师不注意，悄悄向后门移动", "next_node": 6},
            {"key": "2", "text": "继续等待更好的机会", "next_node": 7}
        ]
    ),
    6: StoryNode(
        node_id=6,
        description="你成功离开了教室！走廊里很安静，看起来大家都在上课。你快速向楼梯口走去，但你听到了脚步声——是教导主任！",
        progress=25,
        is_choice=True,
        options=[
            {"key": "1", "text": "躲进旁边的男/女厕所", "next_node": 8},
            {"key": "2", "text": "假装去教务处送作业", "next_node": 9}
        ]
    ),
    7: StoryNode(
        node_id=7,
        description="你决定继续等待。那个同学终于做完了题，走回了座位。老师开始讲解这道题，似乎比刚才更加投入。你觉得自己快要按捺不住了...",
        progress=22,
        next_node=10
    ),
    8: StoryNode(
        node_id=8,
        description="你冲进厕所隔间，屏住呼吸。教导主任的脚步声越来越近，然后停住了。你听到他在门外打电话：'是的校长，我在检查...'几分钟后，脚步声远去了。",
        progress=35,
        next_node=11
    ),
    9: StoryNode(
        node_id=9,
        description="你硬着头皮向教导主任走去，手里拿着刚从书包里掏出来的作业本。'主任好，我去教务处送作业。'教导主任看了你一眼，点了点头。你快步走过，后背已经湿透了。",
        progress=30,
        next_node=12
    ),
    10: StoryNode(
        node_id=10,
        description="你再也等不了了。你注意到老师正背对着全班同学在黑板上写着什么。这是千载难逢的机会！你的心跳加速，准备行动...",
        progress=28,
        is_choice=True,
        options=[
            {"key": "1", "text": "直接向后门冲刺", "next_node": 13},
            {"key": "2", "text": "从侧门慢慢溜出去", "next_node": 6}
        ]
    ),
    11: StoryNode(
        node_id=11,
        description="你从厕所出来，走廊里空无一人。你快速走向楼梯口，下楼时遇到了几个正在打闹的低年级学生。他们没有注意到你。你走到了一楼大厅...",
        progress=45,
        next_node=14
    ),
    12: StoryNode(
        node_id=12,
        description="你快步走下楼梯，来到一楼。教学楼的大门就在前方，但门口有个保安正在看报纸。你需要想办法出去...",
        progress=40,
        is_choice=True,
        options=[
            {"key": "1", "text": "直接走出去，假装是值日生", "next_node": 14},
            {"key": "2", "text": "从侧门绕出去", "next_node": 15}
        ]
    ),
    13: StoryNode(
        node_id=13,
        description="你不顾一切地冲出教室，但刚跑到走廊就被闻讯赶来的班主任抓住了。'逃课？跟我去教务处！'你的越狱计划失败了。",
        progress=25,
        is_ending=True,
        ending_type="caught"
    ),
    14: StoryNode(
        node_id=14,
        description="你深吸一口气，整理了一下衣服，故作镇定地向大门走去。保安抬头看了你一眼，你挤出一个微笑。'同学，现在是上课时间，你去哪里？'保安问道...",
        progress=50,
        is_choice=True,
        options=[
            {"key": "1", "text": '假装痛苦："我肚子疼，要去医务室"', "next_node": 16},
            {"key": "2", "text": '理直气壮："我是学生会的，有急事"', "next_node": 17}
        ]
    ),
    15: StoryNode(
        node_id=15,
        description="你绕到教学楼的侧门，这里平时很少有人经过。门没锁！你小心翼翼地推开门，外面就是停车场。你成功地走出了教学楼！",
        progress=55,
        next_node=18
    ),
    16: StoryNode(
        node_id=16,
        description="你捂着肚子，眉头紧皱，表情痛苦不堪。保安被你的演技骗到了，连忙说：'那快去快去，需要我扶你吗？'你摇摇头，快步走出了大门。",
        progress=60,
        next_node=18
    ),
    17: StoryNode(
        node_id=17,
        description="你的谎言被揭穿了。保安把你送到了教务处，你的班主任很快就赶来了。逃课记过处分是免不了的。你的越狱计划彻底失败了。",
        progress=45,
        is_ending=True,
        ending_type="caught"
    ),
    18: StoryNode(
        node_id=18,
        description="你终于走出了教学楼！阳光洒在你身上，你感到前所未有的自由。但学校大门还在前方，那里肯定有更多保安。你环顾四周，发现有几条路可选...",
        progress=65,
        is_choice=True,
        options=[
            {"key": "1", "text": "从学校正门走出去，假装是放学", "next_node": 19},
            {"key": "2", "text": "翻墙", "next_node": 20},
            {"key": "3", "text": "从自行车道混出去", "next_node": 21}
        ]
    ),
    19: StoryNode(
        node_id=19,
        description="你走向学校正门，那里确实有几个保安在站岗。你故作镇定地继续走着，假装在打电话。一个保安拦住了你：'同学，现在还没到放学时间...'你该如何应对？",
        progress=70,
        is_choice=True,
        options=[
            {"key": "1", "text": '继续假装打电话："好好好，张老师我马上就到"', "next_node": 22},
            {"key": "2", "text": "拔腿就跑", "next_node": 23}
        ]
    ),
    20: StoryNode(
        node_id=20,
        description="你观察了一下围墙的高度，大概两米多，上面还有些铁丝网。你发现一个角落有棵大树，树枝似乎可以够到墙顶。但翻墙有风险...",
        progress=70,
        is_choice=True,
        options=[
            {"key": "1", "text": "借助大树翻墙", "next_node": 24},
            {"key": "2", "text": "还是另寻他路吧", "next_node": 21}
        ]
    ),
    21: StoryNode(
        node_id=21,
        description="你来到自行车道，这里是学生骑车进出的地方。现在正好是午休前的时间，有一些走读生正在骑车离校。你决定...",
        progress=68,
        is_choice=True,
        options=[
            {"key": "1", "text": "假装推车混在学生中走出去", "next_node": 25},
            {"key": "2", "text": "请求一个走读生带你出去", "next_node": 26}
        ]
    ),
    22: StoryNode(
        node_id=22,
        description="你一边假装打电话一边继续往外走，语气自然得连自己都相信了。保安将信将疑，但最终还是没有继续拦你。你就这样大摇大摆地走出了学校正门！",
        progress=85,
        next_node=27
    ),
    23: StoryNode(
        node_id=23,
        description="你逃跑失败，被保安抓住了。不仅如此，你的莽撞行为还引来了路人的围观。你的学校生涯将因为这次冲动而留下一个大大的污点。越狱失败！",
        progress=75,
        is_ending=True,
        ending_type="caught"
    ),
    24: StoryNode(
        node_id=24,
        description="你深吸一口气，开始攀爬那棵大树。树枝虽然有点摇晃，但还算结实。你爬到足够高的位置，纵身一跃抓住了墙头。铁丝网有点刮人，但你顾不得那么多，翻了过去...",
        progress=90,
        next_node=27
    ),
    25: StoryNode(
        node_id=25,
        description="你快步走进自行车道，随手从一排自行车中推出一辆（你假装是自己的），混在走读生中向外走去。保安只是例行公事地看了一眼，没有注意到你。你就这样顺利地出去了！",
        progress=85,
        next_node=27
    ),
    26: StoryNode(
        node_id=26,
        description='你拉住一个正在骑车的男生："同学，能不能带我一程？我有点急事。"那个男生犹豫了一下，但看到你焦急的表情，还是同意了。你坐在后座上，顺利地通过了校门！',
        progress=80,
        next_node=27
    ),
    27: StoryNode(
        node_id=27,
        description="恭喜你！你成功逃脱了！这次惊险的越狱经历将成为你学生时代最刺激的回忆之一。但记住，逃课可不是好习惯哦。下次可别这么做了！",
        progress=100,
        is_ending=True,
        ending_type="success"
    )
}

class GameState(BaseModel):
    note_type: str = NoteType.MATH
    current_node: int = 0
    note_index: int = 0
    keystroke_count: int = 0
    choice_made: bool = False
    game_ended: bool = False

class StartGameRequest(BaseModel):
    note_type: str

class GameUpdateResponse(BaseModel):
    text_to_add: str
    progress_description: str
    progress_percent: int
    is_choice: bool
    options: List[Dict[str, Any]]
    is_ending: bool
    ending_type: Optional[str]
    game_ended: bool

active_sessions: Dict[str, GameState] = {}

@app.get("/")
async def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Index file not found"}

@app.get("/api/note-types")
async def get_note_types():
    return {
        "note_types": [
            {"id": NoteType.MATH, "name": "高等数学"},
            {"id": NoteType.ENGLISH, "name": "大学英语"},
            {"id": NoteType.PHYSICS, "name": "大学物理"},
            {"id": NoteType.HISTORY, "name": "中国近现代史"}
        ]
    }

@app.post("/api/start-game")
async def start_game(request: StartGameRequest):
    if request.note_type not in NOTE_CONTENTS:
        raise HTTPException(status_code=400, detail="Invalid note type")
    
    import uuid
    session_id = str(uuid.uuid4())
    
    active_sessions[session_id] = GameState(
        note_type=request.note_type,
        current_node=0,
        note_index=0,
        keystroke_count=0,
        choice_made=False,
        game_ended=False
    )
    
    return {
        "session_id": session_id,
        "initial_node": STORY_NODES[0].description,
        "progress": 0
    }

@app.post("/api/keystroke/{session_id}")
async def process_keystroke(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = active_sessions[session_id]
    
    if state.game_ended:
        return GameUpdateResponse(
            text_to_add="",
            progress_description="游戏已结束",
            progress_percent=100,
            is_choice=False,
            options=[],
            is_ending=True,
            ending_type=None,
            game_ended=True
        )
    
    current_node = STORY_NODES[state.current_node]
    
    if current_node.is_choice and not state.choice_made:
        return GameUpdateResponse(
            text_to_add="",
            progress_description=current_node.description,
            progress_percent=current_node.progress,
            is_choice=True,
            options=current_node.options,
            is_ending=False,
            ending_type=None,
            game_ended=False
        )
    
    state.keystroke_count += 1
    chars_per_keystroke = 3
    
    note_content = NOTE_CONTENTS[state.note_type]
    text_to_add = ""
    
    end_index = min(state.note_index + chars_per_keystroke, len(note_content))
    if state.note_index < len(note_content):
        text_to_add = note_content[state.note_index:end_index]
        state.note_index = end_index
    
    progress_thresholds = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    node_progress_map = {
        50: 1, 100: 2, 150: 5, 200: 10, 250: 14,
        300: 18, 350: 19, 400: 27
    }
    
    if not current_node.is_choice and not current_node.is_ending:
        for threshold, target_node in sorted(node_progress_map.items()):
            if state.keystroke_count >= threshold and state.current_node < target_node:
                if not STORY_NODES[state.current_node].is_choice and not STORY_NODES[state.current_node].is_ending:
                    if STORY_NODES[state.current_node].next_node:
                        state.current_node = STORY_NODES[state.current_node].next_node
                        current_node = STORY_NODES[state.current_node]
                        break
    
    if current_node.is_ending:
        state.game_ended = True
        return GameUpdateResponse(
            text_to_add=text_to_add,
            progress_description=current_node.description,
            progress_percent=current_node.progress,
            is_choice=False,
            options=[],
            is_ending=True,
            ending_type=current_node.ending_type,
            game_ended=True
        )
    
    return GameUpdateResponse(
        text_to_add=text_to_add,
        progress_description=current_node.description,
        progress_percent=current_node.progress,
        is_choice=current_node.is_choice and not state.choice_made,
        options=current_node.options if (current_node.is_choice and not state.choice_made) else [],
        is_ending=False,
        ending_type=None,
        game_ended=False
    )

@app.post("/api/make-choice/{session_id}")
async def make_choice(session_id: str, choice_key: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = active_sessions[session_id]
    current_node = STORY_NODES[state.current_node]
    
    if not current_node.is_choice:
        raise HTTPException(status_code=400, detail="Current node is not a choice node")
    
    for option in current_node.options:
        if option["key"] == choice_key:
            state.current_node = option["next_node"]
            state.choice_made = False
            
            next_node = STORY_NODES[state.current_node]
            
            return {
                "success": True,
                "next_description": next_node.description,
                "progress": next_node.progress,
                "is_ending": next_node.is_ending,
                "ending_type": next_node.ending_type,
                "is_choice": next_node.is_choice,
                "options": next_node.options
            }
    
    raise HTTPException(status_code=400, detail="Invalid choice key")

@app.post("/api/continue-story/{session_id}")
async def continue_story(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = active_sessions[session_id]
    current_node = STORY_NODES[state.current_node]
    
    if current_node.is_choice:
        state.choice_made = False
        return {
            "description": current_node.description,
            "progress": current_node.progress,
            "is_choice": True,
            "options": current_node.options
        }
    
    if current_node.next_node:
        state.current_node = current_node.next_node
        next_node = STORY_NODES[state.current_node]
        
        return {
            "description": next_node.description,
            "progress": next_node.progress,
            "is_choice": next_node.is_choice,
            "options": next_node.options,
            "is_ending": next_node.is_ending,
            "ending_type": next_node.ending_type
        }
    
    return {
        "description": current_node.description,
        "progress": current_node.progress,
        "is_choice": False,
        "options": []
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2833)
