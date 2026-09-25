# Stage143 实现说明与自查题（提交前供本人理解）

状态：**学习材料，不是最终报告，也不是测试集分数**。以已记录的
[Stage143 完整验证与资源证据](../results/stage143-evidence/final.json)为准。
现有候选尚未冻结、尚未对其计算完整测试集 BPB；仓库根目录的
`REPORT.pdf` 仍对应早期 Stage10，不可作为 Stage143 报告提交。

## 一句话说清任务与当前结果

给定每个独立的 256-token 输入窗口，模型在每个位置输出对**下一个**
token 的 2,048 维归一化概率。课程按固定分词器和评测器计算
`BPB = 全部目标的负对数似然（nats）/ ln(2) / 原文 UTF-8 字节数`，
不是按 token 平均的交叉熵。Stage143 的完整**验证集**结果是
**1.399686162 BPB**，376,599 个目标；它低于历史合规模型，但仍未达到
我们自定的 1.35 目标。评测定义见 [`evaluate.py`](../evaluate.py)
与 [`common.py`](../common.py)，课程边界见 [`GUIDE.md`](../../GUIDE.md)。

## 从输入到概率，实际发生了什么

1. 固定 BPE-2048 把训练文本变成 token。评测时每个 256-token 窗口
   独立处理；最后不足一窗的输入补零，但补齐位置的目标被忽略。
   输入的第 `t` 位可用于预测其后一个 token；模型本身只拿到输入，
   不接收目标。见 [`common.py`](../common.py) 的 `windows`。
2. 冻结的 ONNX 特征图在 CPU FP32 上将 `[B,256]` 输入转为
   `[B,256,288]` 隐状态。其神经祖先是宽度 288、8 层、8 个注意力头
   的因果 Transformer：第 2/4/6/8 层换为 kernel=7 的门控因果深度
   卷积，其余层保留全局因果注意力。ONNX 只存特征提取器的权重；
   输出头及下述其他参数存 checkpoint，避免双份主干权重。见
   [`student_hybrid_conv_structured.py`](../student_hybrid_conv_structured.py)
   与 [`student_stage143_openvino_singlepass.py`](../student_stage143_openvino_singlepass.py)。
3. 线性输出头对 2,048 个词给出 logits，按固定温度及训练文本的一元
   先验校准后做 softmax。另一个“窗口内复制”头在**当前位置及此前**
   的输入 token 上做注意力，再将注意力质量累加到对应词 ID；上三角
   mask 阻止它读取更晚的输入位置。神经词表分布与复制分布由学到的
   copy gate 混合。见 [`student_stage105_gated_singlepass.py`](../student_stage105_gated_singlepass.py)。
4. 另一路是从**提供的训练文本**统计得到的、剪枝的六阶改进
   Kneser–Ney（MKN）模型。它用历史 token 匹配稀疏表，把折扣后的
   已见续词质量与回退质量递归组合；未见历史继续回退至训练文本的
   continuation unigram。训练表构建见
   [`build_kneser_ney.py`](../scripts/build_kneser_ney.py) 与
   [`build_stage73_order6.py`](../scripts/build_stage73_order6.py)，单次稀疏
   遍历的推理实现见 [`student_stage105_gated_singlepass.py`](../student_stage105_gated_singlepass.py)。
5. 最后一层门控取四个只由当前输入及两路预测构成的特征：神经/复制
   分布最高 log 概率、前两名差距、最高匹配阶的 MKN 回退系数、该阶
   最大续词质量。标准化后线性组合，经 sigmoid 得到 MKN 权重 `w`；
   结果是 `(1-w)·p_neural+copy + w·p_MKN`，然后取对数。
   `w` 被限制在 `(0,1)` 内以维持有限概率。最后的标量/特征选择有
   验证集选择成分，不能称为纯训练集拟合。见
   [`student_stage103_gated.py`](../student_stage103_gated.py) 与
   [`student_stage105_gated_singlepass.py`](../student_stage105_gated_singlepass.py)。

## 训练和证据应怎样解释

- 这不是“训练一个 7,200-step 模型”的简单故事。Stage54 的同目标
  对照验证了 R-Drop 和混合卷积/注意力主干；后续继续训练、输出偏置、
  count-aware 调整、蒸馏以及训练派生 MKN/门控依次形成 Stage105 预测器。
  Stage143 **只更换推理封装**，并未重新训练。可追溯的已接受神经血统
  至少累计 255,225,110 次主要训练目标呈现、7,754.94 秒记录的训练
  时间；它不是全部搜索或总计算量。详见
  [`lineage-cost.json`](../results/stage143-evidence/lineage-cost.json)
  与 [`REPORT_STAGE143_DRAFT.md`](../../REPORT_STAGE143_DRAFT.md)。
- Stage54 与同目标 Stage47 的完整验证分数分别为 1.428594045 和
  1.450613003 BPB，但这个对照测的是**主干容量与卷积位置的组合改变**，
  不能单独归因为卷积。Stage143 与 Stage54 的差距又含额外训练预算、
  MKN、门控及蒸馏，不能声称是等预算单因素提升。
- Stage175 的 **1.300186476 BPB** 是逐目标使用正确答案选专家的
  hindsight oracle，只是潜力诊断；它使用了评测目标，**不能部署、不能
  提交，也不能写成合法验证成绩**。后续因果门控试验未兑现这段差距。
  见 [`README.md`](../../README.md) 的 Stage175–176 记录。

## 为什么目前只称“Windows 资源合规候选”

冻结图和 checkpoint 分别为 31,805,041 与 23,824,895 字节；连同源码、
分词器和环境清单，保守计算的 18 项推理资产共 **55,810,412 字节**，
低于 64 MiB。Windows 四线程、三次独立进程的 CPU FP32 同机比较，
候选/基线耗时中位数为 **3.617702 倍**，最大峰值 RSS 为
**2,176,729,088 字节**，分别低于 5 倍和 4 GiB。完整验证、哈希及
因果/归一化冒烟测试见 [`final.json`](../results/stage143-evidence/final.json)。
但另一个 Linux 单线程主机测得 **5.502555 倍**，超出时间限制；
因此 Windows 过门不等于“任意课程 CPU 都过门”。最终仍须按评测环境
复核，不应隐去这个失败。

## 提交前请自己回答

1. 为什么 BPB 分母是原文 UTF-8 字节数，而不是 376,599 个目标？
2. 预测第 `t+1` 个 token 时，复制头能看见哪些输入位置？为什么
   MKN 匹配也不能跨越独立窗口？
3. MKN 的续词概率、回退系数与神经/复制概率如何保证混合后仍归一化？
4. 哪些数字是完整验证集成绩、哪些只是诊断上限或失败筛选？
5. Stage54 对照能证明什么、不能证明什么？Stage143 新增了训练吗？
6. 为什么 Windows 资源结果不能推断 Linux 或课程机器一定合规？
7. README 里怎样披露复用的基线和实质性 AI 协助？请确认你确实理解
   关键代码、选择与局限，而不只是保留这份说明。

冻结、一次完整测试、更新报告和提交的未完成步骤见
[`SUBMISSION_PORTAL_READINESS_20260926.md`](SUBMISSION_PORTAL_READINESS_20260926.md)。
