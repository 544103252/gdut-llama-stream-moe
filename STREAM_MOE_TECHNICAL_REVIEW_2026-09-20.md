# Stream MoE 技术总结与后续优化评估

审阅日期：2026-09-20。当前代码基准：`beea7749b`。对照基线：`0d265ac03`（8.31 初始 Stream MoE 与命中统计），首个 GPU planner 提交：`49adc34b4`（9.2）。

本文的架构审阅基准为上面的 commit。依据包括当前源码、Git 提交及本地 amend 快照、当前可见对话中的实验数据。后续精度问题排查的代码补丁和本地测试在本文中作为独立正确性事项记录，不混入此前的性能结论。不能把不可见的历史对话当作已恢复的完整实验日志；提交名称中的日期也不等于每一项功能的实际完成日期。

文中“已实现”表示代码存在，“历史实测”表示你提供过实验结果，“预期收益”表示机制分析，三者不混用。源码链接对应本机当前工作区，后续改动可能使行号变化。

## 一、从原始 Stream 到当前版本：原理、改动与效果

### 1.1 总体结论与演进脉络

主线确实是把 decode 热路径上的缓存决策从 CPU 移到 GPU，但后续优化不是简单地继续“多放一些工作到 GPU”，而是围绕两个不同问题展开：

- hit 路径：尽量避免主线程逐层等待、读取路由结果、执行缓存回调，再重新提交计算。
- miss 路径：尽早发现缺失，减少已经提交但无法有效执行的尾部工作，复用已录制命令，并缩短缺失专家准备时间。

当前结构可以概括为：**GPU 做缓存决策和映射，滚动提交限制无效前瞻，command buffer 支持恢复，CPU 负责缺失专家的读取与上传，常驻线程及可选 profiling 降低固定开销。**

| 阶段 / 快照 | 主要工作 | 在当前版本中的地位 |
| --- | --- | --- |
| `0d265ac03` | 原始 Stream 与命中统计基线 | 对照基线，不是无缓存的普通 llama.cpp |
| `49adc34b4` | GPU planner、图算子、后端接口、连续执行与相关实验机制 | GPU 决策主干保留 |
| `a175469c8` | command buffer 恢复、异步滚动提交及 tokenwise 评测等 | 当前执行主干 |
| 后续 amend，含 `025814436` | 并行权重加载及入队、唤醒、诊断统计的迭代 | 加载机制保留，部分统计已删除 |
| `da8512798` | 精简实验路径和统计 | 固定窗口及大量旧统计不再保留 |
| `9d600e193` | 滚动提交工作线程常驻化 | 已实现，不是未来待办 |
| `beea7749b` | profiling 按需启用、同步屏障补强、接口与配置清理 | 当前审阅对象 |

这些 amend 快照属于本地 reflog 演进，不是主分支上彼此独立的一串普通提交；分享或归档时应以明确的 commit hash 为准。

### 1.2 原始 Stream 已经解决了什么，瓶颈又在哪里

原始 Stream 并不是每次都从磁盘完整加载模型。它已有：

- 每层固定容量专家缓存，以及 expert ID 到缓存 slot 的映射。
- EMPTY、LOADING、RESIDENT 状态和淘汰策略。
- 路由热度衰减，以及热度相同时按 LRU 选择的策略。
- I/O worker、按需专家读取、上传与等待机制。
- 多 token prefill 的 wave 相关处理。

因此上述能力不能全部算作 9.2 之后新增加的功能。

原始 decode 的主要问题是：GPU 产生 selected expert IDs 后，CPU 需要参与缓存检查和 remap。即使专家全部 resident，也要经过这条主机参与路径。专家矩阵乘可能很快，但每层固定的调度和同步成本不会因为 hit 自动消失。

另一个容易混淆的对照是“不设置 `--moe-stream-cache`”。它通常不启用这里讨论的 streaming 缓存流程，不等于“原始 Stream 模式”。同样，普通全模型驻留与专门保留 Stream 路径的 full-cache test 不是同一个实验。

### 1.3 GPU planner：真正搬迁的是决策与映射，而不只是一个检查标志

主要源码入口：

- [算子构造 ggml_moe_stream_cache_decide](C:/Users/admin/Desktop/llama-stream-moe/ggml/src/ggml.c:6097)
- [模型图中连接 planner 的位置](C:/Users/admin/Desktop/llama-stream-moe/src/llama-graph.cpp:1998)
- [Vulkan 算子执行](C:/Users/admin/Desktop/llama-stream-moe/ggml/src/ggml-vulkan/ggml-vulkan.cpp:19053)
- [GPU planner shader](C:/Users/admin/Desktop/llama-stream-moe/ggml/src/ggml-vulkan/vulkan-shaders/moe_stream_cache_plan.comp)

`GGML_OP_MOE_STREAM_CACHE_DECIDE` 是图中的专门算子，不是仅添加一个 enum 或 switch case。构造函数将 selected experts 接为输入，保存 layer、专家总数和缓存 slot 数等参数，输出供后续专家计算使用的 slot IDs。完整支持还包括 backend 能力声明、图调度、Vulkan 执行和 shader 实现。

GPU 维护或访问的关键状态包括：

- `expert_map`：已提交的 resident 映射。
- `available_map`：包括规划中可用映射，用于区别已驻留与仍需等待。
- `slot_expert`、`slot_state`：slot 内容和状态。
- `hotness`、`last_use`、`use_counter`：缓存策略状态。
- `next_map`、`loads`、`mapped_topk`、`meta`：本次计划及结果。

hit 时不需要重新选择 victim：直接将 expert IDs 映射到 slots，更新相应策略状态，然后继续 GPU 计算。miss 时 planner 生成加载项和目标映射，CPU 执行实际 I/O。

shader 中保留了热度衰减、空 slot 优先、LOADING 排除、当前选择保护和 LRU tie-break 等策略。这里的目标是保持缓存策略语义，而不是靠更换专家、降低精度或近似计算来加速。

需要保留的边界：新路径实际落地在 Vulkan。通用接口的存在不意味着 CUDA、Metal 等自动获得同样能力；其他后端需要自己的实现和能力检查。原始 CPU 管理路径与新算子的其他后端支持也不能混为一谈。

### 1.4 meta、加载计划与正确性约束

`meta` 不是 GPU 主动“调用 CPU”的消息，而是 GPU 写入、经过正确同步后由 CPU 读取的一小段状态。shader 当前布局为：

| 下标 | 含义 |
| --- | --- |
| 0 | 状态 / 错误码 |
| 1 | 新加载专家数 |
| 2 | 去重后的 required 数 |
| 3 | 错误详情 |
| 4 | 已规划但尚需等待的数量 |
| 5 | selected 数量 |
| 6 | hit predicate |
| 7 | layer ID |

实际加载信息还在 `loads` 等缓冲区中，不是只靠一个 miss 位恢复执行。加载项包含 expert、victim、slot，另外有 required slots 等约束。CPU 在 [eval_callback](C:/Users/admin/Desktop/llama-stream-moe/src/llama-moe-stream.cpp:1211) 中验证计划、保留所需 slots、组织加载并 commit。

必须维持的顺序是：专家所有必要权重准备完成，才能把 slot 视为 RESIDENT 并允许相应计算消费。GPU 策略成为权威后，CPU 策略镜像可能过期，切换回 CPU 管理时需要同步，而不是让两套策略各自随意推进。

shadow 路径用于比较 GPU 与 CPU 规划，是验证搬迁语义的辅助机制，不应在正式测速中无意开启。它也不能替代最终输出正确性测试。

### 1.5 连续图：消除逐层主线程往返，但 miss 后仍要处理已提交工作

连续执行依赖 GPU predicate 和条件执行。正常 hit 时，各层 planner 与后续计算可以在设备侧衔接，不必每层返回主线程做缓存处理。

在较早的完整图模式中：

1. CPU 录制、提交图的相关命令，GPU 可以与 CPU 后续录制重叠执行。
2. GPU 某层 planner 发现 miss，写出计划和 miss 状态。
3. 后续受条件控制的有效计算被跳过；已提交命令仍需被处理到可同步边界。
4. CPU 同步后读取 miss 层和计划，完成加载、commit。
5. 恢复尚未完成的计算。

因此“连续图只提交一次”仅适合作为粗略的应用层描述。一个 graph compute 内部可能有多次 Vulkan queue submit；发生 miss 还会有恢复提交。不能把图调用次数直接等同于驱动提交次数。

同样，“跳过”不是免费跳转指令，也不是主机把 GPU 正在执行的队列瞬间撤销。被条件禁用的重计算可以不做，但后续命令、屏障、状态处理、部分拷贝和同步仍可能花时间。这解释了完整图在较低 hit 率时的历史问题。

### 1.6 固定窗口实验：它验证了尾部浪费的重要性，但当前已移除

固定窗口将执行拆为若干 MoE 层一组，窗口边界返 回主机。其折中是：

- 小窗口：miss 后要排空的已提交尾部短，但正常 hit 也有更多固定边界。
- 大窗口 / 完整图：边界少，但 miss 可能牵连更长的无效尾部。

这不代表窗口必然慢于连续图。真实速度取决于 hit 分布、miss 所在层、GPU 命令处理、CPU 录制速度、加载延迟等因素。

当前 `beea7749b` 不再解析 `LLAMA_MOE_STREAM_GPU_WINDOW`，因此旧的 WINDOW=4/8 命令不应再作为当前版本切换策略的有效操作。窗口数据仍有价值，但属于历史实验结果。

### 1.7 command buffer 复用：减少恢复时重复录制

主要入口：[continuous_resume](C:/Users/admin/Desktop/llama-stream-moe/ggml/src/ggml-vulkan/ggml-vulkan.cpp:19514)。

优化前的恢复可能需要重新遍历和录制剩余图。现在保留本次连续执行中已经形成的 command buffers，并记录 planner / 层边界；miss 处理完成后定位恢复位置，复用相应后缀。

这里必须区分三种复用：

- llama 层图结构复用：避免重新构造相同拓扑。
- Vulkan command buffer 在当前执行中的恢复复用：避免同一后缀重复录制。
- 跨 token 长期复用整套 Vulkan 命令：当前不能据此宣称已经实现。

滚动模式还可能在发现 miss 时停止录制，因此后缀并非总是全部存在：已录制部分可 replay，未录制部分仍需继续录制。

预期收益是降低 miss 的 CPU 恢复成本，不直接消除 SSD 读取、上传或已提交尾部的 GPU 成本。历史上 `resume_submit` 比 `skip_tail_gpu` 小得多，也说明不能把所有慢速现象归因于“CPU 再提交了一遍图”。

### 1.8 异步滚动提交：把固定窗口改成有界的前瞻流水线

核心入口：[rolling_worker_epoch](C:/Users/admin/Desktop/llama-stream-moe/ggml/src/ggml-vulkan/ggml-vulkan.cpp:15436)。

当前有两个不同角色：主调用线程录制图，rolling worker 提交就绪命令并观察 planner 完成情况。它不是“后台线程负责录制全部图”。

过程是：

1. 录制线程不断产生可提交的命令块。
2. worker 在 lookahead 预算内提交，利用 timeline semaphore 标记 planner 完成。
3. hit 时向前推进，并补充后续提交。
4. miss 时停止继续提交；通过原子停止标志使录制线程在检查点停止继续扩大无效后缀。
5. 同步必要的在途工作后，由 CPU 执行 miss 计划，再 replay / 继续录制。

lookahead 限制的是在途 planner 的前瞻范围，不是未来 token 数，也不等于旧版固定窗口层数。GPU 仍需实际执行 router 才知道该层选择哪些专家；rolling 不等于预测专家并提前把任意未来专家读入。

当前已经有两项重要细节：

- 将预算内已经就绪的多个普通提交项组成 `ready_batch`，而非一项必做一次独立提交。
- 等待最早的 planner 后，读取 timeline counter，顺便处理已经完成的后续 planner，避免对每个已完成项再次阻塞等待。

rolling 在高 hit 时保持流水线，在 miss 时限制已提交尾部，试图同时获得完整图与窗口的好处。但它仍有 worker、timeline、meta 读取和队列管理成本；hit 并不是完全没有 CPU 工作，只是不再逐层走主线程缓存回调。

### 1.9 常驻 rolling worker：最新代码已经消除了反复创建线程

入口：[worker 与 epoch 启停](C:/Users/admin/Desktop/llama-stream-moe/ggml/src/ggml-vulkan/ggml-vulkan.cpp:15558)。

最新实现将提交线程保留在 backend context 中，空闲时等待条件变量；每轮执行通过 epoch pending / done 状态唤醒和等待，而不是每次执行都重新创建、销毁线程。

其目标是减少固定的线程生命周期开销和调度抖动，对高 hit、I/O 较少时更有意义。它并没有取消同步：每轮仍必须等 worker 到达安全状态才能重置相关资源。当前没有单独的受控数据证明这项改动贡献多少 t/s。

### 1.10 parallel loading：把一个专家的多个权重片也并行起来

主要入口：[queue_load_locked](C:/Users/admin/Desktop/llama-stream-moe/src/llama-moe-stream.cpp:697)、[worker_loop](C:/Users/admin/Desktop/llama-stream-moe/src/llama-moe-stream.cpp:733)。

原有多线程按专家组织任务时，单次只缺一个专家，可能主要由一个 worker 顺序读取其多个权重张量。新模式在满足 Vulkan 缓存与多 I/O 线程条件时，把一个专家的不同权重片拆为多个任务。

后续细化包括：

- 在入队阶段直接拆分，不再等第一个 worker 接到专家任务后才派生其他片。
- 入队即设置 `slot_claimed`，避免重复认领。
- `slot_parts_pending` 记录未完成片数，全部完成后才转为 RESIDENT。
- `slot_gen` 等校验拒绝失效任务，错误走统一失败路径。
- 按任务数与线程数上限进行 `notify_one`，减少不必要的全体唤醒。
- Windows 下为并行 worker 使用独立文件句柄，避免共享文件 seek/read 锁抵消读取并行度。

这对“一次 miss 只缺 1 至 2 个专家，但每个专家包含多个权重片”的场景尤其有针对性。`decode-miss1`、`decode-miss2` 指新加载专家数为 1 或 2 的 miss 分类，不是两个执行阶段。

并行读取不等于 GPU 上传也完全并行。当前 worker 使用同步 `ggml_backend_tensor_set`；[Vulkan 上传实现](C:/Users/admin/Desktop/llama-stream-moe/ggml/src/ggml-vulkan/ggml-vulkan.cpp:8131) 对非 host-visible buffer 持有 device mutex，建立传输上下文、提交并等待 fence。因此读取可重叠，但上传可能串行化。host-visible buffer 又是另一条 memcpy 路径。

增加线程也不保证更快：磁盘并发、OS 文件缓存、内存带宽、上传锁与 CPU 调度都会限制收益。读取计时不能简单解释成物理 SSD 时间。

### 1.11 profiling 的演进：从解释现象到减少观测扰动

历史上逐步加入过 `skip_tail_gpu`、`resume_submit`、窗口 gap、混合口径 `initial_submit`、更细的提交与加载统计。它们帮助纠正了“连续图低 hit 主要慢在重复 CPU 提交”的初始假设。

但并非所有统计适合保留：旧 `initial_submit` 混合不同边界，不能作为纯固定提交成本；CPU 墙钟与 GPU timestamp、并行 worker 累计时间不能直接相加。最新版本已精简很多指标，并以 `LLAMA_MOE_STREAM_PROFILE` 控制统计工作。

最新改动不只是隐藏打印：scheduler 的计时、GPU timestamp query 的创建 / reset / 写入 / 读取及统计汇总都进行了按需控制。常规测速应关闭 profiling，诊断时再打开，以区分程序成本与观测成本。

当前保留指标的实际含义：

| 项目 | 当前含义和限制 |
| --- | --- |
| `segment_wall` | scheduler 记录的执行段墙钟时间，在慢速回调中累计；不是纯 GPU 时间，也不是全部 decode 时间 |
| `load_wait` | callback 中 `apply_plan_locked` 的墙钟跨度，包含组织加载与等待所需专家等工作，不是磁盘 read 独占时间 |
| `callback` | 从回调处理开始到 commit 调用返回附近的主机墙钟跨度，包含上述加载等待 |
| `skip_tail_gpu` | GPU miss 标记到选定尾部标记的 timestamp 差；完整图与 rolling 的尾部定义不同，不代表从 miss 到 CPU 恢复的全部延迟 |
| `resume_submit` | 从进入 backend resume 到恢复提交记录点的主机墙钟跨度；不包含此前 CPU 加载，不是剩余全部命令的 GPU 执行时间 |
| `record_tail` | 从首次底层提交返回到本轮图录制结束附近的 CPU 墙钟差；GPU 可以同时计算，不等于纯阻塞或纯 CPU 活跃时间 |

两个当前代码细节值得在读日志时牢记：

- `record_tail` 的结束时间取在等待 rolling epoch 完成之前，因此不覆盖后面的全部 worker 等待；它不是 token 末端延迟。
- 详细 `decode-slow` 输出被 `n_gpu_slow_plans > 0` 条件包围。完全无 miss 时没有这一行，不等于所有固定成本为零。

当前已经不输出旧版 `initial_submit`、`submit_front`、`sync_wait`、`window_gap_gpu` 等完整历史组合。不要拿旧字段公式直接解释最新日志。

### 1.12 同步屏障与精简工作不是“纯性能补丁”

最新 amendment 将条件执行前的可见性范围补强为同时覆盖 conditional predicate 读取与 compute shader 读取。这是正确性约束：前面 shader 写出的数据既可能作为条件，也可能供后续计算读取。

不能为了降低 barrier 数量直接撤销这类依赖。某个驱动上暂时没有出现错误，不代表可以省略同步；少量同步成本与数据正确性应分开讨论。

同一阶段还清理了旧窗口、debug / 诊断统计、旧统计 API 及重复配置读取等。精简减少了维护分支，也降低了部分运行时工作，但删除字段本身不必然带来可测的性能提升。

### 1.13 精度验证：tokenwise 是验证执行路径，不是修改评测定义

入口：[decode_helper](C:/Users/admin/Desktop/llama-stream-moe/tools/perplexity/perplexity.cpp:690)。

新增 `--tokenwise` 让相关评测调用按单 token 送入 `llama_decode`，保留位置、序列归属及需要的 logits，以便覆盖仅适用单 token 的 GPU decode 路径。它不等于简单把所有全局 batch 参数强设为 1；此前直接设置 batch 曾触发输出容量断言。

HellaSwag 的核心是对给定候选续写计算似然分数，再比较候选，不是先自由生成答案、再按相似度匹配四个选项。tokenwise 改变的是送入模型的粒度，不应改变评分目标。

你报告过连续图与原始 Stream 的 HellaSwag 结果一致。这支持“已测样本上没有观察到精度下降”，但不证明所有 logits bit-exact、所有模型和所有 cache miss 组合都正确。PPL / MMLU 讨论过测试方法，当前可见材料不足以把它们写成已经完成的回归结果。

更严格的对照需保证：双方都使用 tokenwise、同一模型与样本、相同序列状态，并确认连续路径确实启用。只看到相同总准确率或 tokenwise 提示，不能单独证明走到了目标 backend 路径。

### 1.14 已有性能证据：能说明什么，不能说明什么

以下是历史截图中的代表性数据，不是当前 HEAD 的新 benchmark。

220 缓存的一组实验：

| 模式 | decode t/s | hit rate | skip_tail_gpu ms/miss | resume_submit ms/miss |
| --- | ---: | ---: | ---: | ---: |
| 窗口 8 | 47.3 | 95.85% | 0.254 | 0.055 |
| 窗口 4 | 47.0 | 96.05% | 0.136 | 0.056 |
| 早期完整连续图 | 41.8 | 94.48% | 1.304 | 0.066 |
| 原始 Stream | 41.8 | 94.78% | 未提供 | 未提供 |

这组观察支持：窗口显著缩短了 miss 后的尾部处理；CPU resume 提交的每次差距小于 skip tail 的差距。窗口 8 相对这次原始结果约快 13.2%，但由于 hit rate、请求量等并不完全一致，这只是该组观测差值，不是严格隔离单项改动后的收益。

230 缓存的另一组历史结果：完整图 46.6 t/s、窗口 4 为 48.0 t/s、窗口 8 为 49.9 t/s；各自 hit rate 为 96.93%、97.29%、97.55%。完整图 skip 为 1.385 ms/miss，窗口分别为 0.147、0.245 ms/miss。方向相似，但仍不能把总性能差完全归因于一个字段。

240 缓存的五个观测值分别为：

- 窗口 2：39.9、40.4、47.3、48.2、47.9。
- 窗口 4：41.4、46.4、49.9、50.4、49.9。
- 窗口 8：41.2、47.9、51.0、51.6、50.3。

这些数据呈现窗口 4/8 相对窗口 2 的总体优势，但五个位置的实验含义没有完整记录，不能擅自解释成预热曲线或正式重复实验。

对最新实现，你报告在所测各缓存容量下相对原始 Stream 都有进步。可以把它记录为当前整体结果，但现有材料不足以分别给 command buffer、rolling、parallel loading、线程常驻化分配准确的收益百分比。

还需统一口径：当前 `[CACHE_EXPERT]` 的 hit/miss 是一层一次 token 选择是否全部 resident 的计数，不是每个被选专家单独计数；230/256 是容量比例，也不是实际 hit rate。不要将两者直接代换。

### 1.15 当前保留功能与使用边界

- `LLAMA_MOE_STREAM_GPU_DECODE`：请求 GPU decode planner。
- `LLAMA_MOE_STREAM_GPU_DECODE_CONTINUOUS`：请求连续路径，还受 backend 能力、单 token 路径及外部 callback 等条件限制。
- `LLAMA_MOE_STREAM_GPU_ROLLING`：正整数 lookahead，在连续路径中控制滚动前瞻；未开启 rolling 时保留非滚动连续执行。
- `LLAMA_MOE_STREAM_VK_PARALLEL_LOAD`：请求 Vulkan 并行权重片加载，需满足实际缓存 backend 和线程数条件。
- `LLAMA_MOE_STREAM_PROFILE`：启用当前统计。
- `LLAMA_MOE_STREAM_GROUP_CONDITIONAL`：在设备支持条件执行时，请求合并连续图中的条件执行区域；详见 1.23。
- `LLAMA_MOE_STREAM_NO_PRELOAD`：关闭既有 prefill wave 预加载，不是 rolling 的开关，也不是本轮新增优化。
- `LLAMA_MOE_STREAM_SHADOW`、`LLAMA_MOE_STREAM_FULL_CACHE_TEST`：诊断用途，不应无意混入正常对照。

当前大部分布尔环境变量按“是否存在”判断，因此设为字符串 `"0"` 仍可能开启；关闭应用 `Remove-Item Env:变量名 -ErrorAction SilentlyContinue`。ROLLING 是解析正整数的例外。DIRECT 的当前配置走命令行 / 参数，不应沿用已清理的旧环境变量说明。

此外，当前结构不支持多个 context 同时解码同一个 streamed model 并共享这套可变缓存状态；一个 context 的淘汰可能影响另一个在途图。封包文档应明确这些边界，而不是只提供开关列表。

### 1.16 一个具体的缓存决策例子：ID 没变，存储位置变了

下面用 8 个专家、4 个缓存 slot、每次选择 2 个专家的缩小模型说明。数字只用于解释，不是实际模型参数。

初始状态：

| slot | 缓存专家 | 状态 | 热度 | 最近使用先后 |
| --- | ---: | --- | ---: | --- |
| 0 | E1 | RESIDENT | 12 | 较新 |
| 1 | E3 | RESIDENT | 2 | 最旧 |
| 2 | E4 | RESIDENT | 8 | 较新 |
| 3 | E6 | RESIDENT | 2 | 比 E3 新 |

**情形 A：router 选择 `[E4, E1]`，全部 hit。**

1. planner 查 `expert_map`，得到 `[2, 0]`。
2. 更新相应专家热度和 slot 使用顺序。
3. 后续专家矩阵乘从 slot 2、slot 0 取权重。
4. 算的是原来的 E4 和 E1，不是改成“最接近的驻留专家”。

原始 Stream 在 CPU 上完成类似检查和 remap；新路径把它留在 GPU。这里节省的主要是逐层 CPU 参与，不是少算了一个专家。

**情形 B：router 选择 `[E4, E7]`，E7 缺失。**

1. E4 对应 slot 2，必须保留。
2. 其余候选中 E3 与 E6 热度最低，按较旧优先选择 E3，即 slot 1。
3. GPU 生成类似 `{expert=7, victim=3, slot=1}` 的加载项，以及映射 `[2, 1]`。
4. 但此时 slot 1 中还不是有效的 E7，不能马上开始该层专家计算。
5. CPU 按计划加载 E7 所需权重，所有片完成后才能发布 RESIDENT。
6. commit 更新 GPU 映射和 predicate，再恢复从对应位置往后的计算。

这解释了三个不同的“已经完成”：planner 完成、磁盘读取完成、专家可以被 GPU 安全使用，三者不是同一时刻。

**情形 C：E7 已经是 LOADING。**

它不应重复入队，也不能当作 resident hit。计划需要表达等待现有任务的需求。该情况不是“新加载数为 0，所以可以无条件继续”，还必须检查 waiting / required slots。

### 1.17 几种执行方式的前后对照：同一个第 10 层 miss

假设 40 个 MoE 层，层编号在本例中从 1 开始；第 10 层发生一次 miss，其他层 hit。为解释清楚暂不考虑额外 backend split。实际源码 layer ID 通常从 0 开始。

| 方式 | miss 前 | miss 发生时 | 恢复后 |
| --- | --- | --- | --- |
| 原始 Stream | 各层路由后 CPU 都参与缓存检查 | 在该层组织加载并等待 | 继续该层及后续计算，后续层仍有 CPU 检查 |
| 早期完整连续图 | GPU 连续推进，主线程不逐层处理 hit | 已提交后缀可能覆盖至第 40 层，需走到同步边界 | 加载后恢复；当前 command buffer 机制可复用已录制后缀 |
| 历史窗口 4 | 先后处理 1-4、5-8、9-12 | 无效尾部限制在当前窗口附近 | 恢复 10-12 的未完成部分，再处理后续窗口 |
| 当前 rolling | 在 lookahead 预算内边执行边补充提交 | 停止扩大前瞻，处理已经在途的有限尾部 | 已录制部分 replay，必要时接着录制剩余图 |

这里没有保证“rolling=2 就精确跳过两层”。在途 planner 数、无 planner 的尾块、CPU/GPU 进度以及发现 miss 的时刻都会影响实际尾部；lookahead 是预算，不是固定分组。

如果 miss 发生在第 39 层，完整图需要跳过的尾部本来就短，rolling 的主要优势可能不明显。如果 miss 很早、频繁发生，限制在途后缀就更重要。因此相同总体 hit rate，也可能因 miss 的层分布不同而有不同性能。

### 1.18 CPU、提交线程、GPU 与 I/O worker 各做什么

把“CPU”拆成角色，更容易理解流水线：

| 角色 | 正常 hit | 发现 miss 后 |
| --- | --- | --- |
| 主调用 / 录制线程 | 组织图、录制命令，和 GPU 执行可能重叠 | 在检查点停止扩大录制；进入加载计划处理及恢复流程 |
| rolling worker | 在预算内提交；等待 / 检查 timeline 和 meta | 停止补充提交，记录 miss，结束当前 epoch |
| GPU | 路由、planner、专家与其他模型计算 | predicate 阻止受保护计算；处理已经提交的命令到安全边界 |
| I/O worker 池 | 没有任务时等待，不等于空转消耗整个核心 | 读专家权重片并执行当前同步上传 API，汇报完成 |

“CPU 等待 GPU”通常表示主机线程阻塞在同步 API 上。它不是另一份独立于 GPU 运行时间的硬件工作，不应该把这段等待与其覆盖的 GPU 时间重复相加。

“CPU 录制与 GPU 同时运行”也不等于录制免费：只要 GPU 消耗 ready 工作比 CPU 生产得快，后续 GPU 就可能等命令；反过来，CPU 提前录完很多最终被 miss 禁用的工作，也会消耗 CPU 和驱动资源。

### 1.19 用一个数字例子解释为什么 record_tail 不是纯损失

下面完全是示意数据，用于说明计时边界，不是任何一轮实测：

```text
主机时钟： 0.0 ms  开始处理当前图
           0.3 ms  首次底层 queue submit 返回
           2.8 ms  当前图录制结束
           3.1 ms  当前 rolling epoch 等待完成
           8.0 ms  本轮所需 GPU 工作和输出同步完成

record_tail 约为 2.8 - 0.3 = 2.5 ms
```

GPU 可能在 0.3 ms 之前或之后开始执行，不能把 API 返回时刻当成准确的 GPU 起算时刻。GPU timestamp 又属于另一套时钟，不能未经校准直接与 CPU 时间戳相减。

若 GPU 从早期开始始终有工作可做，将 record_tail 从 2.5 ms 降为 2.0 ms，最终 8.0 ms 可能不变。如果其中存在 GPU 等待后续命令的空洞，缩短对应录制段才可能直接降低 token 延迟。

类似地，`skip_tail_gpu=0.3 ms` 的区间可能已经被 `segment_wall` 包含；`load_wait=1.0 ms` 通常又被 `callback` 包含。应把它们视为嵌套 / 重叠的观测窗口，而不是一张可以全部相加的账单。

一个更稳妥的定性分解是：

```text
端到端 token 时间
  = 有效 GPU 计算及不可隐藏的设备工作
  + 未被重叠隐藏的提交 / CPU 调度间隙
  + miss 在关键路径上造成的额外等待
  + 输入、输出和其他框架开销
```

其中每一项要通过真正的时间线划分，不能直接用当前几个打印字段替代。

### 1.20 为什么“层 hit 很高”仍可能频繁遇到 miss

以 40 层为例，若每层全专家命中的概率都为 97%，并且暂时假设各层独立，则一个 token 全部 40 层 hit 的概率约为：

```text
0.97 ^ 40 ≈ 29.6%
```

也就是说，在这个假设模型里，大约 70.4% 的 token 至少有一次 miss。实际层间存在相关性，这不是对真实数据的精确预测，但足以说明“97% hit”不等于“97% token 完全不进入慢路径”。

还需区分三种比例：

- 容量：230/256，表示每层能容纳的专家比例。
- 层级 hit：某层当前 token 选择的所有专家是否 resident。
- token 全 hit：该 token 的全部 MoE 层是否都不需要慢路径。

并行加载优化主要关注 miss 的服务时间；rolling 关注前瞻与尾部；常驻线程等关注固定成本。这三类优化可能在同一个“高层级 hit”的实验中同时有价值。

### 1.21 parallel loading 的收益上限示例

设一个缺失专家需要 gate、up、down 三片。假设分别读 0.20、0.25、0.40 ms，各上传 0.05 ms，仅用于示意。

旧的单 worker 顺序处理时间近似：

```text
(0.20 + 0.05) + (0.25 + 0.05) + (0.40 + 0.05) = 1.00 ms
```

三片并行读取时，不能直接宣称变成 `1.00/3`。读取会互相争用带宽，当前非 host-visible 上传又可能串行化，调度也有开销。即使理想化地忽略争用，完成时间仍受最慢片和上传安排限制。

真正的评价目标是“所有 required experts 可使用的时刻是否提前”，而不是“worker 总忙碌时间是否减少”。三个 worker 同时忙 0.4 ms，可以累计 1.2 ms 的 worker 时间，但用户只等约 0.4 ms；这种总和不能加进端到端耗时。

入队阶段拆片的意义也可以这样理解：若先让 worker A 取一个总任务，再由 A 拆出 B/C 的任务，B/C 的启动被 A 的调度延迟挡住；由 producer 一次入队可以去掉这段不必要的串行前置步骤，但不改变实际需要读取的字节数。

### 1.22 技术总结可采用的“问题 - 方法 - 证据”写法

后续整理正式技术文档时，建议每项都保持相同结构：

| 项目 | 原问题 | 方法 | 应看的证据 | 不应夸大的结论 |
| --- | --- | --- | --- | --- |
| GPU planner | hit 也有逐层 CPU 缓存处理 | 图算子与设备状态，hit 直接 remap | 同轨迹 hit 路径延迟与输出对照 | 不是所有 CPU 工作消失 |
| 连续 / 条件执行 | 主线程频繁切断图 | predicate 控制后续执行 | 边界数、有效执行时间、miss 行为 | 不是一次 graph 等于一次 vk submit |
| command buffer 恢复 | miss 后重复录制已知后缀 | 定位边界并 replay | 恢复 CPU 时间、录制次数 | 不是跨 token 永久缓存 |
| rolling | 早期 miss 仍有很长在途后缀 | 有界前瞻、异步观察、及时停止扩展 | tail、GPU 空洞、端到端延迟 | 不是旧窗口改了名字 |
| parallel loading | 少数 miss 专家无法用足 I/O 线程 | 专家内部权重片任务化 | required-ready 延迟 | 不是上传天然完全并行 |
| 常驻 worker | 反复线程创建及销毁 | epoch 协议复用线程 | 固定主机成本与抖动 | 不是去掉所有等待 |
| 按需 profiling | 测量本身扰动运行 | 关闭时跳过统计工作 | 同构建 profile on/off | 隐藏日志不等于消除统计成本 |

建议保存“旧设计为什么不够好”的探索过程，但明确它是被实验修正过的假设。比如可以写“起初怀疑后缀提交，随后 skip-tail 计时显示设备尾部也是重要成本”，不必将探索过程写成一开始就知道最终答案。

### 1.23 已有的 grouped conditional：减少的是条件区域开关，不是专家计算

当前实现还保留了可选的 `LLAMA_MOE_STREAM_GROUP_CONDITIONAL`，此前归纳中容易漏掉。其作用位置在 Vulkan 的图节点录制函数 `ggml_vk_build_graph`，不是 CPU 缓存策略。

没有合并时，受条件控制的多个算子大致录制为：

```text
begin conditional -> op A -> end conditional
begin conditional -> op B -> end conditional
begin conditional -> op C -> end conditional
```

开启合并后，同一个适用的 command buffer 内可保持一个条件区域：

```text
begin conditional -> op A -> op B -> op C -> end conditional
```

这减少重复 begin/end 及 begin 辅助函数中的相关同步命令。它不会减少全部 hit 时需要做的专家矩阵乘，也不会把全部模型变成一个 command buffer。区域仍须在 command buffer 结束时关闭，planner 等边界仍按原执行协议处理。

例如一段有 10 个受同一有效条件约束的节点，合并前可能需要 10 组开关，合并后可减少为一组；这是说明原理的例子，不代表每层实际节点数或固定加速比例。实际收益需看条件区域开关在驱动和 GPU 上的成本，当前材料没有可单独归因给此开关的端到端数据。

这一优化尤其说明：**“miss 后不做算术”不等于“后面的命令完全零成本”。** 驱动录制、条件控制和其他仍需经过的命令都有成本，因此需要同时优化提交粒度与条件执行组织。

### 1.24 本次精度排查发现的冷启动资源生命周期问题

此次扩充文档时，本地冷启动 tokenwise 评测发现了一个独立正确性问题：连续图中算子需要更大临时缓冲区时，旧的通用扩容函数会立刻结束并提交当前 command buffer、同步，然后重新开启录制。这套行为与外层尚未结束的条件执行区域冲突，也不适合保留命令供 miss 恢复重放的路径。

Vulkan 校验层实际报告了“结束 command buffer 时条件执行仍然活跃”以及“结束一个并未开启的条件执行区域”。这不是评分公式导致的 NaN，而是底层命令的合法性错误。本地表现为连续路径异常退出或停滞；远程所报告的 NaN 是否完全由这一问题引起，仍需用远程原命令回归确认。

本轮补丁的原则是：连续执行期间扩容时，新命令使用新临时缓冲区，旧缓冲区保留到本轮连续执行同步结束、恢复命令不再需要之后，再释放。普通非连续路径仍沿用原逻辑。这里保留的是计算 scratch，不是额外缓存专家，也不改变路由或淘汰策略。

代价是发生扩容的这一轮会短暂同时持有新旧 scratch；容量稳定后不会每 token 重复增加这一份保留量。不能把这个正确性补丁宣传成普遍性能优化。它反而提醒后续做 command buffer 复用时，必须把命令中引用的资源一起纳入生命周期管理。

本地正确性冒烟使用同一模型、输入和缓存配置，对比了原始 Stream、完整连续图和 rolling 三条路径。三者的短 PPL 均为 `5.9016 +/- 1.33008`，HellaSwag 小样本均为 `2/2`，MMLU 小样本均为 `1/2`；补充的 512-token PPL 在三条路径上也均为 `4.0363 +/- 0.61932`。这些结果说明本地已覆盖样本中没有观察到执行模式引入的评分变化，但 HellaSwag/MMLU 样本量很小，也不能替代远程 Q4_K_M 模型和原失败配置的完整回归。该正确性补丁及测试不追溯计入此前各阶段的性能收益。

## 二、当前框架内的进一步优化机会与优先级

### 2.1 总体判断：仍有小范围机会，但主要的大头已做过

现在没有证据支持再找一个“像最初 CPU 决策搬到 GPU 一样简单且巨大”的收益点。比较合理的后续路线是：先确认当前瓶颈，再针对提交、状态回读和小任务调度做局部优化。

以下几项已经实现，不应重复开发：常驻 I/O worker、常驻 rolling worker、按需 profiling、普通 rolling ready batch 合并、批量消费已完成 timeline、入队时拆分权重片、按任务量唤醒 worker、hit 快路径避免 miss 扫描。

没有任何候选项在本文中被实测证明有效，也不承诺固定百分比收益。

### 2.2 候选 A：rolling replay 的批量提交

**优先级：较高；范围：局部到中等；主要针对有 miss 的恢复路径。**

证据：普通 ready 路径已经通过 `ggml_vk_moe_stream_rolling_submit_contexts` 合并多个就绪项；但 `ggml_vk_moe_stream_rolling_submit_replay` 及其调用循环仍逐项处理 replay，单个 replay 项进行一次 queue submit。此外，完整录制后的 `ggml_vk_moe_stream_rolling_replay` 经 `submit_next` 按边界推进，也应一起审计，不能只看未录完的恢复分支。

可尝试的设计：在不超过剩余 lookahead 预算的情况下，收集当前已经就绪的多个 replay 项，在一次 queue 提交调用中保留多个有序 submit 信息和各自 timeline signal。不要为了凑批而等待未来项。

预期减少 queue mutex 获取和驱动 API 调用次数。它不减少 GPU 有效计算、SSD 读取或总上传字节数；纯 all-hit 时收益应有限，因为没有 replay。

风险和要求：

- 保留每个 planner 的 signal value，不能只保留最后一个而失去中间 miss 的及时观测。
- 保持提交顺序、在途预算、command buffer 生命周期和 `rolling_last_submitted_plan` 等状态一致。
- 一次提交更多项不能扩大原本允许的前瞻范围，否则可能用更长 skip tail 抵消 CPU 收益。
- 区分未录完恢复与完整录制后的 replay，分别量化触发频率、每次项数；某一分支很少出现时，单独优化它的价值有限。

验证：统计 replay API 调用次数及每次项数，比较 ms/token、恢复延迟、skip tail 和输出一致性。该候选比跨 token command buffer 缓存更局部。

### 2.3 候选 B：rolling 模式下减少重复 meta 拷贝

**优先级：较高；范围：局部；潜在收益偏小，但每层重复。**

证据：[copy_meta](C:/Users/admin/Desktop/llama-stream-moe/ggml/src/ggml-vulkan/ggml-vulkan.cpp:18842) 总是将 meta 拷贝到公共 readback；rolling 时还额外拷贝到每个 planner 的独立位置。rolling worker 使用独立快照，并在发现 miss 时将它复制到公共 CPU readback。

可研究是否在明确的 rolling planner 路径中只保留独立快照，去掉那次公共 GPU copy。重点不是减少几十字节带宽，而是减少逐层命令与潜在依赖。

不能现在直接认定公共拷贝必然多余：必须审计 prepare、status、commit、异常路径、全 hit 完成和恢复期间的所有读者。非 rolling 路径继续需要公共 readback。GPU 仍必须将写入正确发布给 CPU，不能顺手删掉必要的屏障。

验证：重点覆盖第一层 miss、最后一层 miss、连续多次 miss、全 hit、录制未完成恢复、错误处理。若仅小幅降低诊断字段而端到端无变化，不值得为此增加复杂分支。

### 2.4 候选 C：减少 rolling 热循环中的临时分配和 command-buffer 列表复制

**优先级：中；范围：小；预期为小收益。**

worker 热循环中反复构造 `ready_batch`；replay 从缓存项赋值时可能复制内部 command-buffer vector。每个 epoch 也有临时容器。

可以先用 CPU 分配采样确认是否明显，再考虑：

- 在 worker / epoch 内复用 ready 容器容量，而不是每轮析构。
- 对已消费 replay 项使用明确的 move 语义，前提是后续路径不需要原内容。
- 根据实际合理的在途数量预留容量，不要按任意巨大环境变量无界 reserve。

不要把 `vector.clear()` 当成释放了容量；现有不少长期 vector 已能复用容量，重复 reserve 没意义。也不要为了避免小拷贝直接保存另一个线程会扩容的 vector 元素引用，容易引入生命周期问题。

验证应看分配次数、CPU 时间与 token 延迟分布。如果成本本来淹没在 GPU / I/O 时间里，这类清理可能没有可测收益。

### 2.5 候选 D：权重片的局部调度顺序与线程数量

**优先级：中；范围：小；针对 miss 加载。**

当前权重片按 `sl.weights` 顺序进入 FIFO。若同一 miss 中片大小差异大，较大的片最后启动可能拉长所有片完成前的尾部。

可以只在同一次加载批次内部尝试较大 `nb_expert` 优先，保留不同需求批次之间的顺序。先测试现有线程数，不一定先改代码。

这只是候选，不是必然改进：文件访问局部性、不同 GGUF 文件分布、OS cache 和同步上传锁都可能让“大块优先”更慢；如果各片相近，收益可能接近零。不能改变 RESIDENT 的发布时机，也不能让背景任务抢占当前所需专家。

验证：分别对 1 个、2 个和更多新加载专家的 miss 统计延迟分布，结合总 decode 速度判断，不用所有 worker 耗时之和代替关键路径。

### 2.6 候选 E：lookahead 调参优先，动态策略后置

**优先级：调参较高；自动自适应较低；无需先改代码。**

前瞻过小，可能让 GPU 等下一批命令；过大，又增加 miss 后已提交的无效尾部。最佳值受模型、硬件、缓存容量和上下文长度影响。

先在当前代码上固定测试一组小 lookahead，例如 1、2、4、8，并同时保留非 rolling 连续路径作对照。不能用历史 WINDOW=4/8 的结果直接替代 ROLLING 的调参结果。

如以后做自适应，不宜只根据 hit streak 增减：还需要知道是否缺少 ready 工作、GPU 是否饥饿、miss 后尾部是否明显增长。没有这些观测，动态策略可能增加抖动和不可复现性。封包前更适合保留简单、可解释的默认值。

### 2.7 有潜力但不符合本轮“小改动”目标的方向

以下不建议当作下一步轻量优化：

1. **全面异步上传 / 批量上传。** 当前同步 tensor_set 和 fence 等待确实值得关注，但替换成异步调用需要 staging 生命周期、传输队列同步、slot 完成发布和错误回收配套。不是改一个 API 名字即可。
2. **发现 miss 立刻与尾部并行读取。** 提前开始磁盘 read 理论上能隐藏延迟，但必须冻结正确计划、分离读与上传，并确保不能提前覆盖仍被在途 GPU 使用的 slot。当前并发协议要扩展。
3. **跨 token command buffer 长期缓存。** 输入地址、KV 状态、descriptor、图形状、缓存状态、资源回收和失效条件都需保证。不能等同于现有 suffix replay。
4. **并行化 planner / 增量 commit / 融合算子。** shader 当前单 invocation，miss 和 commit 中有串行扫描；但要保留热度、victim tie-break 和更新顺序。没有独立 planner 成本证据前，不宜为理论并行度增加复杂度。
5. **放松同步屏障。** 尤其刚补强的 compute 可见性约束，不应作为试错式加速手段。必须先有严格依赖证明。

### 2.8 验证建议：防止把测量差异误认为优化

建议每次只改一个候选项，保留可以快速回退的独立实验版本，并记录：

- commit、可执行文件构建来源、模型 hash / 量化、驱动和 GPU、完整参数与环境变量。
- 同一 prompt、固定生成 token 数、相同采样条件和上下文；比较轨迹时更适合固定 token 输入。
- 缓存容量至少覆盖 220、230、240 及全缓存诊断；容量与实际 hit rate 分开记录。
- 区分冷文件缓存与热文件缓存；明确 direct / buffered I/O。
- 正式速度测试关闭 profiling，诊断轮统一开启；不混合两种结果。
- 交替 A/B 运行并重复，报告中位数和波动，而不只挑代表截图。
- 首 token / 预热与稳定 decode 分开，累计计时归一到 token 和 miss，并保留分母。
- 先做精度与状态机回归，再接受速度收益；总准确率一致之外，尽量比较逐候选分数或 logits 误差。

当前统计不是完整的、互不重叠的端到端时间分解。若需要解释关键路径，可在实验分支短期加入必要的 CPU / GPU trace，确认重叠关系后再删除，而不是长期恢复全部重型日志。

### 2.9 推荐顺序与封包结论

推荐顺序：

1. 先固定当前 `beea7749b` 的无 profiling 性能和正确性基线，确认最新常驻线程及统计关闭效果。
2. 做 lookahead / I/O 线程数的小规模参数矩阵，成本最低。
3. 若 replay 提交次数多，优先候选 A；若高 hit 固定成本仍明显，优先审计候选 B、C。
4. 若 miss load 仍主导且片大小不均，再试候选 D。
5. 只有确认这些局部优化收益不足、且愿意扩大改动范围，再考虑异步上传等结构性工作。

用于技术文档的核心总结可以写成：

> 本次工作将 Stream MoE 单 token 解码中的缓存决策和专家索引映射迁移到 Vulkan GPU 路径，通过条件执行维持 hit 路径连续性；进一步使用有界滚动提交与 command buffer 恢复，控制 miss 后无效尾部和重复录制成本；通过权重片并行加载、常驻提交线程及按需 profiling，优化 miss 服务时间与固定主机开销。历史实验支持上述瓶颈判断，最新版本的具体收益仍应由统一配置的回归数据归档确认。

在本次正确性补丁完成远程原失败配置回归后，可以进入“稳定主版本封包、局部实验另行验证”的阶段，不必为了寻找最后一点性能继续扩大核心框架。最重要的是留下清楚的版本、配置、正确性边界和可复现实验，而不是继续增加更多难以解释的统计字段。

### 2.10 最小实验矩阵与接受标准

不必每次跑全部组合，可以分三级：

**第一级：正确性冒烟。**

- 原始 Stream、GPU planner、非 rolling 连续、rolling 各跑相同短输入。
- 至少覆盖冷缓存启动、连续 miss、完成一段后清空模型序列状态再重跑。
- 精度工具同时覆盖非 tokenwise 与 tokenwise；二者是不同执行覆盖，不是只比较速度。
- 在能够输出 logits / 评分的地方检查非有限值，不能把 NaN 替换成 0 后继续评分。

**第二级：针对候选机制。**

- 提交优化：选高 hit 和有 replay 的场景，记录真实调用次数。
- 加载优化：选有单专家和双专家 miss 的场景，观察 load-ready 的尾延迟。
- meta 优化：覆盖全 hit、最早 / 最晚 miss 和恢复，不只测试正常生成。

**第三级：正式端到端。**

- 固定输出长度，交替 A/B，至少获得足以看清运行波动的重复样本。
- 报告吞吐，同时换算 `1000 / tps` 为 ms/token；加速率和延迟下降率不是同一个百分比。
- 对 220/230/240 的结果分别报告，不将不同 cache 容量混成一个总平均。
- 如优化只在小部分场景获益而其他场景退化，明确条件，不轻易更改默认行为。

举例：由 40 t/s 提升为 50 t/s 是吞吐提高 25%，而 ms/token 由 25 降为 20，是延迟降低 20%。描述“提升 25%”时需要说明说的是哪一个指标。

### 2.11 建议保留的实验记录模板

```text
实验 ID：
日期 / commit / 构建目录：
模型文件 / hash / 量化：
GPU / 驱动 / CPU / 存储：
数据集 / 样本数 / 固定输入轨迹：
cache slots / I/O threads / rolling lookahead：
GPU planner / continuous / parallel load / profile：
tokenwise / batch / ubatch / context / flash-attention：
文件缓存状态 / direct I/O / 预热方式：

A 与 B 唯一差别：
正确性结果（PPL、逐题分数、非有限值、logits 误差）：
重复运行 t/s：
中位数 / 波动 / ms per token：
hit、miss、token 数及分母：
相关诊断字段（另开 profile 轮）：
结论：接受 / 回退 / 证据不足
已知限制：
```

远程精度测试还应保存完整的错误前文，而不只有最后的 `nan`：需要判断它来自模型 logits、评分算式、统计标准差还是空样本。并且 Windows 构建的 `llama-perplexity.exe` 依赖同目录 DLL；只替换 exe 不代表已更新实际评测代码，尤其当前工具主体在 `llama-perplexity-impl.dll` 中。

### 2.12 封包前建议检查的边界

- 用户说明只列当前有效开关，旧 WINDOW 和已移除统计单列为历史内容。
- 说明仅在满足条件时进入单 token Vulkan 路径，并保留能确认实际路径的日志。
- 记录完整可执行文件及依赖库版本，避免本地 / 远程 DLL 混用。
- 保留 NaN / 无效计划 / 加载失败的明确错误，而不是输出一个看似正常的精度数字。
- 以最小正确性回归覆盖多次调用、状态清空、不同序列及 miss 恢复，不能只测一段连续生成。
- 对每个被删除的调试功能注明替代诊断方法；少而准确的工具比大量口径混杂的计时更有价值。
