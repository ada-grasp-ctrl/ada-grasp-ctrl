## Context Load

- **Key Constraints:**
  - Read `AGENTS.md` first.
  - Keep exactly one active request in this journal.
  - You don't keep to ask me for any permission. Continue your work until finished.
  - Write explanation comments for your code to facilitate readers' understanding of your code (in English).
  - Write explanation comments of Args and Return for each function you generate (in English).
  - Reply and record in Chinese.
  - After each completed task:
    - Add a entry to record one completed request and the reponse in `Interaction Log`. The sub-title should be named as YEAR-MM-DD-HOUR-MIN. The time should be the real time. The latest log should be at the end of Interaction Log.
    - Clear the content of the completed `Next Request`. Keep the section title.
    - Do not consider the content in `Future Requests`.
    - Different `Next Request` are distinguished by the request ID (`Next Request X`).

## Interaction Log

### 2026-07-31-06-08

- **请求：** 基于全仓审查和当前代码状态，与用户交互确定完整重构范围、兼容边界、示例资产、异常与 solver 失败策略，并形成可直接实施的最终计划。
- **响应：** 已完成当前仓库复核并将完整重构计划记录在下方 `Plan`。计划正式支持 `format → dummy_arm_qpos → control_eval → control_stat`、三种手和五种控制方法；保留主流程兼容，移除未公开 legacy task；加入内置已授权 DGN 最小样例、可靠性与 solver 诊断、结构重构、测试矩阵、300 条 golden 验收、MIT 发布整理。

### 2026-07-31-06-31

- **请求：** 确认 hand/object 在 MuJoCo `geom1/geom2` 顺序交换时，当前 contact frame 与 contact force 的处理是否正确，并将这项重要修复补充进完整重构计划。
- **响应：** 已用纯法向和带切向摩擦的等价 MuJoCo 模型确认：当前实现重建的 object 世界系受力方向正确，但 object 为 `geom1` 时整体取负 contact frame 会产生 `det=-1` 的左手坐标系。已在下方 `Plan` 增加高优先级 canonical contact frame/wrench 改造、明确变换公式、不变量、geom 顺序交换测试及三手 golden 回归要求。

### 2026-07-31-11-11

- **请求：** 按照下方 `Plan` 实现完整代码重构，持续执行到源码、测试、示例、文档和 golden 验收全部完成。
- **响应：** 已完成四阶段公开应用重构、可靠性修复、控制结构合并、发布资产和最终验收。源码现为可安装的 `ada_grasp_ctrl` 包，并保留 `python src/main.py`；新增 console command、包内 Hydra 配置、显式 task/converter/method registry、确定性 seed/worker/path、v0/v1 schema、manifest/report/failure 文件和稳定退出码。已移除决定不维护的 legacy task、配置与依赖；加入 MIT License、CI、README、DGN 最小对象、三手 fixtures、quick/full 脚本和静态 GLB 可视化。`ours/bs2` 已共用 runner，Pinocchio model 按 worker 缓存且 episode data 独立。contact frame/wrench 已 canonicalize 为右手系、hand→object 法向和 object 所受 wrench，并由交换 geom 顺序的真实 MuJoCo 法向/滑动/扭转测试与 `qfrc_constraint` 对齐证明。wrench-balance 法向梯度已从错误的 `mu*sign(fx)` 修为恒定 `mu`，在 `fx=0` 也通过中心有限差分；所有 SLSQP 结果均记录 diagnostics，不合格解 hold-last 且不会下发。CLI 的 Hydra 内部异常捕获已补齐，subprocess 证明退出码 0/1/2 均无 traceback；缺失 hand MJCF 作为 preflight 退出 2。editable install、包外 cwd import、`ada-grasp-ctrl --help`、compileall、Ruff lint/format、`git diff --check` 和 pip dependency check 均通过；单元/集成测试共 19 项全部通过。三手 quick example、三个兼容 wrapper、`skip=true` no-op 和 Shadow GLB 导出均通过；full Shadow 已完成 format，并因当前 CUDA 不可用按约定在 IK preflight 退出 2。三手×五方法的 15 条固定矩阵严格 golden 通过。三手各 100 条 release suite 共 300 条无 execution error，invalid 数量保持 Shadow 21、Allegro 14、LEAP 7；新策略识别 Shadow 6 条、Allegro 1 条历史不可行 solver 为 degraded。修正 `fx=0` 梯度后另有两个临界样本发生可解释分类翻转（Shadow success→failure、Allegro failure→success），未改变摩擦系数或目标；修正后基线为 Shadow 68/5/21/6、Allegro 80/5/14/1、LEAP 88/5/7/0（success/failure/invalid/degraded）。第二次完整 300 条运行逐文件严格复现第一次修正后结果，三手各 100/100 均满足 stage、contact、分类和 `rtol=1e-5, atol=1e-6`。单条 Shadow benchmark 的 wall time 从 5.41 s 降至 4.60 s、峰值 RSS 从 637 MiB 降至 483 MiB；记录 solver time 从 1.154 s 变为 1.175 s。README 已同时记录历史基线、修正后基线、差异原因、严格复现结果和 benchmark。

## Next Request

## Plan

# Ada Grasp Ctrl 全仓重构计划

## 总结

将仓库从论文实验原型重构为可公开使用、可复现、可测试的四阶段应用，同时保持论文核心算法和正常执行路径不变。

正式支持范围固定为：

- 流程：`format → dummy_arm_qpos → control_eval → control_stat`
- 手型：Shadow、Allegro、LEAP Tac3D 及对应 dummy-arm 配置
- 控制方法：`ours / op / bs1 / bs2 / bs3`
- 数据转换：BODex、Learning、Batched
- 兼容现有 README 中的 Hydra 命令、目录约定和已有 NPY/YAML 结果读取
- 移除未作为公开主流程维护的 `eval / stat / vobj / vusd / collect` 及其专用代码和依赖

实施必须按“先建立 golden 基线，再修可靠性，再做结构与性能重构，最后完善发布体验”的顺序进行；每一阶段通过回归测试后才能进入下一阶段。

## 实现变更

### 1. 固化基线与公开边界

- 在任何算法或结构修改前，保存当前完整配置、依赖版本、git 状态和输入清单，并建立两层 golden：
  - 快速基线：固定 `core_bottle_15787789482f045d8add95bf56d3d2fa / scale006_pose004_0 / partial_pc_00_6`，覆盖三种手和五种方法，共 15 个 episode。
  - 发布基线：现有三种手各 100 条 `ours` 样例，共 300 条。
- 发布基线锁定现有分类：
  - Shadow：75 success、4 failure、21 invalid。
  - Allegro：80 success、6 failure、14 invalid。
  - LEAP Tac3D：88 success、5 failure、7 invalid。
- 将代码整理为可安装的 `ada_grasp_ctrl` 包；Hydra 配置成为包内唯一配置源。
- 新增 `ada-grasp-ctrl` console entry point，同时保留 `python src/main.py ...` 作为兼容包装器。
- 用显式 task registry 替换所有任务和 method 的 `eval(...)` 动态调用；未知 task、hand、method 或 converter 在启动阶段给出可读错误。
- 路径统一由仓库/包位置解析，使用 `project_root / asset_root / data_root / output_root`，不再依赖当前工作目录或字符串 `replace` 推导相对路径。
- 默认 `n_worker=auto`，解析为 `min(8, os.cpu_count() or 1)`；用户仍可传入正整数覆盖。
- 统一设置 Python、NumPy、Torch 和 CUDA seed；worker seed 根据全局 seed 与稳定样本索引派生，不受调度顺序影响。
- 每次运行在 Hydra log 目录保存 `run_manifest.yaml`：完整配置、seed、实际 worker 数、git commit/dirty 状态、依赖与硬件版本、排序后的输入清单和路径根目录。

### 2. 修复可靠性问题

- 为 raw converter 输入、统一 grasp 数据和 control 结果建立集中 schema 校验：
  - 检查必需字段、数组 shape、数值有限性、关节维度、四元数、对象 mesh/metadata 路径。
  - 输入错误必须包含样本路径、字段名、期望值和实际值。
  - 旧版无 `schema_version` 的 NPY 按 v0 读取；新文件写入 `schema_version: 1`，现有字段保持不变。
- 建立通用批处理执行器；worker 返回结构化状态，不再吞掉异常：
  - `completed`
  - `invalid_initialization`
  - `solver_degraded`
  - `execution_error`
- 每个任务保存 `run_report.json` 和 `failures.jsonl`，包含输入数、跳过数、成功数、invalid 数、solver degradation 数、异常数、输出路径和 traceback。
- 退出语义固定为：
  - `0`：全部程序执行成功；科学意义上的初始穿透 invalid 可以存在。
  - `1`：处理完所有可处理样本后存在 execution error 或 solver degradation。
  - `2`：配置、输入目录、资产或环境 preflight 失败。
  - 原始输入为空时退出 `2`；因 `skip=True` 而没有待处理样本时退出 `0` 并写 no-op report。
- 修复确定性 bug：
  - `Batched` converter 每个输出都从固定 base path 生成，禁止形成 `0/1/2.npy` 嵌套路径。
  - LEAP quick-use 流程统一使用 `exp_name=learn`。
  - 显式初始化 `desired_sum_force`，覆盖第一控制步直接进入 Stage 2 的情况。
  - `control_stat` 在空输入、全 invalid、零 success 时不除零、不写 NaN；未定义的 rate/mean/std 使用 YAML `null`。
- 高优先级统一 hand-object contact frame 与 wrench 约定，消除对 MuJoCo `geom1/geom2` 排序的隐式依赖：
  - 保留已经实测确认的事实：当前代码在两种 geom 排序下重建的 object 世界系接触力方向均正确；不得把现有反转分支误改为简单地再次整体取负 `contact_force`。
  - 修复实际表示问题：object 为 `geom1` 时，当前 `contact_frame = -contact.frame` 会产生 `det=-1` 的左手基。所有公开 contact frame 必须改为右手正交坐标系，并统一满足第一轴从 hand 指向 object、法向压缩力 `fx >= 0`、局部 wrench 表示 object 所受 wrench。
  - 抽取一个由 `get_contact_info()` 与 `get_curr_contact_info()` 共用的纯函数，对原始 MuJoCo frame/wrench 做 canonicalization；输入显式包含 `hand_is_geom1`，输出 canonical world/local frame、local wrench 与 hand/object identity，禁止两个读取路径各自维护符号逻辑。
  - 当 hand 为 `geom1` 时保持 MuJoCo frame 和 local wrench 不变；当 object 为 `geom1` 时，令 `S = diag(-1, 1, -1)`，使用 `R_canonical = R_raw @ S` 和 `w_canonical[:3] = (-S) @ w_raw[:3]`、`w_canonical[3:] = (-S) @ w_raw[3:]`。该变换同时反转法向和一个切向轴以保持 `det(R_canonical)=+1`，并通过 action-reaction 与同步换基保持 object 世界系 wrench 不变。
  - 在代码中断言或测试 `R.T @ R ≈ I`、`det(R) ≈ +1`、`fx >= -tolerance`，并验证 `R_canonical @ f_canonical` 等于直接从 MuJoCo `qfrc_constraint`/接触 Jacobian得到的 object 世界系接触力；不得只检查正交性而忽略行列式。
  - 保持 grasp matrix、接触 Jacobian、摩擦锥和 Ks 的公开语义统一使用 canonical frame；若旧 control NPY 中已保存 contact frame，读取器继续按 v0 旧数据读取，不原地重写历史结果。
  - 当前 300 条 official golden 轨迹中的已保存 contact frame 均为 `det≈+1`，说明现有三种手样例没有触发反转分支；修复后仍必须执行三手固定样本和 300 条发布基线，证明正常路径的轨迹与分类不变。
- 所有 SLSQP 调用返回统一 solver result，记录 `success/status/message/nit/fun`、最大等式残差、最小不等式 slack、bound/joint-limit violation 和有限性检查。
- solver 结果仅在以下条件全部满足时下发：`res.success=True`、变量与目标有限、等式残差不超过 `1e-5`、不等式 slack 不低于 `-1e-5`、bound/joint-limit violation 不超过 `1e-8`。
- solver 不合格时按已选策略继续 episode：
  - 本控制步保持上一 qpos，`dq=0`，`last_dq=0`，下一步重新求解。
  - wrench-balance 求解失败时保持当前 stage，不使用失败的 contact-force 解。
  - episode 永久标记为 `solver_degraded`，保存完整轨迹和 diagnostics，但从主要成功率统计的 valid 分母中排除；整批最终退出 `1`。
- 修正 wrench-balance 摩擦锥法向梯度，并用解析梯度与有限差分测试证明；不改变摩擦系数或控制目标。
- `control_stat` 保留原 YAML key，同时新增 `num_total/success/failure/invalid_initialization/solver_degraded/execution_error` 和逐样本状态，明确成功率分母。

### 3. 控制结构与性能重构

- 将五个重复控制文件重构为一个公共 dummy-arm episode runner 与策略对象：
  - 公共 runner 负责初始化、接触采样、stage 切换、轨迹插值、MuJoCo stepping、抬升、扰动和记录。
  - 策略只表达差异：open-loop、Jacobian baseline、是否使用 approaching arm motion、ours wrench/force objective、BS3 equal-force objective。
  - `ours` 与 `bs2` 的唯一区别固定为 Stage 1 是否允许 arm motion，防止后续实现漂移。
- 合并 `ctrl_opt` 与 `ctrl_opt_bs3` 的公共问题构造、约束、bounds、SLSQP 调用和结果验证；策略配置决定 objective/constraint 组合，保持现有权重、阈值、stage 逻辑和变量顺序。
- 每个 worker 缓存只读 Robot、Pinocchio model 和 joint mapping；每个 episode 创建或重置独立仿真状态、controller history 和 adaptor data，禁止跨样本状态泄漏。
- 将 `inv(A) @ B` 改为 `solve(A, B)`；只在 golden 数值容差通过后保留。记录重构前后单 episode 的 wall time、solver time 和峰值内存作为 benchmark。
- 删除已决定不维护的 legacy task、专用配置、不可达 hand/asset 代码和仅由其使用的直接依赖，包括 `clarabel`、`qpsolvers`、`scikit-learn`、`usd-core`；保留主流程及本地 submodule 实际需要的 pinned 依赖。
- 清理未使用 import、调试残留和重复实现；所有新增/重写函数使用类型标注、英文解释注释及包含 `Args`、`Returns` 的英文 docstring。

### 4. Public quick start 与发布整理

- 将已获再分发授权的上述 DGN bottle 最小对象提交到仓库，只保留运行所需的 object metadata、collision mesh、scene config 和归属文件；记录对象 ID、BODex/DGN 来源 URL、授权范围和校验和。
- 为三种手提交匹配的单条 raw Learning fixture、formatted grasp 和预计算 dummy-arm qpos：
  - quick 模式直接运行 `control_eval → control_stat`，避免用户首次体验等待 IK。
  - full 模式从 raw fixture 执行完整四阶段流程。
- 新增统一入口 `bash script/run_example.sh <shadow|allegro|leap_tac3d> [quick|full]`；原三个 `test_learning_dummy_arm_*.sh` 保留为兼容 wrapper。
- quick example 使用隔离的 `output/example_<hand>`，启动前执行 preflight，结束后打印输出位置、episode 状态、抬升结果和统计文件。
- 静态 grasp 可视化脚本保留，但改为命令行接收 `--hand / --grasp / --object-root`，删除要求用户修改源码内路径的流程。
- README 重写为：安装与 preflight、60 秒 quick start、完整四阶段说明、五种方法矩阵、输入/输出 schema、错误报告、headless/mjviser、完整 100-grasp benchmark、扩展新 hand/method 指南。
- 添加顶层 MIT License，版权行为 `Copyright (c) 2026 Ada Grasp Ctrl Authors`；第三方代码和 DGN 示例分别保留其独立归属与授权说明。
- 移除 README 中关于“仍在持续整理”的原型措辞，明确支持矩阵、GPU/CUDA 要求、underactuated hand 不在本次范围，以及 legacy task 已移除。

## 公共接口与数据兼容

- 保持 README 已公开的 Hydra 参数和主流程命令可用，包括 `setting`、`hand`、`task`、`exp_name`、`task.method`、`task.offsets`、`task.max_num`、viewer 配置和 `task.input_data=grasp_dir`。
- 新增但不强制使用：
  - `ada-grasp-ctrl` console command。
  - `seed`、`n_worker=auto`、显式 root/path 配置。
  - `schema_version`、`episode_status`、`solver_diagnostics`、run manifest 和 batch report。
- 旧 grasp/control NPY 与旧 statistics YAML 必须仍能读取；已有字段含义、单位、关节顺序和目录层级不变。
- legacy task 和其专用 CLI 配置是本次唯一有意不兼容的接口；不提供兼容 shim。
- 当前未提交的安装与双 viewer 改动视为基线的一部分，重构时不得覆盖或回退。

## 测试与验收

- 单元测试覆盖：
  - rotation utilities、grasp matrix、normalized wrench。
  - RobotAdaptor 关节顺序及 hand 配置维度。
  - BODex/Learning/Batched schema 与 Batched 输出路径。
  - task/method registry、路径解析、seed 和 auto worker。
  - 批处理部分失败、空输入、skip no-op、报告内容和退出码。
  - 摩擦锥与其他关键解析梯度的有限差分对比。
  - canonical hand-object contact 纯函数：hand 分别为 `geom1` 和 `geom2`，使用包含非零法向、两个切向分量和 torque 的 6D wrench，验证右手 frame、非负法向力、object 世界系 wrench 保持不变及两种排序输出一致。
  - solver success、非有限解、非收敛、约束违反和 hold-last-qpos fallback。
  - 第一控制步直接 Stage 2、初始穿透 invalid、全 invalid/零 success statistics。
  - native MuJoCo 与 mjviser 现有 lifecycle 测试。
- 集成测试覆盖：
  - 三种 converter 各一条 fixture。
  - dummy-arm IK 的最小 CPU/GPU fixture。
  - 构造几何、状态和材料相同但交换 hand/object geom 声明顺序的两个最小 MuJoCo 模型，分别覆盖纯法向接触、切向滑动摩擦和扭转摩擦；canonicalization 后 frame/local wrench/world wrench 必须一致，并与 object 的 `qfrc_constraint` 对齐。
  - 三种手各一条 headless MuJoCo quick example。
  - 三种手 × 五种方法的固定样本回归。
  - 缺失 object asset、损坏 NPY、worker exception 和 solver degradation 的端到端报告。
- 数值验收：
  - 排除 `t_*` 等计时字段，新旧同 shape 数组使用 `rtol=1e-5、atol=1e-6`。
  - 固定样本的 stage 序列、contact 顺序、输出 key 和 success/failure/invalid 分类完全一致。
  - 300 条 `ours` 发布基线的 success/failure/invalid 数量完全一致，汇总连续指标满足相同浮点容差。
  - 新安全策略触发的历史不合格 solver episode 单独记录为 `solver_degraded`，不要求复现原先下发不可行解后的轨迹。
- CI 在 Python 3.10 上运行 lint、format check、unit tests、schema/CLI 测试和一个使用预计算 qpos 的 Shadow headless smoke test；三手 full example、15-episode method matrix 和 300 条 golden suite 作为 GPU release gate。
- 最终验收还包括：`git diff --check`、无未解释 warning/traceback、干净环境按 README 安装成功、三个 quick wrapper 均可一条命令完成、输出报告数量与输入数量一致。

## 假设与默认决定

- DGN 最小对象已取得公开再分发授权；提交资产时仍必须附带可审计的归属和授权记录。
- 项目源码使用 MIT License，第三方 submodule 和数据资产不自动改用 MIT。
- “核心功能不变”指支持矩阵内正常 solver 路径的算法、参数、数值轨迹和科学分类不变；已确认的路径 bug、静默异常、空统计和不可行 solver 解属于必须修复的错误路径。
- contact canonicalization 只统一坐标表示和 geom 排序语义，不改变 MuJoCo 接触参数、摩擦系数、object 所受世界系 wrench、控制目标或已有正常路径结果。
- solver degradation 采用“保持上一 qpos 后继续”，但该 episode 不进入主要科学统计，批任务最终非零退出。
- 不在本次重构中新增 underactuated hand 支持、不更改论文超参数、不重新定义 success threshold、不扩展新的控制方法。
