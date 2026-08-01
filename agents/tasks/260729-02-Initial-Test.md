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

### 2026-07-30-15-00

- **请求：** 按 README 验证仓库提供的 Shadow、Allegro 和 LEAP Tac3D example evaluation dataset 是否能够完整跑通，并记录分析与结论。
- **响应：** 已完成三种手各 100 条样例的 `dummy_arm_qpos -> control_eval -> control_stat` 全流程测试。核心代码和样例数据均可跑通，但 LEAP 快捷脚本中的 `exp_name=learn_2` 与预置数据及后续步骤使用的 `learn` 不一致，原脚本不能从干净状态直接跑通；改为 `exp_name=learn` 后可完整运行。测试结果、统计数据、前置资源要求和错误处理风险见本文 `Analysis`。
- **工作区影响：** 未修改源代码或测试脚本；生成了三种手的 dummy-arm、control、statistics 和 Hydra 日志。`assets/object/DGN_2k` 当前链接到 `/data/mingrui/dataset/coorgrasp/object/DGN_2k`。

### 2026-07-30-15-17

- **请求：** 重新连接 `assets/object` 后，基于新的对象资源重新测试 example grasp，并检查三种手的成功率。
- **响应：** 已复用三种手各 100 条 dummy-arm grasp，在隔离目录重新执行 `control_eval -> control_stat`。300 条控制结果均成功生成，日志无 traceback、warning 或资源缺失错误。Shadow、Allegro 和 LEAP Tac3D 的有效样例成功率分别为 94.94%、93.02% 和 94.62%，与上次结果完全一致。
- **工作区影响：** 未修改源代码、配置或 quick-use 脚本；新结果保存在 `output/retest_assets_20260730/`，旧的 `output/learn_dummy_arm_*` 结果保持不变。

## Next Request 1

## Next Request 2

## Next Request 3


## Future Request

## Analysis

### 测试目标

验证 README 声明的三套预采样数据：

- `output/learn_shadow`
- `output/learn_allegro`
- `output/learn_leap_tac3d`

是否能够跳过 `format`，直接依次完成：

1. `dummy_arm_qpos`：使用 IK 计算 dummy arm 关节位置。
2. `control_eval`：执行 `ours` 控制方法并保存操作过程。
3. `control_stat`：读取控制结果并生成统计 YAML。

### 环境与数据预检

- Python：3.10.20。
- PyTorch：2.2.2，CUDA build 12.1。
- MuJoCo：3.3.2。
- Pinocchio：3.0.0。
- GPU：NVIDIA GeForce RTX 4090 可用。
- 三种手的预采样数据均为 100 个 `.npy`，分别覆盖 89 个物体。
- 测试使用 `task.debug_viewer=False`，与 quick-use 脚本一致，适合无图形界面的终端环境。

预采样 grasp 只包含姿态和物体路径，并不包含物体 mesh。完整评测仍依赖 README 的 Object Preparation。当前资源链接为：

```text
assets/object/DGN_2k
-> /data/mingrui/dataset/coorgrasp/object/DGN_2k
```

已确认目标下存在 `processed_data`、`scene_cfg` 和 `valid_split`。`scene_cfg` 也必须存在：样例中的 `obj_path` 包含经过 `scene_cfg/.../../../../processed_data/...` 的相对路径，只有 `processed_data` 而没有中间目录时，文件访问仍会失败。

### 完整测试结果

| Hand | 原始 grasp | dummy-arm IK 输出 | control 输出 | statistics |
| --- | ---: | ---: | ---: | --- |
| Shadow | 100 | 100 | 100 | 成功生成 |
| Allegro | 100 | 100 | 100 | 成功生成 |
| LEAP Tac3D | 100 | 100 | 100 | 修正 `exp_name` 后成功生成 |

三种手的 IK 均在首次完整执行中成功，没有需要按 README 提示重新运行失败 batch。

统计结果如下。成功率以有效样例为分母；invalid 表示初始穿透或接触力等仿真初始化检查未通过，不代表流水线程序执行失败。

| Hand | 有效 | Invalid | 成功 | 失败 | 成功率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shadow | 79 | 21 | 75 | 4 | 94.94% |
| Allegro | 86 | 14 | 80 | 6 | 93.02% |
| LEAP Tac3D | 93 | 7 | 88 | 5 | 94.62% |

统计文件：

- `output/learn_dummy_arm_shadow/control_stat_res/dist_0_ours_default.yaml`
- `output/learn_dummy_arm_allegro/control_stat_res/dist_0_ours_default.yaml`
- `output/learn_dummy_arm_leap_tac3d/control_stat_res/dist_0_ours_default.yaml`

### 发现的问题

#### 1. LEAP quick-use 脚本实验名不一致

`script/test_learning_dummy_arm_leap_tac3d.sh` 的 IK 命令使用：

```bash
python src/main.py setting=tabletop hand=leap_tac3d task=dummy_arm_qpos exp_name=learn_2 task.max_num=-1
```

但 README 提供的目录是 `output/learn_leap_tac3d`，脚本后续 `control_eval` 和 `control_stat` 也都使用 `exp_name=learn`。实际执行上述 IK 命令时：

```text
input: 0
generated dummy-arm data: 0
exit code: 0
```

应将该命令改为：

```bash
python src/main.py setting=tabletop hand=leap_tac3d task=dummy_arm_qpos exp_name=learn task.max_num=-1
```

修正参数后，LEAP 的 IK、control 和 statistics 均完成 100 条测试。

#### 2. `control_eval` 的退出码可能产生假阳性

`src/task/control_eval.py` 的 `safe_eval_one` 捕获每条样例的所有异常，只写 warning 后返回；因此即使所有样例都因缺少 object asset 失败，主进程仍可能打印 `Finish control evaluation` 并返回退出码 0。

测试中曾验证：缺少对象资源时出现 `FileNotFoundError`，但命令仍以 0 退出且没有生成 control 文件。因此不能只检查退出码或结束日志，应同时检查：

- 日志中是否存在 traceback/warning。
- `control` 下实际生成的结果数量是否与输入数量一致。
- `control_stat` 是否找到了预期数量的结果。

本次最终验收以三种手各生成 100 个 control 文件，并且 `control_stat` 各读取 100 个结果为准。

### 结论

- 核心代码及三套 example evaluation dataset 可以完整跑通。
- Shadow 和 Allegro 的 quick-use 流程可按现有脚本运行。
- LEAP Tac3D 的 quick-use 脚本不能原样从干净状态跑通；将 `exp_name=learn_2` 改为 `exp_name=learn` 后可完整运行。
- 即使跳过 `format`，仍必须准备完整的 `DGN_2k` object assets。
- 验收时必须检查实际输出数量，不能只依赖进程退出码。

### 重新连接对象资源后的复测（2026-07-30-15-17）

#### 资源与输入检查

- `assets/object/DGN_2k` 解析到 `/data/mingrui/dataset/coorgrasp/object/DGN_2k`。
- Shadow、Allegro 和 LEAP Tac3D 各有 100 条 dummy-arm grasp，分别覆盖 89 个对象。
- 逐条检查 grasp 引用对象所需的 `info/simplified.json` 和 `urdf/meshes`，缺失数为 0。
- 对象资源只参与控制仿真，因此复用已有 dummy-arm IK 结果，不重复执行 `dummy_arm_qpos`。
- 使用 `/home/ymr/miniconda3/envs/ada-grasp-ctrl/bin/python`，设置 `task.debug_viewer=False`，按顺序执行三种手的 `control_eval` 和 `control_stat`。

为避免旧结果污染，新产物写入独立目录：

- `output/retest_assets_20260730/learn_dummy_arm_shadow`
- `output/retest_assets_20260730/learn_dummy_arm_allegro`
- `output/retest_assets_20260730/learn_dummy_arm_leap_tac3d`

#### 复测结果

三种手均生成了恰好 100 个 `ours_default`、`dist_0` control 文件；统计阶段也分别读取了 100 条结果。日志中未发现 traceback、warning、`FileNotFoundError` 或其他 error。

| Hand | 有效 | Invalid | 成功 | 失败 | 有效样例成功率 | 全部样例成功占比 | 相比上次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shadow | 79 | 21 | 75 | 4 | 94.94% | 75.00% | +0.00 pp |
| Allegro | 86 | 14 | 80 | 6 | 93.02% | 80.00% | +0.00 pp |
| LEAP Tac3D | 93 | 7 | 88 | 5 | 94.62% | 88.00% | +0.00 pp |

仓库定义的 `success_rate` 以有效样例为分母，即 `成功 / (成功 + 失败)`；全部样例成功占比额外以固定的 100 条输入为分母，便于同时观察 invalid 的影响。

统计文件：

- `output/retest_assets_20260730/learn_dummy_arm_shadow/control_stat_res/dist_0_ours_default.yaml`
- `output/retest_assets_20260730/learn_dummy_arm_allegro/control_stat_res/dist_0_ours_default.yaml`
- `output/retest_assets_20260730/learn_dummy_arm_leap_tac3d/control_stat_res/dist_0_ours_default.yaml`

#### 复测结论

- 重新连接后的对象资源可以完整支持三套 example grasp evaluation dataset。
- 300 条控制仿真全部生成实际结果，不存在仅退出码为 0 的假阳性。
- 三种手的成功、失败和 invalid 数量以及成功率均与上次一致，说明重新连接资源没有改变本组 example grasp 的评测结果。
