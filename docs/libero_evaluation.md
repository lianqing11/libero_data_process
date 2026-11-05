# LIBERO / LIBERO-Plus：下载、训练格式转换与模型评测

本指南对应独立仓库 `libero_data_process`，整理日期为 2026-11-05。转换入口是 [`scripts/prepare_libero_data.py`](../scripts/prepare_libero_data.py)，评测入口是 [`scripts/evaluate_libero.py`](../scripts/evaluate_libero.py)。所有命令在仓库根目录执行。

## 1. 整个流程及资源来源

```text
HF 原始 LIBERO demonstrations（actions、states）
    + LIBERO-Plus 仿真代码、BDDL、init states、assets
    → 仿真回放 / camera、action 等扰动
    → 每条成功轨迹一个 HDF5 + meta.json
    → 外部训练程序训练 → 本地模型 checkpoint

本地模型 checkpoint
    + LIBERO / LIBERO-Plus 的 BDDL、init states、assets
    → 在线闭环仿真 → 每条 episode 的成功/失败、步数、可选视频
```

训练 HDF5 是训练程序的输入。评测直接使用仿真任务和初始状态，在环境中执行模型动作，**不读取转换后的训练 HDF5，也不需要为了评测下载全部 demonstrations**。本仓库提供转换和评测，不包含训练入口或 checkpoint。

| 资源 | 下载来源 | 用于哪一步 |
|---|---|---|
| 原始 LIBERO HDF5 demos | [yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) | 转换训练轨迹 |
| 标准 LIBERO 代码、任务定义和初始状态 | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | 标准评测；使用原版时遵循其资产准备说明 |
| Plus 代码、BDDL、初始状态 | [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) | Plus 转换、Plus 评测、运行时扰动；也包含普通 suites |
| Plus mesh / texture 等场景资产 | [Sylvest/LIBERO-plus](https://huggingface.co/datasets/Sylvest/LIBERO-plus) 中的 `assets.zip` | 仿真环境 |
| XVLA checkpoint | 自己的训练产物：`config.json` + HF 模型权重/分片索引 | 模型评测 |
| backbone 的 config / tokenizer / processor 等资源 | checkpoint 的 `qwen3_pretrained` 或 `paligemma_pretrained` 指向的模型仓库/本地目录 | 构建并加载 XVLA；随 checkpoint 配置确定 |

Plus 官方还提供 RLDS / LeRobot 数据。这里的转换器实现的是 **原始 LIBERO HDF5 回放生成** 路径，未实现 RLDS / LeRobot 格式导入。Plus 的 `assets.zip` 本身不是转换好的训练轨迹。

## 2. 数据如何下载、放在哪里

完整可执行步骤集中在 [数据指南第 3–4 节](libero_data_preparation.md#3-下载原始-libero-和-plus-资产)，包括 `hfd.sh`、镜像探测、固定 revision、文件选择、续传、SHA256 和解压验收。避免维护两份不同的下载流程。

建议沿用这些目录：

```bash
export LIBERO_RAW=/VLA-Data/scripts/lianqing/data/yifengzhu-hf/LIBERO-datasets
export LIBERO_ASSETS=/VLA-Data/scripts/lianqing/data/libero-plus-assets
export LIBERO_PLUS_ROOT=/path/to/LIBERO-plus
export CHECKPOINT=/VLA-Data/scripts/lianqing/checkpoints/my-libero-model/checkpoint-final
```

下载顺序：

1. `git clone https://github.com/sylvestf/LIBERO-plus.git "$LIBERO_PLUS_ROOT"`，记录 checkout commit 和本地修改。
2. 按数据指南用 `hfd.sh --dataset --revision <固定commit>` 获取原始 demos（需要转换训练数据时）以及 `Sylvest/LIBERO-plus` 的 `assets.zip`。
3. 先清除 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` 及小写变量，探测 `https://hf-mirror.com`。仅镜像缺失或不可用时按数据指南回退官方源，保留相同 revision 和哈希标准；不要用 `proxy-set` 传输 HF 数据。
4. 用随仓库提供的 `tools/verify_snapshot.py` 对照 manifest 验证文件大小、LFS SHA256、下载 sidecar，再执行 `unzip -t`。
5. 解压到仓库外，将其 `assets/` 目录链接到 `$LIBERO_PLUS_ROOT/libero/libero/assets`。已有目录先检查，保留现有内容。确认同级有 `bddl_files/` 与 `init_files/`。

本机历史原始数据下载记录是 `hfd.sh + aria2c`，缓存 revision 为 `f13aa24a3da8c43c7225569f28c562979fa0e35a`，证据范围见 [数据指南第 2 节](libero_data_preparation.md#2-本机已有数据当时如何下载)。本次整理没有重新下载数据、模型或验证全量资产哈希。

如果需要下载 backbone，使用同一套工具的 `--repo-type model`，`hfd.sh` 不带 `--dataset`，固定 checkpoint 对应的模型 revision，目标设为 `/VLA-Data/scripts/lianqing/checkpoints/<resource-name>/`。本指南不替用户选择或更新 backbone 版本。

## 3. 如何转换成训练格式

在已配置仿真依赖和 LIBERO 路径的环境中执行：

```bash
# 单个 suite 的普通训练数据
.venv/bin/python scripts/prepare_libero_data.py generate \
  --input "$LIBERO_RAW/libero_spatial" \
  --suite libero_spatial --libero-root "$LIBERO_PLUS_ROOT" \
  --modes base \
  --output-root /VLA-Data/scripts/lianqing/data/libero-base

# 同一批原始 demos 分别生成 camera / camera++ / action 变体
.venv/bin/python scripts/prepare_libero_data.py generate \
  --input "$LIBERO_RAW/libero_spatial" \
  --suite libero_spatial --libero-root "$LIBERO_PLUS_ROOT" \
  --modes camera camera_plusplus action \
  --variants-per-demo 5 --action-level 2 \
  --output-root /VLA-Data/scripts/lianqing/data/libero-variants

.venv/bin/python scripts/prepare_libero_data.py meta \
  --root /VLA-Data/scripts/lianqing/data/libero-variants
```

首次应加 `--max-demos 1`，选择一个任务文件作为 `--input` 做 smoke。输出根目录必须为空或不存在。转换器的 `--libero-root` 使用连字符；评测器的 `--libero_root` 使用下划线。

转换步骤是读取 `data/demo_i/actions` 与 `states`、恢复初始状态、settle、施加选定扰动、过滤 no-op、执行原始 delta 动作，并记录控制器的绝对目标位姿。只保存成功轨迹，失败或异常写入 `generation_report.json`。转换器的 LIBERO 配置准备方式见 [数据指南第 4 节](libero_data_preparation.md#4-准备-libero-plus-环境)；评测器则自动创建隔离的路径配置。

执行 N 个有效动作后：

| HDF5 字段 | 形状和含义 |
|---|---|
| `observation/agentview_rgb` | `[N+1,H,W,3]`，默认 224×224，旋转 180° |
| `observation/eye_in_hand_rgb` | `[N+1,H,W,3]`，按模式处理手腕视角 |
| `abs_action_6d` | `[N+1,7]`，xyz + 三维 axis-angle + gripper；字段名是历史命名 |
| `rewards` / `dones` | `[N]`，末项为 1 |
| attrs | 指令、任务、扰动、来源、seed 等 |

下游训练读取器负责将 axis-angle 转为 rot6d、构造 proprio 和 action chunks。本仓库未复制训练数据加载器。`meta.json` 记录生成文件清单。

## 4. 评测环境和模型准备

推荐 Python 3.10，独立 `.venv`。先按 GPU/CUDA 环境安装匹配的 torch / torchvision，然后安装评测依赖和仿真依赖：

```bash
python3 -m venv .venv
# 本机参考版本为 torch 2.8.0 / torchvision 0.23.0；按自己的 CUDA 选安装源。
.venv/bin/python -m pip install 'torch==2.8.0' 'torchvision==0.23.0'
.venv/bin/python -m pip install -r requirements-eval.txt
# 以下是基础仿真依赖；安装后还需按 LIBERO-Plus 文档补齐系统渲染库。
.venv/bin/python -m pip install 'robosuite==1.4.0' 'mujoco==3.3.7' \
  'bddl==3.6.0' 'gym==0.26.2' 'opencv-python==4.11.0.86' \
  easydict cloudpickle einops matplotlib numba wand scikit-image
.venv/bin/python -m pip install --no-deps -e "$LIBERO_PLUS_ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export USE_TF=0
.venv/bin/python scripts/evaluate_libero.py --help
```

上述仿真版本来自当前共享环境的包元数据盘点，**尚未在新 `.venv` 中完成依赖安装与真实仿真验证**。Plus 的旧 `requirements.txt` 固定了 transformers 4.21.1，与这里的 Qwen3-VL / transformers 4.57.3 不兼容，安装时须保留 `requirements-eval.txt` 的模型依赖版本。系统 EGL/OpenGL、ImageMagick 等依赖按 Plus 官方安装说明准备。不要向共享 `X-VLA/.venv` 安装或升级包。

checkpoint 必须是可由 `XVLA.from_pretrained()` 加载的本地模型目录。只有 DeepSpeed optimizer shards 的训练目录需要先由原训练项目导出为 HF 模型目录；本仓库不做 checkpoint 合并。LoRA 合并可用 `--lora_path`，另装匹配版本的 `peft`；配置启用 flash attention 时需相应的 `flash-attn`。

模型类及 processor 已收入本仓库 `models/`，运行时不依赖原训练项目的 Python 源码。仍需要 backbone 的 tokenizer / processor / config 等本地资源。例如 checkpoint 的 `qwen3_pretrained` 为 `Qwen/Qwen3-VL-2B-Instruct`，可以设置：

```bash
export VLA_MODEL_ROOTS=/VLA-Data/scripts/lianqing/checkpoints
# 解析到 $VLA_MODEL_ROOTS/Qwen/Qwen3-VL-2B-Instruct
```

多个查找根目录以冒号分隔。若 checkpoint 保存的是已不存在的绝对路径，应在评测副本的 config 中改成准确的本地资源路径，并记录修改。`--processor_path` 用于加载后的 processor 选择，不能替代模型构造阶段的 backbone 路径。解析器还会查询本地 HF cache；为了可复现，正式实验应使用固定 snapshot 路径，避免多份缓存中的版本歧义。

评测入口强制启用 HF 离线模式。缺少资源时直接失败，不在评测过程中发起模型下载。每个运行目录自动生成 `.libero_config/config.yaml` 和 `config_plus.yaml`，隔离 BDDL / init states / assets 路径，不修改 `~/.libero`。

## 5. 怎么运行评测

### 5.1 先做单任务 smoke

```bash
.venv/bin/python scripts/evaluate_libero.py \
  --model_path "$CHECKPOINT" --libero_root "$LIBERO_PLUS_ROOT" \
  --sim libero --task_suites libero_spatial \
  --episodes 1 --max_tasks 1 --device cuda \
  --act_type abs --domain_id 3 --denoising_steps 10 \
  --output_dir evaluation_outputs/base-smoke --save_video
```

这验证模型加载、图像输入、动作形状、仿真和输出链路。单条轨迹成功率不能代表整体模型能力。`--domain_id`、动作表示、chunk 长度和 IC 设置需与训练配置一致；这里的 domain 3 / abs 是现有 LIBERO XVLA 路径的默认值。

### 5.2 普通 LIBERO 四个 suite

```bash
.venv/bin/python scripts/evaluate_libero.py \
  --model_path "$CHECKPOINT" --libero_root "$LIBERO_PLUS_ROOT" \
  --sim libero \
  --task_suites libero_spatial libero_goal libero_object libero_10 \
  --episodes 10 --seed 42 --act_type abs \
  --output_dir evaluation_outputs/base-four-suites
```

`--episodes` 是**每个任务**的 episode 数；不指定 `--max_tasks` 时跑所选 suite 的全部任务。标准 client 的 `--max_tasks N` 取前 N 个任务，Plus client 使用等间距任务抽样 `floor(i * task_count / N)`。两个 client 的子集结果不能默认视为同一任务覆盖。

### 5.3 LIBERO-Plus 七类环境扰动

```bash
.venv/bin/python scripts/evaluate_libero.py \
  --model_path "$CHECKPOINT" --libero_root "$LIBERO_PLUS_ROOT" \
  --sim libero_plus \
  --task_suites libero_plus_camera libero_plus_robot libero_plus_language \
                libero_plus_light libero_plus_background libero_plus_noise libero_plus_layout \
  --episodes 10 --seed 42 --runtime_perturbations none \
  --output_dir evaluation_outputs/plus-seven
```

这些名称由 Plus checkout 的 benchmark registry 提供。环境从相应 BDDL 和初始状态创建。也支持如 `libero_spatial_camera` 这样的细粒度 suite，以 checkout 实际注册项为准。训练转换器没有 robot 模式，评测仍可运行 Plus 的 robot suite；训练转换范围与评测覆盖范围不同。

### 5.4 单独测 camera++ / action

```bash
.venv/bin/python scripts/evaluate_libero.py \
  --model_path "$CHECKPOINT" --libero_root "$LIBERO_PLUS_ROOT" \
  --sim libero_plus --task_suites libero_spatial \
  --perturbation camera_plusplus --runtime_perturbations none \
  --episodes 1 --max_tasks 1 \
  --output_dir evaluation_outputs/camera-smoke --save_video

.venv/bin/python scripts/evaluate_libero.py \
  --model_path "$CHECKPOINT" --libero_root "$LIBERO_PLUS_ROOT" \
  --sim libero_plus \
  --task_suites libero_spatial libero_goal libero_object libero_10 \
  --perturbation action --perturbation_level 2 --runtime_perturbations none \
  --episodes 10 --output_dir evaluation_outputs/action-four-suites
```

`--perturbation` 只给当前 `--task_suites` 加一种扰动；必须同时指定 `--runtime_perturbations none`，避免意外叠加额外评测 passes。运行时扰动要求 `--sim libero_plus --act_type abs`。

`--runtime_perturbations camera_plusplus action` 会在当前 suites 评测完成后，额外对**四个原始 base suites** 分别跑 camera++ 和 action。此额外 suite 列表固定，与 `--task_suites` 无关。`--sim libero_plus` 不指定该参数时，默认就是这两个额外 passes。`all` 包含 camera_star、camera_plusplus、camera_plusplus_star、action、action_star；`none` 关闭额外 passes。每种实验要用不同输出目录。

## 6. 评测执行协议与训练扰动的区别

每条 episode 的步骤：

1. 读取任务 BDDL / language，创建 256×256 双视角环境。
2. 环境 seed 使用 `seed + ep + 100`，初始状态索引为 `ep % num_init_states`。
3. 执行 10 次 `[0,0,0,0,0,0,-1]` settle，再将控制器切到绝对目标模式。
4. 清空 policy 的 action queue、proprio 和 IC 历史。读取实际末端位置/姿态作为闭环 proprio；agentview 旋转 180°，手腕按模式处理。
5. 模型预测 action chunk；ee6d 输出取第一个机械臂，将 rot6d 转为轴角。gripper `>0.5` 映射为 `+1`，否则为 `-1`（阈值可配置）。依次执行 chunk，队列清空后重新推理。`--action_chunk_size` 可截断每次预测的执行长度，默认执行完整模型 chunk。
6. LIBERO `env.step()` 返回 `done` 时记成功并结束；达到 horizon 记失败。这里沿用该环境的 success/done 语义。

现有 horizon：spatial / goal / object 为 800，libero_10 为 900；Plus 合并的 `libero_plus_*` 为 800，按 base 命名的细分 suite 继承 base horizon。步数统计不含 settle，但失败 episode 按完整 horizon 计入平均步数。这些是本代码的协议，比较其他论文或评测脚本前须对齐。

| 机制 | 训练数据转换 | 运行时评测 |
|---|---|---|
| `camera` | 用 Plus camera 任务/参数生成回放轨迹 | `libero_plus_camera` 等 BDDL suite |
| `camera_plusplus` | 外部相机离散采样 + wrist flip / FOV | 每个 episode 固定相机配置与 wrist 变换 |
| `camera_plusplus_star` | 此生成路径主要扩大 wrist FOV 到 50–100°，agentview 仍标准离散集合 | 扩大 agentview 与 wrist 的扰动范围；不保证与训练分布一致 |
| `action` | settle 后插入一次固定 delta 动作，再回放原 demo | 每条 episode 采一次 bias，对每一步绝对目标执行平移加法、`bias * target` 旋转合成，夹爪保持原值 |

评测 action level 1/2/3 对应每轴位置范围 ±5/10/20 mm，旋转角范围 ±1/3/6°；action_star 对应 ±15/30/50 mm 和 ±5/10/15°。bias 的随机 seed 是 `seed + task_id * 1000 + ep`。camera 扰动修改渲染后直接刷新 observation，不额外执行机器人动作。

IC 默认从 checkpoint 读取 `num_ic_frames` 与 `include_next_obs`。`K` 包括当前 query，最多使用 `K-1` 个本 episode 已记录的历史条目；历史不足时使用已有条目。`K=0` 是 baseline，`K=1` 是 query-only IC。历史单位为模型调用/action chunk，`--ic_stride` 按历史条目采样；chunk 结束时记录实际 next observation / proprio。`--clone_query_as_demo`、`--zero_ic_actions`、`--ic_demo_mode fixed_first/early` 是诊断参数，不应混入默认正式结果。

## 7. 结果文件与完成判定

若 checkpoint 为 `/checkpoints/experiment/checkpoint-10000`，`--output_dir evaluation_outputs/base` 的实际目录为：

```text
evaluation_outputs/base/experiment_checkpoint-10000/
  run_manifest.json              # complete / failed、参数、代码哈希和路径来源
  summary.json                   # 全部请求的 suites/passes 完成后生成
  results.json                   # 当前 dataset pass 的 suite 汇总
  <suite>/
    episodes.jsonl               # 逐 episode 日志，末尾还包含 suite 汇总行
    raw_episodes.json             # 本 suite 完成后的逐 episode 结构化记录
    *.mp4                        # 仅 --save_video 时输出
    gripper_curve_*.csv          # policy 产生夹爪轨迹时输出
  runtime_<perturbation>/...     # 额外 runtime pass
  .libero_config/                # 本次运行隔离的仿真路径配置
```

输出实际目录必须为空或不存在，避免不同参数/不同 checkpoint 的结果混写。当前不支持断点续评。

`raw_episodes.json` 记录 suite、实际 benchmark 类、task name / id、episode、成功标记和步数。`summary.json` 同时给出各 suite 成功率、成功次数、episode 数、平均步数，以及 suite 等权的 `macro_success_rate` 和 episode 加权的 `micro_success_rate`。Plus 不同 suites 任务数可能不同，两种均值不能混用。代码会检查实际 episode 数与任务数 × episodes 是否一致。

完整结果要求 `run_manifest.json` 的 `status=complete`，且 `summary.json` 中覆盖预期的 suites / runtime passes。异常会记录 `status=failed` 并重新抛出，不能把未完成评测当作 0% 模型成功率。视频默认关闭，可先开视频排查 smoke。

manifest 记录本仓库 Git commit、evaluation/models 的逐文件 SHA256、LIBERO Git commit、checkpoint config SHA256 和顶层文件名/大小。`weights_hash_verified=false` 明确表示**没有读取全部模型权重做哈希**；也未验证所有 BDDL、init states、assets 的内容。正式实验需另外保存 checkpoint 分片 SHA256、资源下载 verification、LIBERO 的本地 diff 和环境版本。相同 seed 也不保证不同驱动/仿真版本逐位一致。

## 8. 代码位置与本次整理范围

| 独立仓库代码 | 职责 | 原实现位置（相对 `x_vla_ic`） |
|---|---|---|
| `scripts/prepare_libero_data.py` | 单文件训练数据转换、meta 校验 | 上一次已整理的独立转换器 |
| `scripts/evaluate_libero.py` | 统一评测命令入口 | 本次新增封装 |
| `evaluation/scripts/evaluate_libero.py` | 模型加载、动作队列、IC history、评测编排 | `x_vla_fast/evaluation/scripts/evaluate_libero.py` |
| `evaluation/libero/libero_client.py` | 标准 LIBERO rollout | `x_vla_fast/evaluation/libero/libero_client.py` |
| `evaluation/libero/libero_plus_client.py` | Plus suites 和运行时扰动 rollout | `x_vla_fast/evaluation/libero/libero_plus_client.py` |
| `evaluation/libero/runtime_perturbations.py` | camera / action / star 采样及应用 | `x_vla_fast/evaluation/libero/runtime_perturbations.py` |
| `evaluation/run_support.py` | 参数检查、隔离配置、manifest、汇总 | 本次新增 |
| `models/` | XVLA / MoT / IC 与 processor 的模型加载实现，保留 OpenVLA 路径 | `x_vla_fast/models/` |

源文件提取前的 SHA256 和来源 HEAD 在 [`evaluation_source_manifest.json`](evaluation_source_manifest.json)。该清单记录原工作区文件，来源 HEAD 不代表原工作区没有未提交改动。仓库附带原 Apache-2.0 LICENSE 和来源 NOTICE。

本次可移植性和可靠性调整：去掉硬编码 checkout/backbone 目录；隔离 LIBERO 配置；规范 JSON/JSONL；拒绝参数冲突与结果覆盖；记录完成状态和 episode 数；异常时关闭已创建的 rollout 环境。另修正了相机刷新时原零动作可能改变绝对目标的问题，以及 camera++ 重用旧 observation 时重复翻转的问题（现在该模式每步刷新图像）。这些变化可能影响旧 camera++ 结果，旧结果与新结果比较时应记录代码版本。

验证命令：

```bash
USE_TF=0 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q scripts tools tests evaluation models
USE_TF=0 .venv/bin/python scripts/evaluate_libero.py --help
```

本次 14 项离线测试通过，覆盖原转换契约、IC 历史选择/清空、action chunk 执行/夹爪阈值、参数拒绝、隔离配置、完成/失败记录、macro/micro 统计。模型源码导入、CLI help 和语法检查通过。测试未加载实际 checkpoint，未运行真实 MuJoCo episode。

当前机器只读检查还发现共享 robosuite 导入试图写 `/tmp/robosuite.log` 时 `PermissionError`。本次未修改共享环境或该日志；运行仿真前需在自己的仿真环境配置可写日志或关闭 robosuite 文件日志，再验证 EGL 和单任务 smoke。离线测试通过不代表此环境问题已解决。
