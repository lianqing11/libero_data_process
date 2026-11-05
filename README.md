# libero_data_process

将原始 LIBERO demonstration 转换为双视角、绝对动作轨迹的训练 HDF5，生成 LIBERO-Plus camera/action 等扰动数据，并使用本地 VLA checkpoint 运行普通 LIBERO、Plus suites 和 camera/action 运行时扰动评测。

**完整说明：[数据下载、环境准备与转换指南](docs/libero_data_preparation.md)。**

**评测说明：[资源下载 → 训练格式转换 → 模型评测与结果解读](docs/libero_evaluation.md)。**

## 功能

- 普通 LIBERO：读取 `data/demo_i/actions` 和 `states`，恢复环境、过滤 no-op、回放并保存成功轨迹。
- Plus：`language`、`light`、`background`、`add`、`noise`、`camera`。
- Camera++：外部相机扰动、手腕图像翻转与 FOV 变化。
- Action：回放前插入一次 delta 偏移动作，提供三个强度等级及 star 版本。
- 自动生成 `meta.json` 和 `generation_report.json`。
- 提供独立的 Hugging Face 版本清单、网络路由探测和下载校验工具。
- 独立评测入口与模型代码，保留 XVLA / MoT / IC 历史构造，支持 Plus 七类环境套件和 camera/action/star 运行时扰动。
- 评测输出逐 episode 记录、macro/micro 成功率、代码来源与完成/失败 manifest。

## 仓库结构

```text
scripts/prepare_libero_data.py    # 自包含转换入口；不导入原训练仓库
scripts/evaluate_libero.py       # 独立评测入口；调用本仓库 evaluation/ 和 models/
evaluation/                     # 模型推理、仿真 rollout、扰动、结果记录
models/                         # 从当前 XVLA 项目抽出的模型/processor 源码
requirements.txt               # numpy / h5py
requirements-download.txt      # 可选下载工具依赖
requirements-eval.txt          # 可选模型评测依赖，仿真依赖另行准备
tools/hf_manifest.py            # 解析固定版本、枚举文件
tools/probe_route.py            # 小范围下载探测
tools/verify_snapshot.py        # 文件大小、LFS SHA256 与未完成文件检查
docs/libero_data_preparation.md # 中文完整说明
docs/libero_evaluation.md       # 下载、转换、评测命令、协议和结果文件
docs/evaluation_source_manifest.json # 原评测/模型源码的 SHA256
tests/test_prepare_libero_data.py
tests/test_evaluation_*.py
```

## 安装与依赖

Python 3.10+。在自己的虚拟环境中安装基础依赖：

```bash
git clone git@github.com:lianqing11/libero_data_process.git
cd libero_data_process
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

基础依赖可用于读取、校验 HDF5 和运行无需仿真的测试。实际 `generate` 还需要兼容的 **robosuite、LIBERO / LIBERO-Plus、MuJoCo 和离屏渲染环境**；`.pruned_init` 加载额外需要 torch。这些仿真依赖按所使用的 LIBERO 版本安装，详见完整说明。不要把这一步理解为已经装好整个仿真环境。

下载辅助工具可在独立环境中安装 `requirements-download.txt`；文档也提供隔离的 `uv run --no-project` 用法。不要为下载工具升级共享训练环境。

## 最小使用示例

以下 `python` 指向已配置仿真依赖的解释器。数据和输出放在仓库外，路径按自己的机器修改。

```bash
# 单任务、单 demo 的普通 LIBERO 转换
python scripts/prepare_libero_data.py generate \
  --input /path/to/LIBERO-datasets/libero_spatial/task_demo.hdf5 \
  --suite libero_spatial --libero-root /path/to/LIBERO-plus \
  --modes base --max-demos 1 \
  --output-root /path/to/data/libero_base_smoke

# camera / camera++ / action，每条 demo 尝试 5 个变体
python scripts/prepare_libero_data.py generate \
  --input /path/to/LIBERO-datasets/libero_spatial \
  --libero-root /path/to/LIBERO-plus \
  --modes camera camera_plusplus action \
  --variants-per-demo 5 --action-level 2 \
  --output-root /path/to/data/libero_camera_action

# 对已转换的数据重新校验并生成 meta（不需要仿真依赖）
python scripts/prepare_libero_data.py meta \
  --root /path/to/data/libero_camera_action
```

输入可以是一个原始任务 HDF5、一个 suite 目录或数据根目录。根目录会递归处理所有 `*_demo.hdf5`；`--suite` 是覆盖推断的参数，不是 suite 过滤器。输出目录必须为空或不存在。

## 评测示例

先按 [评测指南](docs/libero_evaluation.md#4-评测环境和模型准备) 配置独立环境、模型资源和 LIBERO-Plus assets，再做一个任务、一个 episode 的 smoke：

```bash
.venv/bin/python scripts/evaluate_libero.py \
  --model_path /path/to/checkpoint \
  --libero_root /path/to/LIBERO-plus \
  --sim libero --task_suites libero_spatial \
  --episodes 1 --max_tasks 1 --save_video \
  --output_dir evaluation_outputs/base-smoke
```

仅运行 camera++ 扰动时，改用 `--sim libero_plus --perturbation camera_plusplus --runtime_perturbations none`。仅测 action 时将 perturbation 换为 `action`，并用 `--perturbation_level 1/2/3` 控制强度。完整四 suite 和 Plus 七类 suite 的命令见评测指南。

评测读取 checkpoint、BDDL、初始状态和环境资产，不读取训练 HDF5。Plus 默认额外在四个原始 suites 上跑 camera++ 和 action；只测指定 suite 时显式使用 `--runtime_perturbations none`。输出根目录下按 checkpoint 名创建子目录，要求该实际目录为空；结果查看 `run_manifest.json` 和 `summary.json`。

## 输出约定

每条成功轨迹写一个 HDF5；执行 N 个有效动作后：

| 字段 | 内容 |
|---|---|
| `observation/agentview_rgb` | `[N+1, H, W, 3]`；默认 224×224，旋转 180° |
| `observation/eye_in_hand_rgb` | `[N+1, H, W, 3]`；按模式处理手腕翻转 |
| `abs_action_6d` | `[N+1, 7]`；xyz + axis-angle + gripper |
| `rewards`、`dones` | `[N]`，末项为 1 |
| attrs | 指令、任务名、扰动参数、来源和随机种子 |

`abs_action_6d` 使用历史字段名，磁盘里的旋转是三维轴角。目标位姿来自控制器 `goal_pos/goal_ori`。rot6d 转换和 action chunk 切分由下游训练读取器完成。

## 数据来源与实现边界

- 原始 demo：[yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)。
- 仿真环境：[LIBERO-Plus 官方代码](https://github.com/sylvestf/LIBERO-plus)。
- Plus 环境资产：[Sylvest/LIBERO-plus](https://huggingface.co/datasets/Sylvest/LIBERO-plus)。
- 下载过程采用镜像优先、固定 commit 和下载后校验，命令见完整说明。

`action` 采用回放前的一次 delta 偏移；与评测时逐步给绝对动作施加固定偏置的实现应分别描述。`camera_plusplus_star` 在此生成路径中扩大手腕 FOV，外部相机仍使用标准离散配置。各类别分别生成，不包含旧 combined 随机叠加、robot 套件、视频导出、并行 worker 或生成续跑。

## 验证

```bash
USE_TF=0 python -m unittest discover -s tests -v
python -m compileall -q scripts tools tests evaluation models
```

14 项离线测试覆盖转换契约、IC 历史隔离、action chunk / 夹爪阈值、评测配置与结果统计，不需要 GPU、LIBERO 或 robosuite。模型测试需要 torch/transformers，基础环境会跳过这三项。真实 MuJoCo 回放及加载实际 checkpoint 的闭环评测尚未验证；共享仿真环境的日志权限诊断见评测指南，正式使用前应先做单任务 smoke。

本仓库由 `x_vla_ic` 中整理的独立脚本与说明拆出，不包含数据、模型权重或原训练项目的运行目录。上游 LIBERO/Plus 的代码和数据继续遵循各自授权。
