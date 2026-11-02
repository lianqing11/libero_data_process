# libero_data_process

将原始 LIBERO demonstration 转换为双视角、绝对动作轨迹的训练 HDF5，并生成 LIBERO-Plus camera/action 等扰动数据。

**完整说明：[数据下载、环境准备与转换指南](docs/libero_data_preparation.md)。**

## 功能

- 普通 LIBERO：读取 `data/demo_i/actions` 和 `states`，恢复环境、过滤 no-op、回放并保存成功轨迹。
- Plus：`language`、`light`、`background`、`add`、`noise`、`camera`。
- Camera++：外部相机扰动、手腕图像翻转与 FOV 变化。
- Action：回放前插入一次 delta 偏移动作，提供三个强度等级及 star 版本。
- 自动生成 `meta.json` 和 `generation_report.json`。
- 提供独立的 Hugging Face 版本清单、网络路由探测和下载校验工具。

## 仓库结构

```text
scripts/prepare_libero_data.py    # 自包含转换入口；不导入原训练仓库
requirements.txt               # numpy / h5py
requirements-download.txt      # 可选下载工具依赖
tools/hf_manifest.py            # 解析固定版本、枚举文件
tools/probe_route.py            # 小范围下载探测
tools/verify_snapshot.py        # 文件大小、LFS SHA256 与未完成文件检查
docs/libero_data_preparation.md # 中文完整说明
tests/test_prepare_libero_data.py
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
python -m unittest discover -s tests -v
python -m compileall -q scripts tools tests
```

测试使用模拟环境对象验证回放顺序、图像翻转、no-op/夹爪处理、结束条件和 HDF5/metadata 契约，不需要 GPU、LIBERO 或 robosuite。真实 MuJoCo 端到端回放尚未验证，正式使用前应先做单任务 smoke。

本仓库由 `x_vla_ic` 中整理的独立脚本与说明拆出，不包含数据、模型权重或原训练项目的运行目录。上游 LIBERO/Plus 的代码和数据继续遵循各自授权。
