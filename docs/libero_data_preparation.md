# LIBERO / LIBERO-Plus 数据下载与训练格式转换

本文包含作者机器上的历史路径示例，换机器时请替换数据、环境和 LIBERO checkout 路径。

本文对应独立脚本 [`scripts/prepare_libero_data.py`](../scripts/prepare_libero_data.py)。代码及本机下载记录核对日期：2026-11-02。

脚本完成：**原始 LIBERO demonstration → 仿真回放及可选扰动 → 一条轨迹一个 HDF5 → 训练数据清单 `meta.json`**。脚本不依赖本仓库其他 Python 模块，但需要 LIBERO/robosuite 仿真依赖。

2026-11-05 补充：仓库已加入独立评测入口、模型源码和 [下载—转换—评测完整说明](libero_evaluation.md)。评测直接读取仿真任务/初始状态与模型 checkpoint，训练 HDF5 用于训练。

## 1. 要准备哪几类数据

| 资源 | 来源 | 用途 |
|---|---|---|
| 原始 LIBERO demonstrations | [yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/main) | 提供原始 `actions`、`states` 和任务信息，是本脚本输入 |
| LIBERO-Plus 仿真代码、BDDL、初始状态定义 | [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) | 创建具有相机、语言、物体等变化的环境 |
| LIBERO-Plus 物体与纹理资产 | [Sylvest/LIBERO-plus 的 assets.zip](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main) | 为 Plus 场景提供 mesh、纹理等资源 |
| 我们的训练 HDF5 | 由本脚本回放生成 | 双视角图像、绝对目标位姿和语言指令 |

官方还提供 [RLDS](https://huggingface.co/datasets/Sylvest/libero_plus_rlds)、[LeRobot](https://huggingface.co/datasets/Sylvest/libero_plus_lerobot) 和 [按 suite 划分的数据](https://huggingface.co/datasets/Sylvest/libero_plus_data_4suite)。当前脚本直接读取原始 LIBERO 的 `data/demo_i/actions` 与 `states`；没有实现 RLDS/LeRobot reader。使用本文流程时下载原始 demo 和 Plus 环境资产即可。

`camera_plusplus`、`action` 等训练轨迹在本地回放时生成。`assets.zip` 是环境资源，解压后还需要执行转换。

## 2. 本机已有数据当时如何下载

原始数据位置：

```text
/VLA-Data/scripts/lianqing/data/yifengzhu-hf/LIBERO-datasets/
```

本次读取的证据文件：

```text
.hfd/last_download_command
.hfd/repo_metadata.json
```

记录表明使用了 `hfd.sh` 的 `aria2c` 后端，`REPO_ID=yifengzhu-hf/LIBERO-datasets`，开启 dataset 模式，当时请求 `main`，没有 include/exclude 过滤。缓存中的具体版本是：

```text
f13aa24a3da8c43c7225569f28c562979fa0e35a
```

这能确认下载工具、仓库和缓存版本；没有从这些记录确认当时使用的网络路由，也没有在本次文档工作中重新验证全量文件哈希。下文给出固定版本的可复现下载流程。

本机现有 Plus checkout：

```text
/VLA-Data/scripts/lianqing/projects/vla/X-VLA/libero_plus
```

本次读到其 Git HEAD 为 `075db8aa1b09c8741e47adbd1b2458657eea0f56`。HEAD 只表示基础提交，不涵盖本地修改；复现实验还应保存 `git diff`。其 `libero/libero/assets/` 下已有物体与纹理目录，目录存在不等同于完成完整性验收。

## 3. 下载原始 LIBERO 和 Plus 资产

以下代码是待执行的操作说明。本次编写文档没有启动数据下载。数据放在仓库外部；先用 `df -h` 检查空间，原始数据、资产压缩包、解压内容及重新渲染的轨迹都需要占用空间。

### 3.1 下载工具与网络

Hugging Face 先清除代理变量、检查国内镜像，再下载。不要使用 `proxy-set` 选择的代理传输 HF 文件。

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT=https://hf-mirror.com

export LIBERO_RAW=/VLA-Data/scripts/lianqing/data/yifengzhu-hf/LIBERO-datasets
export LIBERO_ASSETS=/VLA-Data/scripts/lianqing/data/libero-plus-assets
export LIBERO_DL_TOOLS=/VLA-Data/scripts/lianqing/downloads/libero-download-tools
export HF_HELPERS="$(pwd)/tools"
mkdir -p "$LIBERO_DL_TOOLS" "$LIBERO_RAW" "$LIBERO_ASSETS"
df -h "$LIBERO_RAW"

# 如已有审核过的 hfd.sh，直接设置 HFD 指向它。
# 没有时，从镜像站获取；下载后先查看脚本内容，再运行。
curl -fL --connect-timeout 10 --max-time 60 \
  https://hf-mirror.com/hfd/hfd.sh -o "$LIBERO_DL_TOOLS/hfd.sh"
export HFD="$LIBERO_DL_TOOLS/hfd.sh"
sha256sum "$HFD" > "$LIBERO_DL_TOOLS/hfd.sha256"
bash "$HFD" --help
command -v aria2c
command -v jq
```

`hfd.sh` 需要其帮助文档列出的系统命令。本例使用 `aria2c`；缺少这些工具时先在自己的环境安装。不要修改共享 X-VLA Python 环境。下面的 manifest 工具通过 `uv run --no-project` 使用独立缓存环境，避免改变训练环境的 `huggingface_hub` 版本。

`HF_HELPERS` 指向本仓库的 `tools/`，请在仓库根目录设置该变量。三个下载验证工具已随仓库提供；转换脚本本身不依赖它们。

### 3.2 固定版本并生成远端文件清单

原始 LIBERO 固定到本机历史缓存版本；Plus 资产在下载前将 `main` 解析成不可变 commit，后续始终使用该 commit。

```bash
uv run --no-project --with 'huggingface_hub>=1,<2' --with httpx python \
  "$HF_HELPERS/hf_manifest.py" \
  --repo-id yifengzhu-hf/LIBERO-datasets --repo-type dataset \
  --revision f13aa24a3da8c43c7225569f28c562979fa0e35a \
  --endpoint "$HF_ENDPOINT" --direct \
  --output "$LIBERO_RAW/manifest.full.json"

uv run --no-project --with 'huggingface_hub>=1,<2' --with httpx python \
  "$HF_HELPERS/hf_manifest.py" \
  --repo-id Sylvest/LIBERO-plus --repo-type dataset \
  --revision main --endpoint "$HF_ENDPOINT" --direct \
  --output "$LIBERO_ASSETS/manifest.full.json"

export LIBERO_RAW_REV=$(python3 -c \
  'import json,os; print(json.load(open(os.environ["LIBERO_RAW"]+"/manifest.full.json"))["revision"])')
export LIBERO_ASSETS_REV=$(python3 -c \
  'import json,os; print(json.load(open(os.environ["LIBERO_ASSETS"]+"/manifest.full.json"))["revision"])')
```

清单记录路径、大小、LFS SHA256 等信息。准备四个常用 suite 时，可以只选 `spatial/object/goal/10`；`libero_90` 不在下面的下载范围中。

```bash
python3 - <<'PY'
import json, os
from pathlib import Path

for variable, prefixes in [
    ('LIBERO_RAW', ('libero_spatial/', 'libero_object/', 'libero_goal/', 'libero_10/')),
    ('LIBERO_ASSETS', ('assets.zip',)),
]:
    root = Path(os.environ[variable])
    manifest = json.loads((root / 'manifest.full.json').read_text())
    files = [f for f in manifest['files']
             if (f['path'].endswith('_demo.hdf5') and f['path'].startswith(prefixes))
             or (variable == 'LIBERO_ASSETS' and f['path'] == 'assets.zip')]
    assert files, f'Empty selection: {variable}'
    manifest['selection'] = list(prefixes)
    manifest['files'] = files
    manifest['remote_files'] = len(files)
    manifest['remote_bytes'] = sum(f['size'] for f in files)
    (root / 'manifest.selected.json').write_text(json.dumps(manifest, indent=2))
    print(variable, len(files), manifest['remote_bytes'], 'bytes')
PY
```

先用清单中的一个真实 demo 路径做小范围镜像传输探测。替换 `<清单中的demo路径>`，不要带原始数据根目录：

```bash
uv run --no-project --with requests python "$HF_HELPERS/probe_route.py" \
  --repo-id yifengzhu-hf/LIBERO-datasets --repo-type dataset \
  --revision "$LIBERO_RAW_REV" --path '<清单中的demo路径>' \
  --endpoint "$HF_ENDPOINT" --route-label mirror --direct --bytes 1048576

uv run --no-project --with requests python "$HF_HELPERS/probe_route.py" \
  --repo-id Sylvest/LIBERO-plus --repo-type dataset \
  --revision "$LIBERO_ASSETS_REV" --path assets.zip \
  --endpoint "$HF_ENDPOINT" --route-label mirror --direct --bytes 1048576
```

只有镜像不可用、不完整或缺少指定版本/文件时，才回退官方 `https://huggingface.co`。回退时清除镜像覆盖，使用本环境指定的官方回源 HTTP 转发配置；不要调用 `proxy-set`。若使用显式回源代理，probe/manifest 命令要去掉 `--direct`。保留同一仓库、commit、文件选择和哈希验收标准；不要退回浮动的 `main`。代理地址和凭据不写入数据清单或下载报告。

### 3.3 下载与续传

```bash
bash "$HFD" yifengzhu-hf/LIBERO-datasets --dataset \
  --revision "$LIBERO_RAW_REV" \
  --include 'libero_spatial/*_demo.hdf5' 'libero_object/*_demo.hdf5' \
            'libero_goal/*_demo.hdf5' 'libero_10/*_demo.hdf5' \
  --local-dir "$LIBERO_RAW" --tool aria2c -x 4 -j 4

bash "$HFD" Sylvest/LIBERO-plus --dataset \
  --revision "$LIBERO_ASSETS_REV" --include 'assets.zip' \
  --local-dir "$LIBERO_ASSETS" --tool aria2c -x 4 -j 2
```

长下载在持久终端会话中执行。中断后重新执行同一命令续传，保留 `.hfd` 和下载器 sidecar；完成后再次执行同一命令检查是否幂等。不要把仍有 `.aria2`/`.incomplete` 的文件交给转换器。

如果需要 `libero_90`，同时修改 manifest 选择范围和 `hfd --include`，保持二者一致。原来的历史下载记录没有过滤；上面是按常用四个 suite 缩小范围的新下载示例。

### 3.4 验收

```bash
python3 "$HF_HELPERS/verify_snapshot.py" \
  "$LIBERO_RAW/manifest.selected.json" "$LIBERO_RAW" \
  --hash lfs --output "$LIBERO_RAW/verification.json"

python3 "$HF_HELPERS/verify_snapshot.py" \
  "$LIBERO_ASSETS/manifest.selected.json" "$LIBERO_ASSETS" \
  --hash lfs --output "$LIBERO_ASSETS/verification.json"

unzip -t "$LIBERO_ASSETS/assets.zip"
```

要求所选文件全部存在、大小正确、LFS SHA256 匹配、没有未完成 sidecar，且 ZIP CRC 检查通过。`verification.json` 的 `verified` 必须为 `true`。这只验收所选文件集合；不代表未选择的 `libero_90` 也已下载。

原始 demo 还应至少抽查一次 HDF5 结构：

```python
import h5py
with h5py.File('/path/to/task_demo.hdf5', 'r') as f:
    data = f['data']
    assert 'problem_info' in data.attrs
    assert 'bddl_file_name' in data.attrs
    for name in data:
        actions = data[name]['actions']
        states = data[name]['states']
        assert actions.ndim == 2 and actions.shape[1] == 7
        assert len(actions) > 0 and len(states) > 0
```

## 4. 准备 LIBERO-Plus 环境

已有本机 checkout 时优先复用。新机器可以从官方仓库获取代码：

```bash
git clone https://github.com/sylvestf/LIBERO-plus.git /path/to/LIBERO-plus
git -C /path/to/LIBERO-plus rev-parse HEAD
```

GitHub 网络配置与 HF 下载分开处理。记录代码 commit 和本地修改；不要在已有 checkout 中直接切换版本影响其他任务。

官方要求解压后形成 `libero/libero/assets/`。为保持数据位于仓库外部，可先解压到资产目录，再在新 checkout 中设置软链接：

```bash
mkdir -p "$LIBERO_ASSETS/extracted"
unzip -n "$LIBERO_ASSETS/assets.zip" -d "$LIBERO_ASSETS/extracted"
# 确认此路径下已有 articulated_objects、new_objects、textures 等目录。
ls "$LIBERO_ASSETS/extracted/assets"
```

仅在目标 `assets` 尚不存在时创建链接：

```bash
export LIBERO_PLUS_ROOT=/path/to/LIBERO-plus
# 在该 checkout 的适当 .gitignore 中添加精确规则 /libero/libero/assets。
# 已有资源目录应保留，先检查其内容，不要覆盖。
ln -sT "$LIBERO_ASSETS/extracted/assets" "$LIBERO_PLUS_ROOT/libero/libero/assets"
```

环境依赖以 [官方安装说明](https://github.com/sylvestf/LIBERO-plus#%EF%B8%8F-installation) 和本项目已验证环境为准。需要安装包时使用当前项目自己的 `.venv`，不要向共享的 `X-VLA/.venv` 执行 pip install。已有共享环境可以用于运行和只读检查。

特别检查两个路径层面：

1. `--libero-root` 决定导入哪个 LIBERO Python 包。
2. `get_libero_path()` 返回 BDDL、init states、assets 的实际路径；它们来自 LIBERO 配置，不能仅靠 `--libero-root` 保证一致。

本机 Plus 代码读取 `$LIBERO_CONFIG_PATH/config_plus.yaml`，变量默认对应 `~/.libero`；上游不同版本可能使用 `config.yaml`。以实际导入代码为准。需要调整时使用单独的 `LIBERO_CONFIG_PATH`，避免改动其他任务的配置。不存在配置文件时首次 import 可能触发交互初始化。

配置检查示例（使用已经正确配置的环境）：

```bash
PYTHONPATH="$LIBERO_PLUS_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  /path/to/python - <<'PY'
import libero
from libero.libero import get_libero_path
print('package:', libero.__file__)
for key in ('bddl_files', 'init_states', 'assets'):
    print(key, get_libero_path(key))
PY
```

## 5. 转换脚本使用

所有示例在仓库根目录执行。`PY` 指向已有仿真 Python；在新环境中可换成项目 `.venv/bin/python`。这里只运行程序，不修改共享环境。

```bash
cd /VLA-Data/scripts/lianqing/projects/vla/libero_data_process
export PY=/VLA-Data/scripts/lianqing/projects/vla/X-VLA/.venv/bin/python
export LIBERO_RAW=/VLA-Data/scripts/lianqing/data/yifengzhu-hf/LIBERO-datasets
export LIBERO_PLUS_ROOT=/VLA-Data/scripts/lianqing/projects/vla/X-VLA/libero_plus
"$PY" scripts/prepare_libero_data.py generate --help
```

### 5.1 普通 LIBERO

```bash
"$PY" scripts/prepare_libero_data.py generate \
  --input "$LIBERO_RAW/libero_spatial" --suite libero_spatial \
  --libero-root "$LIBERO_PLUS_ROOT" --modes base \
  --output-root /VLA-Data/scripts/lianqing/data/libero_prepared_base
```

`base` 使用当前导入的仿真环境；若要严格复现旧普通 LIBERO 运行，应传当时的普通 LIBERO checkout、配置及环境版本，并通过回放检查。

### 5.2 Camera 与 action

```bash
"$PY" scripts/prepare_libero_data.py generate \
  --input "$LIBERO_RAW" --libero-root "$LIBERO_PLUS_ROOT" \
  --modes camera camera_plusplus action \
  --variants-per-demo 5 --action-level 2 \
  --output-root /VLA-Data/scripts/lianqing/data/libero_prepared_camera_action
```

### 5.3 其他 Plus 类别

```bash
"$PY" scripts/prepare_libero_data.py generate \
  --input "$LIBERO_RAW" --libero-root "$LIBERO_PLUS_ROOT" \
  --modes language light background add noise \
  --variants-per-demo 2 \
  --output-root /VLA-Data/scripts/lianqing/data/libero_prepared_plus
```

三组示例使用不同输出目录。需要合在一次运行时可以同时传多个 `--modes`，例如 `base camera camera_plusplus action language light background add noise`。

### 5.4 最小回放检查

```bash
"$PY" scripts/prepare_libero_data.py generate \
  --input '<一个真实的task_demo.hdf5绝对路径>' --suite libero_spatial \
  --libero-root "$LIBERO_PLUS_ROOT" \
  --modes base camera_plusplus action \
  --max-demos 1 --variants-per-demo 1 \
  --output-root /VLA-Data/scripts/lianqing/data/libero_prepared_smoke
```

`--max-demos 1` 是**每个输入任务文件**最多处理一条 demo。要只测一个任务，应传单个 HDF5。传数据根目录时会递归处理其中所有 `*_demo.hdf5`，包括存在的 `libero_90`；`--suite` 只覆盖 suite 推断，不是过滤条件。只做四个 suite 时应逐个 suite 运行，或使用仅包含这四个 suite 的输入根目录。

本机诊断 smoke 控制在五分钟内。全量造数有大量仿真和渲染开销，应先确认资源与执行环境；本文没有启动这些任务。

## 6. 参数、模式与实现语义

| 参数 | 默认/含义 |
|---|---|
| `generate --input` | 必填；原始单文件、suite 目录或数据根目录 |
| `--libero-root` | 指定 LIBERO 包所在 checkout；省略时使用环境已有包 |
| `--suite` | 覆盖 suite 推断；多 suite 数据不要强制同一个值 |
| `--modes` | 默认 `base`；支持多个模式 |
| `--variants-per-demo` | 默认 5；base 始终只生成一个版本 |
| `--action-level` | 默认 2；可选 1、2、3 |
| `--resolution` | 默认 224；两路图像使用相同分辨率 |
| `--seed` | 默认 42 |
| `--max-demos` | 默认处理所有 demo；限制每个任务文件的 demo 数 |
| `--output-root` | 默认 `/VLA-Data/scripts/lianqing/data/libero_prepared`；必须为空或不存在 |

### 6.1 共同回放流程

1. 读取 `data/demo_i/actions`、`states[0]`、任务 BDDL 与指令。
2. reset，恢复初始状态；add 有匹配 `.pruned_init` 时可使用其状态。
3. 稳定 10 步：base 使用 `[0,0,0,0,0,0,-1]`；Plus 使用全零动作。
4. action 模式插入一次偏移动作，然后记录初始帧。
5. 过滤 no-op 并回放原始 action，记录每次执行后的图像与控制器目标。
6. base 完整回放并用最终 `done` 判断；Plus 遇 `done` 停止并调用 `check_success()`。
7. 成功且有有效动作才写 HDF5。回放失败记录为 `replay_failed`；异常记录为 `error`。

no-op 条件：前六维范数小于 `1e-4`，且夹爪与前一保留动作相同。首个动作没有前序夹爪时只检查前六维。

### 6.2 Camera

- `camera`：外部相机从 16 个水平角 × 2 个垂直角 × 5 个 scale 共 160 个配置中采样，通过 `_view_*` BDDL 命名约定传入环境。
- `camera_plusplus`：同样的外部相机采样，加手腕 `none/hflip/vflip/rotate180`，以及 60°～90° FOV。
- `camera_plusplus_star`：保留上述离散外部相机采样，手腕 FOV 扩大至 50°～100°。
- 所有模式都把 agentview 图像旋转 180°；手腕仅执行明确指定的翻转。

### 6.3 Action

每条变体采样固定位置、旋转偏移，在稳定之后执行**一次 delta action**。随后继续回放原 demo；标签记录新的控制器绝对目标。

| 模式 | Level 1 位置/角度 | Level 2 位置/角度 | Level 3 位置/角度 |
|---|---|---|---|
| action | ±0.005 / ±1° | ±0.010 / ±3° | ±0.020 / ±6° |
| action_star | ±0.015 / ±5° | ±0.030 / ±10° | ±0.050 / ±15° |

位置各轴独立采样，旋转为随机轴和有界角度。位置数值直接作为控制器 delta 输入，实际物理位移取决于 action scaling。现有评测的逐步绝对动作偏置与这条造数流程有区别，实验说明需分别描述。

### 6.4 其他模式和与旧脚本的关系

- `language` 读取变体 BDDL 的语言，保证保存改写后的指令。
- `background` 对应 `table/tb` 变体；`add` 可使用对应初始状态；`noise` 从 5/10/15 中采样。
- 当前独立脚本按类别分别生成，没有移植旧 `combined` 模式的 camera/noise 随机叠加，也没有实现 `robot` 初始状态套件生成。
- 当前使用按 suite、任务、demo、mode、variant、seed 派生的稳定随机种子。扰动类型和范围沿用旧实现，但相同总 seed 不保证逐条复现旧脚本抽样。
- 当前顺序处理，不提供并行 worker、视频导出或生成续跑；空目录要求用来保护已有输出。HF 下载续传与仿真生成续跑是两件事。

## 7. 输出格式与训练接入

```text
<output-root>/
├── base/<suite>/<task>_demo_0_v0.hdf5
├── camera/<suite>/...
├── camera_plusplus/<suite>/...
├── action/<suite>/...
├── meta.json
└── generation_report.json
```

执行 N 个有效动作时：

| 字段 | 形状/含义 |
|---|---|
| `observation/agentview_rgb` | `[N+1,224,224,3]`，默认分辨率 |
| `observation/eye_in_hand_rgb` | `[N+1,224,224,3]` |
| `abs_action_6d` | `[N+1,7]`，xyz + axis-angle + gripper |
| `rewards` / `dones` | `[N]`，最后一项为 1 |
| attrs `instruction` / `task_name` | 指令、任务名 |
| attrs `perturbation_type` | JSON；扰动参数、来源文件、demo、seed 等 |

`abs_action_6d` 名称沿用现有数据，磁盘中实际旋转是三维轴角。位置与姿态来自 `controller.goal_pos/goal_ori`，不是测量得到的末端状态。

`meta.json` 自动生成，核心内容如下：

```json
{
  "observation_key": ["observation/agentview_rgb", "observation/eye_in_hand_rgb"],
  "dataset_name": "libero",
  "language_instruction_key": "instruction",
  "datalist": ["/absolute/path/to/episode.hdf5"]
}
```

可将绝对路径传给训练入口的 `--train_metas_path`。训练启动配置和计算资源单独确认；本文只说明数据接口。

下列文件属于原 `x_vla_ic` 训练仓库，用于解释下游接口；本仓库不包含也不导入这些模块：

- 原训练仓库的 `datasets/domain_handler/simulations.py`：`LiberoHandler` 转换轴角到 rot6d、夹爪二值化，单臂 10D 加另一臂零值组成 20D。
- 原训练仓库的 `x_vla_fast/datasets/domain_handler/base.py`：插值、采样 chunk，组装 IC 多帧。
- 原训练仓库的 `x_vla_fast/datasets/utils.py`：第一轨迹点作 proprio，后续 H 个点作 action。

现有 `LiberoHandler` 会对图像执行 `img[1:]`，动作数组完整保留。本文脚本保存原有 N+1 结构，没有调整训练读取器的时间对齐；新一轮实验应单独验证其合理性。

只重新生成清单：

```bash
"$PY" scripts/prepare_libero_data.py meta \
  --root /VLA-Data/scripts/lianqing/data/libero_prepared_camera_action \
  --output /VLA-Data/scripts/lianqing/data/libero_prepared_camera_action/meta.json
```

`meta` 递归扫描 `.hdf5`，校验动作形状与有限值、图像维度、reward/done 长度和非空指令。应只对已生成数据目录运行；原始 LIBERO HDF5 的 schema 不符合这一要求。存在坏文件会报错；该检查不重新执行仿真，也不验收物理时间对齐。

## 8. 常见问题与当前验证范围

| 现象 | 检查项 |
|---|---|
| 导入失败、缺少 wand/MuJoCo 依赖 | 检查 Python 环境和 Plus 官方依赖；不要修改共享训练环境 |
| 第一次运行等待输入 | 检查 LIBERO 配置文件是否存在，先完成配置初始化 |
| BDDL/mesh 路径不存在 | 检查实际导入包、`get_libero_path()`、Plus 资产目录 |
| `No ... BDDL variants` | 对应 Plus checkout 是否包含该任务变体；普通 LIBERO 包不提供全部 Plus 资源 |
| 输出目录不为空 | 使用新的输出目录；脚本没有生成续跑模式 |
| 文件数小于 demo × variants | 查看 `replay_failed` 与 `error`；base 始终只生成 1 个版本 |
| 部分成功但程序非零退出 | 成功文件和清单被保留；异常数量非零时最终会报错，检查报告 |
| 旧 meta 指向不存在的路径 | 核对实际磁盘文件，重新生成 meta；清单条数不代表现存数据量 |

此前已完成：Python 语法/CLI 检查、模拟环境对象上的回放顺序/no-op/翻转/结束条件检查、HDF5 与清单校验，以及一条现有普通 LIBERO HDF5 结构检查。尚未对独立脚本执行真实 MuJoCo 端到端回放，也没有在本次文档工作中进行全量下载或数据生成。正式使用前先执行第 5.4 节的小规模回放并检查结果。
