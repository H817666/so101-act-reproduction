# LeRobot SO-ARM101 上的 ACT 抓取复现

用一套 SO-ARM101 Pro 主从机械臂、单个第三视角相机，在固定桌面场景完成“抓起小物块并放入金属盒”的示教采集、ACT 训练和真机闭环推理。这里复现的是 **[ACT](https://arxiv.org/abs/2304.13705) 的策略训练与执行流程**，使用 [LeRobot](https://github.com/huggingface/lerobot) 实现；不是论文中 ALOHA 双臂硬件、任务和结果的逐项复现。

## 已完成的结果与边界

- 第二轮固定场景数据：60 段示教、21,378 帧、20 Hz，640×480 的第三视角图像；输入为图像与白臂六维状态，输出为六维关节/夹爪目标。
- LeRobot ACT 从头训练 100,000 步，batch size 24；最后一次验证损失约 0.2236。损失数值不能直接当作抓取成功率。
- 用户在训练覆盖的三个物块位置分别完成了真机抓取测试，均报告成功。尚无足够的重复试验来给出可信的统计成功率，也尚未证明对未见位置的泛化。
- 第二轮模型在物块偏离训练位置约 3 cm 的一次测试中未能抓取；第一轮则因场景没有稳定固定，在原训练位置也未能成功抓取。
- GPU 推理加 ACT 时间集成后，用户报告动作比最初 CPU/低频重规划更快、更平顺。离线 60 帧测试（预热后）推理中位数约 5.9 ms、P95 约 7.0 ms；这是本机测量，不保证在其他机器上成立。

原论文的 ACT 预测一段未来动作；本项目使用 LeRobot 的 ACT 实现。`--temporal-ensemble` 让每个 20 Hz 控制周期重算动作并融合重叠预测（系数 0.01）。不加该选项时脚本采用约 4 Hz 的异步动作块重规划。无论哪种模式，真机控制代码还保留关节边界、速度、跟踪误差、相机时效与起始姿态检查。

[观看第二轮 GPU 推理的抓取视频（16 秒、已静音）](media/act_v2_gpu_success.mp4)。这是一次演示，不作为统计成功率证据；拍摄画面与模型实际输入的第三视角画面不同。

## 硬件与环境

| 部件 | 本次配置 |
| --- | --- |
| 主臂 | 黑色 SO-ARM101 Pro Leader，7.4 V 舵机；套件标配 5 V / 6 A 适配器 |
| 从臂 | 白色 SO-ARM101 Pro Follower，12 V 舵机；12 V / 5 A 适配器，TPU 夹爪 |
| 相机 | 第三视角 USB 相机，录制为 640×480 / 20 Hz；双相机套件中的腕部相机未用于此训练 |
| 软件 | Linux、LeRobot 0.6.1、PyTorch 2.11.0+cu128、OpenCV 4.13.0、NumPy 2.2.6 |

电源和串口要按**自己设备的标签**再次核对，不能因颜色或本表直接套用。先完成 LeRobot 官方的 [SO-101 装配与校准教程](https://huggingface.co/docs/lerobot/so101)，并按官方 [真实机器人模仿学习教程](https://huggingface.co/docs/lerobot/il_robots) 配好环境。仓库不提供个人校准文件；所有真机步骤必须由现场人员看护。

本仓库脚本依赖 Linux V4L2、Feetech 总线、交互终端和图形桌面。使用本机已验证的 LeRobot 环境运行，例如先激活你的虚拟环境：

```bash
source /path/to/your/lerobot-venv/bin/activate
python -c 'import lerobot, torch; print(lerobot.__version__, torch.__version__)'
```

复制 `.env.example` 为 `.env`，填入本机设备路径后执行 `set -a; source .env; set +a`。`.env` 已被 Git 忽略。可用 `ls -l /dev/serial/by-id/` 和 `ls -l /dev/v4l/by-id/` 确认路径。必须先明确区分主臂和从臂；不要边运动边交换电源或数据线。

## 复现流程

所有命令在仓库根目录执行。先固定白臂底座、相机、盒子和光照，确定物块的训练位置；主臂不应进入训练画面。场景变动后应重新采集数据，不要把不同场景直接混训。

### 1. 只读预检与示教采集

```bash
python scripts/so101_follow_test.py
python scripts/so101_record_v2.py --prepare \
  --preview-image /path/to/current_camera_preview.jpg
python scripts/so101_record_v2.py
```

第一条只读检查两个串口、校准、舵机模式与力矩状态。场景参考图只保存在本地。采集窗口先按回车让白臂原位保持，再按 `R` 跟随、`C` 开始一段、`S` 保存、`X` 丢弃、`Q` 退出。`S` 后必须等待终端提示“已保存”，再开始下一段。单段最长 45 秒，录制状态需显示“正在录制”。未通过采样检查的片段保留在 `raw/` 供排查，但不计入可训练数据。

采集时每段应完整覆盖“起始静止 → 接近 → 张爪/闭爪抓取 → 抬起 → 放置 → 结束”，避免人为停顿、乱晃和遮挡。物块位置要覆盖任务所需范围；仅在三个精确位置成功，不等于能处理任意偏移。相机与控制帧是软件时间同步，不是硬件曝光同步。

### 2. 审计并合并 60 段数据

```bash
python scripts/so101_audit_v2.py
python scripts/so101_prepare_v2.py
```

这两个脚本是**本次 60 段数据**的整理流程：审计帧数、时间戳、视频、动作值及采集质量，再把多个采集目录合并为 `~/so101-datasets-v2/act_pick_place_v2_60`，核对动作/状态和视频哈希。`so101_prepare_v2.py` 会拒绝非 60 段或未通过审计的数据。若你采集的新数据不是 60 段，需先重新设计划分和检查条件，不能简单改数字假装等价。原始数据及合并数据都不在 Git 仓库中。

### 3. 训练前检查与训练

```bash
python scripts/so101_train_act.py --v2 --check
python scripts/so101_train_act.py --v2 --smoke --batch-size 24
python scripts/so101_train_act.py --v2 --batch-size 24
```

此配置固定 6 段验证集（编号 8、18、28、38、48、58，零起始），其余 54 段训练；训练 100,000 步，约每 5,000 步做离线验证，每 25,000 步存检查点。脚本要求 CUDA，默认在 `~/so101-training/runs/` 输出模型和日志，且不会访问机械臂。验证划分按 episode 而非随机帧，但若同一物块位置同时出现于训练与验证集，验证损失不能代表跨位置泛化。

### 4. 离线检查、姿态对齐、真机推理

把下面路径替换成你自己的训练输出目录：

```bash
CKPT=/path/to/so101-training/runs/act_xxx_v2/checkpoints/last/pretrained_model
python scripts/so101_infer.py --v2 --check --checkpoint "$CKPT" --device cuda --temporal-ensemble
python scripts/so101_pose_check.py --v2
python scripts/so101_infer.py --v2 --preview --checkpoint "$CKPT" --device cuda --temporal-ensemble
python scripts/so101_infer.py --v2 --run --checkpoint "$CKPT" --device cuda --temporal-ensemble --seconds 30
```

`--check` 完全离线；`so101_pose_check.py` 只读白臂姿态；`--preview` 只读相机与关节。退出预览后再运行 `--run`，此时仍须按 `R` 并经过倒计时才启动力矩。`P` 原位保持，`Q`/空格停止并卸力。卸力后白臂可能下落，务必有软垫和现场人员承托。只有需要记录某一次试跑时才加 `--record-video`。

真机前确认：校准与采集时一致、相机画面与训练场景一致、物块/盒子位于训练覆盖区域、白臂起点接近同一条示范起点、工作区无障碍物、旁边可立即切断外部电源。软件保护不是物理急停。

## 项目文件

- `scripts/so101_follow_test.py`：主臂相对跟随与安全预检；`--profile full` 为采集档位。
- `scripts/so101_record.py`、`so101_record_ui.py`、`so101_record_v2.py`：20 Hz 采集、保存状态反馈、场景包装。
- `scripts/so101_audit_v2.py`、`so101_prepare_v2.py`：第二轮数据审计与合并。
- `scripts/so101_train_act.py`：训练与 episode 划分。
- `scripts/so101_pose_check.py`、`so101_infer.py`：起点检查、离线/在线推理。
- `docs/experiment.md`：实验过程、局限与下一步。
- `tests/test_offline.py`：不连接设备的采集质量与推理限幅单元测试；运行 `python -m unittest discover -s tests -v`。

## 公开范围与隐私

仓库包含脚本、说明和上方这一段经确认的演示视频，**不包含**真实数据集、其他原始画面、模型权重、校准文件、诊断日志和设备序列号。演示视频仍包含实验场地背景，发布前应确认背景适合公开。若将来公开数据或权重，建议单独托管并补上版本、校验和、授权说明；当前 README 中的本地性能数字不是公开可复算的基准。

## 参考资料

- [Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT)](https://arxiv.org/abs/2304.13705)
- [ACT 原作者代码](https://github.com/tonyzhaozh/act)
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
- [LeRobot ACT 文档](https://huggingface.co/docs/lerobot/act)
- [LeRobot SO-101 文档](https://huggingface.co/docs/lerobot/so101)
- [LeRobot 真实机器人模仿学习文档](https://huggingface.co/docs/lerobot/il_robots)

本项目代码是针对个人设备与实验场景的工程实现，具体兼容性以你的 LeRobot 版本、硬件和校准为准。
