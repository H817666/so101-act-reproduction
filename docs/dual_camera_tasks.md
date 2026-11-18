# 双相机 ACT 任务记录：盖帽与试管入槽

这两项任务使用 SO-ARM101 Pro 主从臂、固定桌面场景、第三视角和腕部相机。策略输入为两路 640×480 RGB 画面与从臂六关节状态，输出六关节目标。采集以 20 Hz 记录；两路相机按主机接收时间做软件配对，**不是硬件同步**。本仓库不提供数据集、模型权重或个人校准文件。下面是本次实验的配置记录，不是跨场景即插即用的结果保证。

## 盖帽

任务是抓起帽子并放到试管口，**不包括压紧或密封**。整理后数据集为 `act_cap_tube_dual_50`，50 段示教，45 段训练、5 段留出验证；整理与检查脚本为 [`so101_cap_prepare.py`](../scripts/so101_cap_prepare.py)。ACT 训练脚本为 [`so101_train_act_cap.py`](../scripts/so101_train_act_cap.py)，本次使用 100,000 步训练。公开视频是一次完成“盖上”目标的外部视角演示，不证明统计成功率。

## 试管抓取与入槽

固定场景里设置三个起点与各自对应的槽位，每次只抓取、放置一支试管。每个起点录制 20 段，共 60 段；合并数据集 `act_tube_slot_dual_60`，整理脚本为 [`so101_prepare_tube.py`](../scripts/so101_prepare_tube.py)。最初用 60 段合训一个 ACT 模型，真机表现不均衡：中间位置相对可靠，两端位置出现抓偏或放错槽。之后用 [`so101_train_act_tube_positions.py`](../scripts/so101_train_act_tube_positions.py) 为三个位置分别训练模型，每个模型 18 段训练、2 段留出验证、25,000 步。本次位置 1/2 在云端 batch 32 训练，位置 3 在本地 batch 16 训练；因此这三个模型并非严格同配置对比。

用户随后在三个**训练位置**分别报告过成功。该反馈不是留出位置测试，也没有完整试次数统计。公开视频为试管入槽的单次演示，不代表三个位置连续运行，更不能推出新起点/新孔位的泛化成功率。

## 复现脚本与使用边界

本机安装与仓库 README 一致的 LeRobot 环境，复制 [`.env.example`](../.env.example) 中的变量并按实际设备配置主臂、从臂和两路相机。示教采集入口为 [`so101_record.py`](../scripts/so101_record.py)，任务文本、数据目录和相机参数均由命令行提供；同一数据集的任务文本和相机映射应保持一致。示例命令（数据目录必须尚不存在）：

```bash
python scripts/so101_record.py --run --profile full \
  --camera "$SO101_CAMERA" --wrist-camera "$SO101_WRIST_CAMERA" \
  --task "Pick up the cap and place it onto the test tube." \
  --dataset-dir "$HOME/so101-datasets/my_cap_recording" --episode-seconds 60
```

训练前先运行只读检查；这些命令要求相应的**私有数据集**和整理报告已经存在：

```bash
python scripts/so101_train_act_cap.py --check
python scripts/so101_train_act_tube.py --check
python scripts/so101_train_act_tube_positions.py --position 1 --check
```

训练命令示例（仅启动所选任务，不要把合训与分位置模型当作同一模型）：

```bash
python scripts/so101_train_act_cap.py --batch-size 16 --steps 100000
python scripts/so101_train_act_tube_positions.py --position 1 --run --batch-size 16 --steps 25000
```

云端历史封装脚本位于 [`scripts/cloud/`](../scripts/cloud/)，依赖当时的 `/root/gpufree-data/so101` 布局，迁移到新平台前需要核对路径、数据与依赖版本。

[`so101_infer.py`](../scripts/so101_infer.py) 负责离线检查、只读预览和需人工确认的真机运行。双相机任务需指定匹配的 `--cap` 或 `--tube`、`--checkpoint`、对应 `--dataset-root` 与采集配置 `--collection-config`；分位置模型还必须指定 `--tube-position 1/2/3`。先 `--check`，再 `--preview` 检查相机和起点，现场清空机械臂工作区并有人看护时才使用 `--run`。脚本保持校准边界、启动位姿及运动限幅保护，不能把离线损失或演示视频当作真机安全证明。

后续 SmolVLA 研究可以沿用这些采集/评估经验，但现有视频与结果均来自 ACT。加入文字任务指令并不会自动获得新物体、新位置或新场景的泛化能力；应为 SmolVLA 单独设计留出场景和可重复的真机评测。
