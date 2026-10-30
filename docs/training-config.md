# G2 公共训练配置

OpenPI 和 LeRobot 的 G2 入口统一使用 `schema_version: 7`。公共字段只在启动或加载 checkpoint 时解析；训练循环、采样器和推理循环不做配置转换。未知公共字段和后端选项中重复声明的公共字段会直接报错。

```yaml
schema_version: 7
backend: lerobot                # openpi | lerobot，须与启动入口一致
experiment:
  name: TeaPi
  arm: right
data:
  root: ./data
  repo_id: vr/TeaPi/lerobot_v21
  cameras: null
policy:
  # state/action 规则见 policy-config.md
  timing:
    action_horizon: 50
model:
  type: pi05                   # LeRobot 也可用 act
  pretrained_path: lerobot/pi05_base
training:
  batch_size: 8
  steps: 50000
  num_workers: 4
  seed: 42
  log_interval: 50
  save_interval: 5000
  learning_rate: 2.5e-5
runtime:
  gpus: [0]
  resume: false
wandb:
  mode: 'off'                  # auto | on | off；on/off 加引号避免 YAML 1.1 布尔解析
  project: G2
backend_options:
  lerobot:
    policy:
      device: cuda
      push_to_hub: false
      n_action_steps: 50
      use_relative_actions: false
```

完整数值语义与各后端默认值以仓库示例为准。OpenPI 自定义文件覆盖完整默认 YAML；LeRobot 文件独立配置，省略的原生参数沿用 LeRobot 默认值。

| 公共字段 | OpenPI 映射 | LeRobot 映射 |
|---|---|---|
| `model.type` | 当前 G2 入口固定 pi05 | 原生 policy.type |
| `model.pretrained_path` | 参数加载路径 | 原生 policy.pretrained_path |
| `training.steps` | num_train_steps | steps |
| `training.learning_rate` | 余弦调度器 peak_lr | policy.optimizer_lr |
| `training.log_interval` | log_interval | log_freq |
| `training.save_interval` | save_interval | save_freq |
| `policy.timing.action_horizon` | model.action_horizon | policy.chunk_size，并核对实际 action_delta_indices |
| `runtime.resume` | 布尔值，恢复当前实验最新完整 checkpoint | false / true / checkpoint 路径 |
| `wandb.mode` | auto 在 smoke 时关闭 | auto/on 开启，off 关闭 |

`batch_size` 保留框架语义：OpenPI 是全局 batch，LeRobot 是每进程 batch。相同 batch 字段和学习率并不保证跨后端训练等价；多卡对比需核对有效全局 batch、优化器与学习率调度。

OpenPI 专属选项放在 `backend_options.openpi`：`finetuning`、`run.smoke`、`model.discrete_state_input`、`training` 中的调度器/FSDP/保留周期、`checkpoints` 的目录与覆盖设置、`normalization`、`runtime.mem_fraction`。LeRobot 专属选项放在 `backend_options.lerobot`，沿用原生 policy、PEFT、评估和数据集选项。公共字段只能在顶层设置；不要在后端 policy 中重复写 type、pretrained_path、optimizer_lr、chunk_size。

两边均接受 `--steps`、`--batch-size`、`--num-workers`、`--log-interval`、`--save-interval`、`--learning-rate`。后端现有 CLI 继续可用，显式 CLI 优先于 YAML。

checkpoint 保存自包含的 v7 配置。运行时没有旧训练配置兼容分支；已有 checkpoint 的配置元数据须直接迁移，权重、优化器状态和归一化数值保持不变。标准 LeRobot 数据的 `adapter: null` 配置继续沿用 LeRobot 原生结构。
