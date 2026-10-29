# 数采、训练和部署的公共契约

以当前导出契约为准：数据集 `meta/info.json` 的 `wholebody_export`，以及训练入口的 `validate_dataset_contract`。不按某个 wholebody git 分支名判断数据是否可用。

## 各项目负责什么

- wholebody：保存原始 state/command、连续段和采样时间；导出标准列；按服务声明准备观测；决定执行预算 K、重规划和执行器绑定。
- g2-policy-protocol：wire v2/v3、JSON/MessagePack、通用 WebSocket 边界；g2_policy_tools 提供共同的 state/action schema、采样、几何编码、磁盘契约和能力推导。无框架或机器人运行时依赖。
- lerobot-finetune：每种 policy 的原生训练配置、LeRobot 加载/processor/训练入口；G2 adapter 接入公共工具；部署适配原生 chunk 输出。
- OpenPI：原生模型、优化器、训练与完整恢复；G2 adapter 接入同一公共工具。服务端只做观测键映射、模型调用和动作还原。

训练 YAML 的 `experiment / data / policy` 语义共用；模型超参、优化器、checkpoint 仍遵循各自框架，不强行使用同一种完整 YAML。

## 连续段与监督

新 v2.1 和 v3 导出遵循 `wholebody_export.segment_policy: separate_episodes`。一个 raw episode 有多个 segment 时，一个连续 segment 对应一个训练 episode。元数据保留 `source_episode`、`source_episode_id`、`source_segment_id` 和 `source_segment_count`。每段仍使用原有原子提交；中断后按 raw 路径和段 ID 去重，所有段完成后 watcher 才把原始记录视为完成。

新导出固定 `action_supervision: command`，不把 state 重命名成 action。测量监督由 `policy.action.pose.source: state` 或 `policy.action.joint.source: state` 选择。旧 `action_supervision: state` 导出配置会在入口被拒绝。

非空数据集的导出契约必须一致。旧数据集缺少 segment_policy 时，新数据应导出到新目录；不要修改 info.json 假装已迁移，也不要把两种 episode 边界写进同一数据集。历史原始记录和既有数据不自动重写。

两套训练入口调用 `validate_dataset_contract`：检查所需列、声明的向量维度、有效 fps、已声明的 base_link/flange/rot6d、command 监督一致性，以及显式标注的 concatenated 段策略。保留对历史数据缺省元信息的兼容；缺省不能证明坐标或段边界正确，需要先核对来源。此工具不猜测或补写历史元信息。

## 预测块与执行预算

H 为模型预测长度，R 为响应帧数，K 为客户端提交给执行队列的步数。两套第一方 v3 服务声明 `extensions.response_mode: full_chunk`，固定 R=H；公共响应入口拒绝截短输出。RTC 的历史前缀允许短于 H，不受完整响应约束。

K 只由 wholebody 的 `infer.execution_steps` 决定，满足 K≤R。服务端不截取 execution_steps，也不把每次推理调用当成一拍。旧 v2 服务保留既有行为；第三方 v3 服务可独立声明自己的响应模式。

LeRobot G2 部署配置删除 `inference.execution_steps`；`temporal_ensemble_coeff` 非空会报错，原因是单步输出尚无目标时间对齐。模型原生动作索引必须连续并从当前观测开始；其他时间布局需先提供专属输出时间适配，离线预测和训练的原生能力不因此增加。

状态历史、图像历史、episode 起点和 command/object 参考由 wholebody 的采样时间提供。OpenPI 不再根据不均匀的推理调用缓存历史或猜测 episode 起点。两种服务共用 policy_capabilities 推导需求；缺少必要 context 会在请求边界报错。

pose_target / joint_position 可由现有 wholebody 执行器处理。关节速度、力矩仅有协议/训练语义，应用没有绑定执行器时启动前拒绝。LeRobot 尚不支持 RTC；OpenPI 保留自身 RTC 实现。

## 环境和验收

两套训练框架保持独立虚拟环境。公共包版本统一为 0.6.0，wire 仍为 v3/v2，未改变协议主版本。

开发使用相邻源码安装；跨机部署应构建一次 wheel，复制同一个文件，记录其 SHA256，并记录四个仓库的提交和未提交变更。相同版本号不能证明两份 editable 源码一致。测试输出中的 stack_manifest 包含提交、分支和源码变更摘要；有未提交变更时还需保存对应 patch 和新文件。

```bash
# 在四个仓库的共同父目录，客户端使用 LeRobot 环境；worker 自动选各自环境。
lerobot-finetune/.venv/bin/python g2-policy-protocol/tools/check_stack.py --stack-root "$PWD"
```

脚本用真实 wholebody 客户端、共享服务边界、LeRobot G2Backend 和 OpenPI G2Policy/OpenPIBackend，覆盖两种编码、pose/joint、L=1/2、显式参考、H=R=4/K=2 和 reset。只替换模型数值输出，不加载权重、不访问 GPU/HAL，不等同于真机轨迹验收。

wholebody 的 test_export_segments 覆盖真实 JPEG/ffmpeg/v2.1/v3 写盘、暂停边界采样、第二段失败重试；共享 test_pipeline_contract 覆盖数据契约和短响应拒绝。训练恢复由 OpenPI scripts/train_test.py 的真实小模型测试验证；运行时设 JAX_PLATFORMS=cpu，以免被服务器 GPU 数影响测试批量。
