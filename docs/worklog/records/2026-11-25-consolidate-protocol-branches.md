# 2026-11-25 协议分支整合

| 项 | 值 |
|---|---|
| 作者 | Codex |
| 机器 | `g2-dc` |
| 分支 @ 提交 | 整合分支 `integration/protocol-main`，基于 `feat/g2-recording-schema` @ `c57d8c2` 和 `refactor/shared-g2-adapter` @ `7bd7960` |
| 环境 | Linux aarch64，域控现有 Python 3.10，测试从整合分支源码导入协议包 |
| 类型 | 源码核对、离线测试和本机回环 |
| 结果 | 协议仓 388 项通过。wholebody 快测 152 项通过，168 个子用例通过。定向回归 65 项通过，10 个子用例通过。完整跨仓检查在 LeRobot worker 启动失败，OpenPI worker 和真实 GPU 服务未运行。 |

保留两个功能分支的原始提交及其祖先关系。整合版沿用 0.7.0 模块布局，包含 RTC 时间语义说明和录制列契约。跨仓脚本兼容 wholebody 的 `execution_plan` 和 `resolve_execution_steps` 接口。

LeRobot worker 的类型别名使用 Python 3.12 语法，域控的 Python 3.10 无法加载。完整跨仓检查需要通过 `--lerobot-python` 与 `--openpi-python` 指定各自的适配器环境。
