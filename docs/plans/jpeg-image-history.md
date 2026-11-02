# 实施计划：图像历史支持 JPEG 传输

状态：代码与离线验证已完成；待实机验证 · 创建：2026-11-02 · 涉及仓库：`g2-policy-protocol`、`wholebody`、`openpi`、`lerobot-finetune`

后续优化已实现：有完整 `context.image_history` 时，wholebody 跳过当前帧准备，线上省略重复的 `images`。历史最后一帧就是当前帧，模型输入不变。无历史请求保留 `images`；新服务端继续接受旧客户端发送两份图像。需先更新 GPU 侧协议包及支持历史的 backend，再更新客户端；旧协议实现会拒绝缺少 `images` 的历史请求。以下 §2 保留初始实施前的代码状况，§8 第 1 项已完成，最新契约以 [通信协议](../protocol.md) 为准。

## 1. 背景

### 1.1 系统概览

```text
域控（G2 真机）                                  GPU 服务器
  wholebody（client）                              openpi / lerobot-finetune（server）
    ObservationStream 按 history_hz 采样             PolicyServer + 各自的 backend
    G2WebSocketPolicyClient ── obs ──────────▶       解码图像 → 模型推理 → 还原动作
                            ◀── action ─────
                 WebSocket · g2.policy.v3 · MessagePack（默认）或 JSON
```

四个仓库位于同一父目录（如 `/home/zijianwang/`），协议包以 editable 方式装进各环境，两端 `git pull` 同步。协议细节见 [通信协议](../protocol.md)，安装与兼容规则见 [安装](../install.md#兼容性如何保证)。

### 1.2 什么是图像历史

训练配置 `policy.timing.history_steps > 1` 时（典型场景：UMI 风格，state 与 action 都相对当前末端编码），模型需要过去若干帧观测。握手 metadata 的 `observation_spec.context_keys` 会包含 `image_history`，客户端在每个 `obs` 的 `context.image_history` 里发送每个必需相机的最近 `history_steps` 帧（从旧到新，间隔 `1 / history_hz`，**最后一帧就是当前帧**）。

`policy.timing.history_steps` 同时控制位姿历史与图像历史，本计划不改变这一点。

### 1.3 现状与问题

| 项 | 当前帧 `images` | 图像历史 `context.image_history` |
|---|---|---|
| 编码 | 随握手 `image_encoding`：`jpeg`（两个第一方 server 默认）或 `raw` | **只能 raw**：每相机一个 `uint8` 数组 `(L, H, W, 3)` |
| 校验位置 | `protocol.validate_images` | `protocol._parse_image_history` |
| 服务端解码 | `images.decode_images` | 无需解码 |

问题：

1. **体积大**。224×224×3 原始帧约 147 KB，JPEG（quality 90）通常 15–30 KB。单臂两相机、`history_steps: 2` 时，图像历史每次请求约 600 KB，而当前帧 JPEG 合计只有几十 KB。JSON 编码下原始数组是数字列表，更大。
2. **当前帧发两遍**：`images` 里一份 JPEG，`image_history` 最后一帧又是一份 raw。两个 server 在有图像历史时只用历史、不解码当前帧（见 §2.3、§2.4）。

## 2. 实施前实现（改动点）

以下路径均相对各仓库根目录。函数名比行号稳定，以函数为准。

### 2.1 协议包 `g2-policy-protocol`

- `src/g2_policy_protocol/protocol.py`
  - `parse_context` → 对 `image_history` 调用 `_parse_image_history(value, steps, metadata)`。
  - `_parse_image_history`：要求键恰为 `required_cameras`；每个值 `np.asarray` 后必须是 `uint8`、`ndim == 4`、长度 `== history_steps`、通道 3，且 `(H, W)` 等于 `image_sizes[camera]`（若声明）。错误信息 `"image_history must contain raw uint8 (L,H,W,3) frames"`、`"image_history size disagrees with metadata"`。
  - `validate_metadata`：`history_steps > 1` 且有必需相机时，强制 `context_keys` 含 `image_history`（不改）。
- `src/g2_policy_protocol/images.py`：`decode_images(images, metadata)` 按 `metadata["image_encoding"]` 解码当前帧并调用 `protocol.validate_images(..., decoded=True)`；损坏 JPEG 抛 `ValueError`（server 回 `invalid_request`）。`encode_jpeg(image)` 用 Pillow，quality 90。
- `src/g2_policy_protocol/codec.py`：MessagePack 原生支持 `bytes`；JSON 把 `bytes` 编成 `{"$g2": "bytes", "encoding": "base64", "data": ...}`。**`bytes` 的列表两种编码都已支持，codec 无需改动。**
- `src/g2_policy_protocol/diagnostics/samples.py`：`synthetic_context` 为 `g2-policy-check` 构造合成图像历史，目前固定 raw。
- `tools/check_stack.py`：跨仓库联调。客户端在 `history > 1` 时手工构造 `context["image_history"] = {"head_color": np.full((2, *size, 3), 128, np.uint8)}`（raw）。
- 测试：`tests/test_protocol.py::test_history_context_and_timestamps`、`tests/test_server_tools.py::test_synthetic_image_history_uses_negotiated_cameras_and_size`。
- 文档：`docs/protocol.md` 的 context 表写着 `image_history` "只用 raw"。

### 2.2 wholebody（客户端）

- `external_services/policy/runtime/obs/history.py`，`ObservationStream.snapshot`：
  ```python
  if 'image_history' in self.spec['context_keys']:
      raw_meta = {**self.metadata, 'image_encoding': 'raw'}
      prepared = [prepare_images(item.images, raw_meta) for item in selected]
      values['image_history'] = {camera: np.stack([item[camera] for item in prepared])
                                 for camera in self.metadata['required_cameras']}
  ```
  `selected` 是按 `history_hz` 选出的 `history_steps` 个样本（最后一个即最新观测）。`snapshot` 在推理生产者线程里调用，**这段耗时直接计入推理延迟**。
- `external_services/policy/runtime/images.py`：`prepare_images(images, metadata)` 按 metadata 缩放（resize/letterbox，OpenCV），`image_encoding == "jpeg"` 时用 `jpeg_encode_rgb`（OpenCV，quality 90，内部 RGB→BGR）编码。
- `external_services/policy/runtime/transport.py`，`G2WebSocketPolicyClient.infer`：对当前帧调用 `prepare_images(images, self._metadata)`，然后 `protocol.build_request(..., context=context)`。`InferStats.letterbox_s` 只统计当前帧准备。
- 域控 CPU 紧张，线程与核绑定见 `docs/dev/cpu-resources.md`；JPEG 编码要考虑开销（§4.4）。

### 2.3 openpi（服务端）

- `src/openpi/adapters/g2/server.py`，`OpenPIBackend.infer`：
  ```python
  images = request["context"].get("image_history") or decode_images(request["images"], self.metadata)
  ```
  有历史时直接使用历史数组（raw），否则解码当前帧。`decode_ms` 计入 `server_timing`。`_warmup_jpeg_transport` 在监听前预热 JPEG 解码。
- `image_encoding` 来自部署 YAML `scripts/g2/configs/deploy.yaml` 的 `server.image_encoding`（默认 `jpeg`）。
- 历史帧随后在 `src/openpi/adapters/g2/transforms.py::_prepare_images` 横向拼图，与本计划无关。
- 测试：`src/openpi/adapters/g2/tests/transport_test.py::test_real_advanced_server_transports_history_and_typed_joint_actions`（构造 raw 历史，服务端以 `image_encoding="raw"` 启动）。

### 2.4 lerobot-finetune（服务端）

- `src/lerobot_finetune/deploy/g2_server.py`，`G2Backend.infer`：
  ```python
  histories = request["context"].get("image_history")
  images = decode_images(request["images"], self.metadata) if histories is None else histories
  ```
  `metadata_for(runtime, image_encoding)` 的编码来自部署 YAML `server.image_encoding`（默认 `jpeg`）。
- 注意：lerobot 当前支持的 ACT / pi0 / pi05 不接受多帧观测（`observation_delta_indices` 为 None），训练入口会拒绝 `history_steps > 1`；只有 diffusion 等模型会走到这条路径。服务端仍须正确处理。
- 测试：`scripts/g2/tests/test_backend.py::test_history_is_used_without_decoding_unused_current_images`——用 monkeypatch 禁止调用 `decode_images`，并传入 raw 历史，而 `metadata_for` 默认声明 `jpeg`。**改动后这个测试必须调整**（§5）。

## 3. 目标与非目标

目标：

1. `image_encoding: jpeg` 时，`image_history` 每帧以 JPEG 传输，体积降到约十分之一。
2. 服务端拿到的历史仍是解码后的 RGB `uint8 (L, H, W, 3)`，形状、dtype 和色彩约定不变，像素会受 JPEG 有损压缩影响。
3. `image_encoding: raw` 时行为完全不变。

非目标（本计划不做，见 §8 后续项）：

- 拆分位姿历史与图像历史的帧数。
- 服务端缓存历史帧（协议设计上历史由客户端时间轴决定，服务端无状态）。

## 4. 设计

### 4.1 线上格式

`image_history` 的格式**跟随握手的 `image_encoding`**，与当前帧规则一致：

| `image_encoding` | `image_history[camera]` |
|---|---|
| `raw`（不变） | `uint8` 数组 `(L, H, W, 3)` |
| `jpeg`（新） | 长度为 `L` 的列表，每项为非空 `bytes`（一帧 JPEG），从旧到新 |

选择列表而非 `(L,)` 对象数组：codec 不接受 object dtype；`bytes` 列表在 MessagePack / JSON 下都已有确定表示。

### 4.2 校验分工（与当前帧一致）

- `protocol.parse_request` / `parse_context`（两端都调用）：只做**表示层**检查——jpeg 时为列表、长度 `== history_steps`、每项为非空 `bytes`；raw 时保持现有全部检查。协议核心不解码图像。
- 服务端解码后检查**尺寸**：新增 `images.decode_image_history(history, metadata)`，逐帧解码为 RGB、`np.stack` 成 `(L, H, W, 3)`，再按 `image_sizes` 检查尺寸。

### 4.3 兼容性

这是已有字段的格式变化。按项目规则（[兼容性如何保证](../install.md#兼容性如何保证)），结构不变但含义变化的改动需要升级 `PROTOCOL`；但这里**旧客户端发 raw 数组、新服务端期望 `bytes` 列表**（或反之）会被表示层检查直接拒绝，不会被静默误读，所以**不升级 `g2.policy.v3`**。为了让报错一眼看出原因，表示层检查失败时在信息末尾追加 `_messages.VERSION_HINT`（"the client and server may run different g2-policy-protocol versions"）。

部署时同步更新两端协议仓、wholebody 和对应模型服务的适配代码，再重启 GPU server、启动域控 client。JPEG 历史格式不兼容会在首个相关请求中被拒绝；新校验器附带版本提示，旧校验器可能只报格式错误。raw 与无图像历史的请求不受影响。

### 4.4 客户端编码开销

`snapshot` 每次推理都会重新准备 `history_steps` 帧；相邻两次推理的历史帧大量重叠，逐次重新 JPEG 编码是浪费，而且在推理延迟路径上。

实现：**按样本缓存准备好的图像**。以 `id(obs)` 为键保存样本引用和准备结果，引用避免 ID 重用；缓存随历史缓冲淘汰，episode 首帧与 `self.first` 一起保留。`snapshot` 用独立图像锁保护缓存，先查缓存再编码，不占采样锁。不要在采样线程里对每个样本预先编码（30 Hz × 相机数的编码量会持续占用域控 CPU，而多数样本永远不会被选中）。

后续优化：历史请求不再单独准备和发送 `images`，当前帧直接使用历史最后一帧，避免重复缩放、编码、打包和传输。无需新增 prepared 参数或编码状态标记。

### 4.5 JPEG 质量与色彩

- 客户端用 OpenCV 编码（`jpeg_encode_rgb`，quality 90，先 RGB→BGR），服务端用 Pillow 解码（得到 RGB）。当前帧已是这条路径，结果正确。
- 历史帧与当前帧使用相同传输方式，但视频解码后再进行 JPEG 编码会增加一次有损压缩；对模型效果的影响需用代表性图像和实机任务验证。

## 5. 实施步骤

按顺序进行，每步跑对应测试。

### 步骤 1：协议包（`g2-policy-protocol`）

1. `protocol.py` 的 `_parse_image_history` 按 `image_encoding` 分支校验：jpeg 必须是 `list`/`tuple`、长度 `== steps`、每项为非空 `bytes`，规范化为列表；raw 分支保持现状。表示不匹配的错误追加 `_messages.VERSION_HINT`。单次校验直接放在原循环内，不另抽函数。
2. `images.py` 新增：
   ```python
   def decode_image_history(history, metadata):
       """JPEG (or raw) image history -> {camera: uint8 (L, H, W, 3)}, sizes checked against metadata."""
   ```
   - jpeg：逐帧复用 `decode_images` 的私有 `_decode_jpeg(name, data)`，损坏时抛 `ImageDecodeError`（`ValueError` 子类），与当前帧一致；用 `np.stack` 合并。
   - raw：原样返回（已在 `parse_context` 校验）。
   - 尺寸：若 `metadata["image_sizes"]` 声明了该相机，检查每帧 `(H, W)`，否则只要求各帧尺寸一致。
   - `PolicyServer._run_model` 将 `ImageDecodeError` 归为 `invalid_request`，模型的其他异常仍为 `inference_error`。
3. `diagnostics/samples.py` 的 `synthetic_context`：jpeg 时用 `prepare_images(sample, metadata)` 的输出直接生成 `[image] * steps`，不重复编码；raw 时保持现状。
4. `tools/check_stack.py`：客户端构造 `image_history` 时按 `metadata["image_encoding"]` 生成（jpeg 用 `images.encode_jpeg`）。两个第一方 server 默认 jpeg，所以改完后 `history=2` 的用例会覆盖新路径。
5. 测试（`tests/`）：
   - `test_protocol.py`：新增 jpeg 历史的接受用例（JSON 与 MessagePack 各往返一次）与拒绝用例：长度不对、非列表、空 bytes、raw 数组配 jpeg metadata、bytes 列表配 raw metadata；拒绝信息含 `different g2-policy-protocol versions`。保留现有 raw 用例。
   - 新增 `decode_image_history` 测试：正常解码形状/dtype、尺寸与 `image_sizes` 不符、损坏 JPEG 抛 `ValueError`。可放 `test_server_tools.py` 或新建 `test_images.py`。
   - `test_server_tools.py::test_synthetic_image_history_...`：拆成 raw / jpeg 两个参数化用例。
6. 文档：`docs/protocol.md` 的 context 表与"图像"节改为"图像历史编码跟随 `image_encoding`；jpeg 时为每帧 bytes 列表"，并在"接入模型"示例的 backend 里提示用 `decode_image_history`。`CHANGELOG.md` 加一条（按日期）。

### 步骤 2：wholebody（客户端）

1. `obs/history.py` 的 `snapshot`：去掉强制 `'image_encoding': 'raw'`，按握手 metadata 准备：
   ```python
   if 'image_history' in self.spec['context_keys']:
       prepared = ...  # 从样本缓存取得结果，缺失时调用 prepare_images，见 §4.4
       if self.metadata['image_encoding'] == 'jpeg':
           history = {camera: [item[camera] for item in prepared] for camera in cameras}
       else:
           history = {camera: np.stack([item[camera] for item in prepared]) for camera in cameras}
   ```
   `selected` 是样本对象列表（`PolicyObservation`），缓存实现见 §4.4。保持 `snapshot` 线程安全，在 `self.lock` 内取样本快照，编码在锁外做。
2. `transport.py` 无需改动（`build_request` 会走协议包的新校验）。
3. 测试：`tests/test_policy_capabilities.py` 已有 `ObservationStream` 历史用例（搜索 `history_steps=2`）。新增：jpeg metadata 下 `snapshot().context['image_history'][camera]` 是长度 `L` 的 `bytes` 列表；raw 下是 `(L, H, W, 3)` 数组；连续两次 `snapshot` 不重复编码同一样本（可 monkeypatch `prepare_images` 计数）。
4. 文档：`docs/dev/policy-image-transport.md` 补一句图像历史也随 `image_encoding` 压缩。

### 步骤 3：openpi（服务端）

1. `src/openpi/adapters/g2/server.py`，`OpenPIBackend.infer`：
   ```python
   history = request["context"].get("image_history")
   images = (decode_images(request["images"], self.metadata) if history is None
             else decode_image_history(history, self.metadata))
   ```
   `decode_ms` 计时包住这两种解码。可在 `_warmup_jpeg_transport` 里额外解码一组历史（同一解码器，非必须）。
2. 测试：`transport_test.py::test_real_advanced_server_transports_history_and_typed_joint_actions` 改为参数化 `image_encoding` ∈ {raw, jpeg}；jpeg 时客户端用 `images.encode_jpeg` 构造历史，并断言模型收到的图像形状 `(2, 224, 224, 3)`。

### 步骤 4：lerobot-finetune（服务端）

1. `src/lerobot_finetune/deploy/g2_server.py`，`G2Backend.infer`：
   ```python
   histories = request["context"].get("image_history")
   images = decode_images(request["images"], self.metadata) if histories is None \
       else decode_image_history(histories, self.metadata)
   ```
2. 测试：`scripts/g2/tests/test_backend.py::test_history_is_used_without_decoding_unused_current_images`——该测试禁止调用 `decode_images`（当前帧），改动后保持这一点；历史改为按 `metadata_for` 的编码构造（默认 jpeg：`[encode_jpeg(frame)] * 2`），或显式 `metadata_for(obj, "raw")` 并另加一个 jpeg 用例。

## 6. 验证

### 6.1 各仓库测试命令

```bash
# 协议包（约 1 秒）
cd ~/g2-policy-protocol && ../lerobot-finetune/.venv/bin/python -m pytest -q

# openpi
cd ~/openpi && .venv/bin/python -m pytest -q src/openpi/adapters/g2/tests scripts/g2/tests

# lerobot-finetune
cd ~/lerobot-finetune && .venv/bin/python -m pytest -q tests scripts/g2/tests

# wholebody（本机没有 wholebody 专用 venv；借用 lerobot 的 venv，并让 `wholebody` 可导入）
cd ~/wholebody && PYTHONPATH=$HOME ../lerobot-finetune/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_policy_capabilities.py tests/test_g2_policy_transport.py tests/test_policy_control_loop.py
```

改动开始前先各跑一遍记录基线。已知的既有失败（与本计划无关，不需要修）：

- lerobot：`scripts/g2/tests/test_training_hooks.py::test_pinned_lerobot_submodule_is_unmodified`（`third_party/lerobot` 子模块有本地修改）。
- wholebody 全量测试：本机缺 `grpc`、`ppadb`、`importlinter`，9 个文件无法收集；另有 13 个遥操/启动配置相关失败（`test_vr_follow_profiles`、`test_launch_env`、`test_shared_launch`、`test_umi_gripper` 等）。用"改动前后失败列表逐条对比"判断是否引入回归。

### 6.2 跨仓库联调

```bash
# 四个仓库的共同父目录；用真实 wholebody 客户端 + 两个 backend，覆盖 history 1/2、两种编码
cd ~ && lerobot-finetune/.venv/bin/python g2-policy-protocol/tools/check_stack.py --stack-root "$PWD"
```

另建议跑附录 A 的 UMI 端到端脚本，它用真实的 `ObservationStream` 采历史，能覆盖客户端编码与缓存路径（`check_stack` 的客户端是手工构造历史的）。改完后脚本里服务端以 `image_encoding="jpeg"` 启动即覆盖新路径；断言 `image.shape == (2, 224, 224, 3)` 不变。

### 6.3 实机检查

- 推理日志 / `metrics.csv` 的 `wire_kb_last` 在开启图像历史时应明显下降（单臂两相机、两帧历史：从约 600 KB 降到约 100 KB 量级）。
- `server_decode_ms` 会因多解码 `L × 相机数` 帧而增加（224×224 每帧约 1 ms 量级），确认总往返时间下降。
- 关注域控 CPU：对比改动前后推理生产者线程的耗时（`snapshot` 内的编码）。

## 7. 验收标准

- [x] `image_encoding: jpeg` 且 `history_steps > 1` 时，线上 `image_history` 为每帧 JPEG `bytes` 列表；`raw` 时与现在一致。
- [x] 两个服务端拿到的历史解码后为 `uint8 (L, H, W, 3)`，尺寸与 `image_sizes` 一致；尺寸不符或损坏 JPEG 回 `invalid_request`。
- [x] JPEG 历史新旧表示不匹配会被请求校验拒绝；新校验器信息含 `different g2-policy-protocol versions`，旧校验器可能只有格式错误。
- [x] 客户端连续推理不重复编码同一历史样本。
- [x] §6.1 四个仓库测试无新增失败；`check_stack.py` 与附录 A 脚本通过。
- [x] `docs/protocol.md`、`CHANGELOG.md`、wholebody `docs/dev/policy-image-transport.md` 已更新。
- [ ] 实机验证带宽、往返延迟、域控 CPU 和模型任务效果（§6.3）。

2026-11-02 离线验证记录：协议包 320 passed；openpi 509 passed（基线 507）；wholebody 定向 39 passed（基线 37）；lerobot 155 passed，仅已知的 `test_pinned_lerobot_submodule_is_unmodified` 失败。`check_stack.py` 的两个 backend、两种动作、history 1/2、JSON/MessagePack 共 16 个用例通过；附录 A 输出 `UMI end-to-end OK`。缓存测试覆盖首帧填充、连续快照、重叠历史和淘汰；图像错误测试覆盖损坏内容、非 JPEG 和错误尺寸，并确认模型 `ValueError` 仍为 `inference_error`。离线测试需要允许本机回环端口和线程运行，沙箱限制导致的失败不计作代码回归。

后续去重优化验证：协议包 323 passed；wholebody 相关 43 passed；openpi transport 14 passed；lerobot backend 6 passed；跨仓库 16 个用例与附录 A 再次通过。新增测试检查 raw/JPEG、JSON/MessagePack 下省略 `images`、旧请求仍可解析、无历史时拒绝缺图、非法历史不回退到当前图像，且 transport 不调用当前帧准备。同一组合成的两相机、两帧历史，MessagePack 请求 raw 从 903,561 减至 602,369 bytes，JPEG 从 273,727 减至 182,513 bytes，均约减少 33.3%；仅证明去重收益，不代表真实画面压缩比或实机延迟。

## 8. 风险与后续项

- **客户端 CPU**：按样本缓存后，每个被选中的样本只编码一次；若实机仍吃紧，可降低 `history_hz`（随训练 `sample_hz`）或考虑 §8 第 2 项。
- **JSON 编码**：`bytes` 走 Base64，比 MessagePack 大约 33%，仍远小于 raw 数字列表。
- 后续项（不在本计划内）：
  1. **已完成**：有完整 `image_history` 时省略当前帧 `images`，无历史时仍必需；两个第一方 backend 原本就优先使用历史，无需额外分支。
  2. 拆分 `state_history_steps` / `image_history_steps`，让 UMI 风格可以"位姿多帧、图像单帧"，同时解决 lerobot ACT/pi05 无法使用位姿历史、openpi 横向拼图导致单帧分辨率减半的问题。

## 附录 A：UMI 端到端联调脚本

用真实 openpi `G2PolicyServer` + `G2Policy`（假原生模型）和真实 wholebody `ObservationStream` + `G2WebSocketPolicyClient` 跑一遍 UMI 风格配置（state/action 均 `se3` 相对当前末端、`history_steps: 2`）。服务端运行在 openpi venv，客户端运行在可导入 wholebody 的 venv。

运行：

```bash
S=/tmp/umi_e2e.py   # 把下面的代码保存到这个绝对路径
(cd ~/openpi && exec .venv/bin/python -I "$S" serve > /tmp/umi_server.log 2>&1) &
SERVER=$!
for i in $(seq 1 60); do PORT=$(grep -E '^[0-9]+$' /tmp/umi_server.log | head -1); [ -n "$PORT" ] && break; sleep 0.5; done
cd ~ && PYTHONPATH=$HOME lerobot-finetune/.venv/bin/python "$S" client "$PORT"
kill $SERVER
```

服务端启动日志第一行可能是 transformers 的警告，所以端口要用 `grep` 取纯数字行。

```python
"""End-to-end check of a UMI-style policy (state+action se3 relative to current EE, history 2)."""
import asyncio
import sys
import time

import numpy as np

H = 4
HISTORY = 2
SAMPLE_HZ = 10.0
BRANCH = {"delta": True, "method": "se3", "reference": {"source": "state", "scope": "chunk"}}
RAW = {"state": {"pose": BRANCH}, "action": {"pose": BRANCH},
       "timing": {"history_steps": HISTORY, "sample_hz": SAMPLE_HZ}}


def rot6d_matrix(pose10):
    """Independent rot6d -> 4x4 (columns), not using the protocol geometry module."""
    a, b = np.asarray(pose10[3:6], float), np.asarray(pose10[6:9], float)
    x = a / np.linalg.norm(a)
    b = b - x * (x @ b)
    y = b / np.linalg.norm(b)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, np.cross(x, y)], axis=1)
    T[:3, 3] = pose10[:3]
    return T


def matrix_pose10(T, grip):
    return np.concatenate([T[:3, 3], T[:3, 0], T[:3, 1], [grip]])


def serve():
    from types import SimpleNamespace

    from g2_policy_protocol.data.encoding import encode_state
    from g2_policy_protocol.data.policy import resolve_policy
    from websockets.asyncio.server import serve as ws_serve

    from openpi.adapters.g2.policy import G2Policy
    from openpi.adapters.g2.server import G2PolicyServer

    settings = resolve_policy(RAW, horizon=H)

    def infer(obs):
        history = obs["observation/pose_history"]
        assert history.shape == (HISTORY, 20), history.shape
        image = obs["observation/image"]
        assert image.shape == (HISTORY, 224, 224, 3), image.shape  # image history reaches the model
        state = encode_state(settings, obs, "right", flatten=False)
        current = rot6d_matrix(obs["observation/pose"][10:])
        for index in range(HISTORY):
            expected = matrix_pose10(np.linalg.inv(current) @ rot6d_matrix(history[index, 10:]), history[index, 19])
            np.testing.assert_allclose(state[index], expected, atol=1e-5)
        # Fake model: move +1 cm per step along the current EE's own x axis, gripper 0.5.
        actions = np.zeros((H, 10), np.float32)
        actions[:, 0] = 0.01 * np.arange(1, H + 1)
        actions[:, 3:9] = [1, 0, 0, 0, 1, 0]
        actions[:, 9] = 0.5
        return {"actions": actions}

    native = SimpleNamespace(metadata={"arm": "right"}, infer=infer, reset=lambda: None)
    server = G2PolicyServer(
        G2Policy(native, settings),
        metadata={"policy_contract": settings, "arm": "right", "action_horizon": H,
                  "required_cameras": ["head_color"], "requires_prompt": False},
        image_encoding="jpeg",
    )

    async def run():
        async with ws_serve(server._handler, "127.0.0.1", 0, compression=None, max_size=None) as listener:
            print(listener.sockets[0].getsockname()[1], flush=True)
            await listener.serve_forever()

    asyncio.run(run())


def client(port):
    from wholebody.external_services.policy.runtime.obs.builder import build_observation
    from wholebody.external_services.policy.runtime.obs.history import ObservationStream
    from wholebody.external_services.policy.runtime.transport import G2WebSocketPolicyClient

    start = time.monotonic()
    image = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)

    def pose_at(t):
        angle = 0.8 + 0.5 * t
        T = np.eye(4)
        T[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        T[:3, 3] = [0.4 + 0.05 * t, -0.2, 0.9]
        left = np.array([0.3, 0.2, 0.8, 1, 0, 0, 0, 1, 0, 0.0])
        return np.concatenate([left, matrix_pose10(T, 0.2)]).astype(np.float32)

    def observe():
        return build_observation(task="", require_task=False, joint_state=np.zeros(16),
                                 ee_pose=pose_at(time.monotonic() - start), images={"head_color": image},
                                 camera_ids=("head_color",))

    with G2WebSocketPolicyClient(port=port) as policy:
        meta = policy.metadata
        assert set(meta["observation_spec"]["context_keys"]) == {"pose_history", "image_history"}
        stream = ObservationStream(observe, meta)
        stream.start()
        try:
            time.sleep(0.6)
            for attempt in range(2):
                obs = stream.snapshot()
                kwargs = obs.as_infer_kwargs()
                kwargs["observation_time_ns"] = round(obs.captured_at * 1e9)
                actions = policy.infer(**kwargs)
                current = rot6d_matrix(obs.ee_pose[10:])
                for step in range(H):
                    local = np.eye(4)
                    local[0, 3] = 0.01 * (step + 1)
                    np.testing.assert_allclose(actions[step, 10:], matrix_pose10(current @ local, 0.5), atol=1e-5)
                np.testing.assert_allclose(actions[:, :10], np.tile(obs.ee_pose[:10], (H, 1)), atol=1e-6)
                if attempt == 0:
                    policy.reset()
        finally:
            stream.close()
    print("UMI end-to-end OK")


if __name__ == "__main__":
    serve() if sys.argv[1] == "serve" else client(int(sys.argv[2]))
```
