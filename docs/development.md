# 开发与发布

[返回 README](../README.md) · [协议参考](protocol.md)

本页面向修改共享契约的维护人员。仅部署或接入时，请阅读[域控 / Client](client.md) 或 [GPU Server](server.md)。

## 源码组织与定义来源

```text
src/g2_policy_protocol/
  g2.schema.json   JSON Schema 唯一结构定义（消息、字段、布局、图像、能力）
  _generated.py    由 Schema 生成的常量与静态校验，禁止手改
  _validation.py  生成代码复用的轻量类型/范围检查
  protocol.py     消息构造、解析及动态语义校验
  codec.py        JSON / MessagePack；NumPy / Base64
```

[`g2.schema.json`](../src/g2_policy_protocol/g2.schema.json) 按阅读顺序组织：先看 `metadata / obs / action / error` 四类消息，再看 pose、joint、actions 和 image 的数据表示；末尾是共用的尺寸、计时、请求 ID 和数值类型。`x-g2` 中列出向量字段顺序和单位。

JSON Schema 校验只用于开发测试；运行时不读取 Schema 文件，不逐帧执行完整 Schema 验证。

静态规则由 [`tools/generate_protocol.py`](../tools/generate_protocol.py) 在开发阶段编译成普通 Python：字段、必填项、枚举、类型、范围、集合长度与唯一性，以及计时/错误文本检查。NumPy 向量维度和原始图像规格也从 Schema 提取，运行时不遍历像素或逐元素解释 Schema。

`protocol.py` 保留数组归一化/有限值、非空白 prompt、能力匹配、RTC 和 request_id 对应关系等动态检查；`codec.py` 保留编解码和数据完整性检查。

## 修改与验证

生成器只支持当前使用的静态关键字，遇到新的不支持规则会报错。修改 Schema 后必须重新生成并运行测试；测试也检查修改边界、枚举或必填字段会改变生成代码的行为。

测试依赖 `pytest`、`jsonschema`；发布包不依赖它们。先在开发环境安装本包和测试依赖：

```bash
python -m pip install -e . pytest jsonschema
```

在本仓库根目录执行生成和验证：

```bash
python tools/generate_protocol.py
python tools/generate_protocol.py --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

没有真实权重/GPU 的回环测试只验证协议和模型边界，不代表模型动作质量。

## 打包与跨端升级

打包环境需已安装 setuptools 和 wheel。验证通过后执行：

```bash
python -m pip wheel --no-index --no-deps --no-build-isolation . -w dist
```

当前发布包为 `g2_policy_protocol-0.4.1-py3-none-any.whl`。发布新版本时同步更新版本声明、调用方依赖和相关文档；涉及不兼容契约变更时，明确通信协议的兼容边界。

部署时：

1. 为域控和 GPU 环境分发同一版本的源码或同一个 wheel。
2. 同步更新 wholebody、G2_pi / OpenPI 中相应的接入代码和依赖。
3. 两端安装同一版本协议包。
4. 重启 GPU server 后再启动域控 client，验证能力握手和观测/动作往返。

不能只替换一端。OpenPI 非 G2 机器人的通用原生 server 不受本协议升级影响。

## 对外交付

只交付 wheel 时，接收者可安装 `wheel路径[tools]`，或单独安装 wheel、`websockets>=12` 和 `Pillow>=9`。wheel 包含 `g2-policy-check` / `g2-policy-sample` 命令及其模块；不包含 Markdown 文档和 examples 源码。使用当前源码重新构建 wheel，旧的同版本 wheel 没有新工具。

需要完整文档与示例时运行：

```bash
python tools/build_handoff.py
```

产物 `dist/g2-policy-protocol-handoff.zip` 包含源码、文档、示例、生成工具和测试；不包含 `.git`、构建缓存、模型权重或观测录制。接收者解压后可以独立安装、开发并验收，无需 wholebody、G2_pi 或 OpenPI 仓库。版本要求仍与域控一致；本次增加工具不改变 `g2.policy.v2` 格式。
