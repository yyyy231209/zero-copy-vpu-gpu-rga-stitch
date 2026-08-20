# 贡献指南 / Contributing Guide

感谢你对本项目感兴趣！任何形式的贡献（bug 报告、文档、代码、想法）都欢迎。

## 🐛 报告问题 / Reporting Issues

1. 先搜索 [Issues](https://github.com/yyyy231209/zero-copy-vpu-gpu-rga-stitch/issues) 是否已存在相同问题
2. 使用 [Bug 报告模板](https://github.com/yyyy231209/zero-copy-vpu-gpu-rga-stitch/issues/new/choose)（请完整填写硬件/软件环境，没有硬件信息的报告无法排查）
3. 附上日志：`dmesg`、程序输出、崩溃栈（gdb bt）

## 🛠️ 提交代码 / Submitting Code

### 环境要求

- RK3588 开发板（aarch64）或 GitHub Actions 交叉编译检查
- GCC 10+ / C++17
- 提交前跑通：`make all`、`make check-libs`

### 开发流程

```bash
# 1. Fork 并克隆
git clone https://github.com/<your-name>/zero-copy-vpu-gpu-rga-stitch.git
cd zero-copy-vpu-gpu-rga-stitch

# 2. 创建分支
git checkout -b feature/your-feature

# 3. 修改 + 本地验证
make all
./build/panorama_output_lease_test assets/open_chain_v1 /tmp/report.txt

# 4. 提交（信息清晰）
git commit -m "feat: add xxx"      # 新功能
git commit -m "fix: xxx"           # 修复
git commit -m "docs: xxx"          # 文档
git commit -m "refactor: xxx"      # 重构
git commit -m "test: xxx"          # 测试

# 5. 推送并创建 PR
git push origin feature/your-feature
```

### 代码风格 / Code Style

- **C++17**，与现有代码风格保持一致（4 空格缩进、`snake_case` 命名）
- 核心库（`src/` `include/`）**不依赖 OpenCV**（只有 `apps/` `tests/` 可用）
- 所有几何常量（尺寸/接缝/槽位）加注释说明来源
- 错误处理：返回错误码 + `fprintf(stderr, ...)`，不抛异常（嵌入式环境）
- 不引入新的第三方依赖（除非必要且讨论过）

### 测试要求 / Testing

| 变更类型 | 必须测试 |
|---|---|
| 核心逻辑 | `panorama_output_lease_test` 通过（`passed=1`） |
| 几何/拼接 | 300 帧回归，输出确定性 |
| 仅文档 | CI 通过即可 |

## 📝 文档贡献 / Documentation

- README/Wiki 修改请保持**中英双语**结构
- 技术术语保留英文原文（DMA-BUF、MPP、RGA、OpenCL 等）
- 文档中的命令必须可复制粘贴执行（含完整参数）

## 🔒 安全注意 / Security

- **绝不**提交：密钥、密码、内网 IP、个人路径（如 `/home/xxx`）
- 摄像头画面截图注意隐私

## 🙏 感谢 / Thanks

再次感谢你的贡献！问题响应时间通常 1~3 天。
