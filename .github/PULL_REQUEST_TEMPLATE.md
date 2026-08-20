---
name: Pull Request
about: 提交代码变更
title: ""
labels: []
assignees: ""
---

## 变更内容 / Changes

[描述你的改动，修复了什么/新增了什么]

## 关联 Issue / Related Issue

Fixes #[issue number]（如有）

## 测试情况 / Testing

- [ ] 已在 RK3588 板编译通过（`make all` + `make check-libs`）
- [ ] 运行 `panorama_output_lease_test` 输出 `passed=1`
- [ ] 代码通过 CI 语法检查

## 检查清单 / Checklist

- [ ] 代码遵循项目现有风格（C++17，与现有代码一致）
- [ ] 未引入对 OpenCV 的**核心库**依赖（仅 apps/tests 可用）
- [ ] 常量/几何值有注释说明来源
- [ ] 未包含敏感信息（路径、密钥、个人信息）
- [ ] README/文档已同步更新（如涉及）

## 备注 / Notes

[其他需要 maintainer 知道的信息]
