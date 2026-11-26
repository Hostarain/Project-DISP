# MEMORY.md（已迁移，仅保留指针）

> **内容已并入 [`AGENTS.md`](AGENTS.md)。**
> 原因：DSH 的项目指令加载器（`dsh-agent-instructions`）只认 `AGENTS.md` / `CLAUDE.md` 及其 `.local.md` 覆盖层，**不解析 `@` 导入**，所以原先靠 `CLAUDE.md` 里的 `@MEMORY.md` 引用在这里不生效。
> 现在：DSH 直接自动加载 `AGENTS.md`；Claude Code 通过 `CLAUDE.md` 末尾的 `@AGENTS.md` 导入同一份内容。
> 迁移时间：2026-10-28。请勿在本文件继续维护内容。
