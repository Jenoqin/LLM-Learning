# LLM Learning 预训练第一课 · 交互学习应用

从 [课程讲义](dist/assets/lesson-1.md) 查看本课学习资料。

默认主线约 35–45 分钟：四步概览 → 同一句话的目标与概率 → 单次更新前后对照 → 三个真实日志片段 → 六道迁移自测。八步骤工程地图、8 批累积和完整公式可展开学习。

六个环节：训练地图、token 预测、参数更新演示、真实日志、九个概念的第一性原理、课后自测。

概念讲解按“为什么需要 → 本质 → 小例子 → 对应训练代码 → 检验理解”组织，可点击选择概念并展开理解检查的答案。

纯 HTML/CSS/JavaScript，浏览器本地保存学习进度与笔记。分词编号来自项目分词器，784 条记录来自已完成的 RTX 4090 预训练。概率滑块和单参数更新为明确标注的教学演示，不执行真实模型训练。

## 一键启动与关闭（macOS）

在 Finder 中打开这个目录，双击：

- **`启动课程.command`**：后台启动服务器，自动打开浏览器。重复启动会复用已有服务器。
- **`关闭课程.command`**：关闭本应用的服务器；浏览器中的笔记和学习进度保留。

无需预先激活虚拟环境。优先使用项目的 `.venv`，解压到其他位置时使用系统可用的 Python 3。默认端口为 8765，被占用时尝试 8766–8784；终端会显示实际地址。

启动状态与日志位于 `.runtime/`，不作为网页资源提供，不纳入 Git 或应用压缩包。关闭脚本通过本应用专属的关闭接口操作，不会按端口杀死其他服务。

也可以在终端运行（在本目录）：

```bash
python3 local_server.py start
python3 local_server.py stop
```

如果启动没有成功：

- 终端现在会直接显示本次服务器启动的具体错误；完整日志在 `.runtime/server.log`。
- 已显示“访问地址”但浏览器未打开时，复制该地址到浏览器即可。
- 解压后若双击提示没有执行权限，在该目录的终端运行 `bash ./启动课程.command`；也可执行 `chmod u+x ./*.command` 恢复双击权限。
- 请保留 `local_server.py`、`dist/` 和两个脚本在同一目录。

原有手动前台启动方式仍可使用（需在该终端按 Ctrl+C 关闭）：

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory dist
```

打开 `http://localhost:8765`。课程资源通过本地 HTTP 加载，不能只双击 HTML 文件。

检查：

```bash
node --check dist/app.js
node checks.mjs
node ui-checks.mjs
```

这是 LLM Learning 的独立网页应用，可克隆后在本地运行。可选 PyTorch 实验需要完整的 [MiniMind 项目](https://github.com/Jenoqin/minimind)。
