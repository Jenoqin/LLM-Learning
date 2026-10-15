# LLM Learning

An interactive learning lab for understanding large language model pretraining. The course is in Chinese and covers token prediction, loss, gradients, parameter updates, real training logs, core concepts, and a self-check quiz.

Built with HTML, CSS, and JavaScript. Learning progress and notes are saved locally in your browser. No frontend dependencies or build step are required.

## Run locally

Requires Python 3.

```bash
git clone https://github.com/Jenoqin/LLM-Learning.git
cd LLM-Learning/pretrain-lesson-1
python3 -m http.server 8765 --bind 127.0.0.1 --directory dist
```

Open http://localhost:8765. Press Ctrl+C to stop the server.

On macOS, you can also double-click `pretrain-lesson-1/启动课程.command` to start and `pretrain-lesson-1/关闭课程.command` to stop.

## Course materials

- [Course guide](pretrain-lesson-1/README.md)
- [Lesson notes](pretrain-lesson-1/dist/assets/lesson-1.md)
- [Training implementation](pretrain-lesson-1/dist/assets/train_pretrain.py)

The optional PyTorch exercise requires the full [MiniMind project](https://github.com/Jenoqin/minimind). The web course runs independently.

## Checks

Requires Node.js.

```bash
cd pretrain-lesson-1
node --check dist/app.js
node checks.mjs
node ui-checks.mjs
```

## Attribution and license

Based on the MiniMind learning materials and training experiment. Original project references and training evidence are retained. See [LICENSE](LICENSE).
