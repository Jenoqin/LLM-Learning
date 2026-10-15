#!/usr/bin/env python3
"""第一课：用项目真实模型实现做一次 CPU 更新，不读取正式权重。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / 'model/model_minimind.py').is_file()), None)
if ROOT is None:
    raise SystemExit('请将脚本放回完整 MiniMind 项目的 scripts/ 目录后运行。')
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-json', type=Path, help='可选：把本次实测结果保存为 JSON')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'model'), local_files_only=True)
    config = MiniMindConfig(hidden_size=64, num_hidden_layers=2, max_position_embeddings=64)
    model = MiniMindForCausalLM(config).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    text = '春天到了，花开了。'
    body = tokenizer(text, add_special_tokens=False).input_ids
    ids = [tokenizer.bos_token_id] + body + [tokenizer.eos_token_id]
    x = torch.tensor([ids + [tokenizer.pad_token_id] * (32 - len(ids))])
    labels = x.clone()
    labels[x == tokenizer.pad_token_id] = -100
    print('正文:', text)
    print('input_ids:', x[0].tolist())
    print('labels:', labels[0].tolist())
    print('每个输出位置与下一位置的目标对照：')
    for index in range(len(ids)):
        target = int(labels[0, index + 1])
        print(index, tokenizer.decode([int(x[0, index])]), '→',
              '忽略' if target == -100 else tokenizer.decode([target]))

    # 前向：产生预测和 loss。手工对齐目标，验证模型内部的标签移位。
    parameter = model.model.embed_tokens.weight
    before = parameter.detach().clone()
    result = model(x, labels=labels)
    manual_loss = torch.nn.functional.cross_entropy(
        result.logits[:, :-1, :].reshape(-1, config.vocab_size),
        labels[:, 1:].reshape(-1), ignore_index=-100,
    )
    torch.testing.assert_close(result.loss, manual_loss)
    # 反向：仅计算梯度。step：真正更新参数。然后在同一道题上重新前向。
    result.loss.backward()
    unchanged = torch.equal(before, parameter.detach())
    gradient_norm = parameter.grad.norm().item()
    assert unchanged and gradient_norm > 0
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    difference = (parameter.detach() - before).abs().max().item()
    assert difference > 0
    model.eval()
    with torch.no_grad():
        after_loss = model(x, labels=labels).loss.item()
        # 索引 4 是“，”；它预测索引 5 的“花”。改动更后面的索引 6。
        changed = x.clone()
        changed[0, 6] = (changed[0, 6] + 1) % config.vocab_size
        a, b = model(x).logits, model(changed).logits
        prefix_difference = (a[:, :5] - b[:, :5]).abs().max().item()
        suffix_difference = (a[:, 6:] - b[:, 6:]).abs().max().item()
    assert prefix_difference < 1e-6 and suffix_difference > 0
    evidence = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'source': 'scripts/lesson1_micro_lab.py',
        'environment': {'torch': torch.__version__, 'device': 'cpu', 'seed': 42},
        'model': {'hidden_size': 64, 'num_hidden_layers': 2,
                  'vocab_size': config.vocab_size, 'weights': 'random initialization'},
        'text': text, 'input_ids': x[0].tolist(),
        'valid_targets': int((labels[:, 1:] != -100).sum()),
        'update': {'loss_before': result.loss.item(), 'manual_loss': manual_loss.item(),
                   'parameters_unchanged_after_backward': unchanged,
                   'embedding_gradient_norm': gradient_norm,
                   'embedding_max_change_after_step': difference, 'loss_after': after_loss},
        'causal': {'changed_index_zero_based': 6, 'old_token_id': int(x[0, 6]),
                   'new_token_id': int(changed[0, 6]), 'compared_prefix_positions': 5,
                   'prefix_max_difference': prefix_difference,
                   'changed_suffix_max_difference': suffix_difference},
    }
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    print('通过：标签移位与 loss 一致；反向不改参数；step 后参数改变；前面的预测不受未来改动影响。')
    if args.export_json:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_json.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
