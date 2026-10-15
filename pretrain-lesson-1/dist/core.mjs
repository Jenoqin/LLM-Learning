export const sections = [
  {id:'map', title:'训练地图', label:'整个过程发生了什么'},
  {id:'tokens', title:'一句话的练习题', label:'答案就藏在原文里'},
  {id:'cycle', title:'亲手更新参数', label:'从误差到一次改变'},
  {id:'evidence', title:'读懂真实记录', label:'用证据还原训练'},
  {id:'terms', title:'九个概念的原理', label:'先问为什么需要它'},
  {id:'quiz', title:'检查我的理解', label:'合上讲义，试着解释'},
];
export const tokens = [
  {text:'BOS',id:1,special:true},{text:'春',id:3102},{text:'天',id:993},{text:'到了',id:4520},
  {text:'，',id:294},{text:'花',id:1884},{text:'开',id:727},{text:'了',id:423},{text:'。',id:302},
  {text:'EOS',id:2,special:true},{text:'PAD',id:0,special:true},
];
export const stages = [
  {letter:'A',name:'取一批数据',en:'INPUT',icon:'data',title:'先准备一小批练习题',text:'分词器把文本换成编号，Dataset 准备输入与标签，DataLoader 把它们组成一批。默认一次取 32 条，每条补齐或截断到 340 个位置。',code:'input_ids = input_ids.to(args.device)\nlabels = labels.to(args.device)',why:'标签来自原文中的下一枚 token；PAD 的标签设为 -100，不参与判分。',ref:'dataset/lm_dataset.py · PretrainDataset'},
  {letter:'B',name:'设置步幅',en:'LEARNING RATE',icon:'dial',title:'决定这次更新的步幅',text:'学习率控制优化器更新参数时的步幅。本次从约 0.0005 逐渐下降到 0.00005。设置学习率这一步，还没有改变参数。',code:'lr = get_lr(global_step, total_steps, args.learning_rate)\n# 原脚本用 epoch * iters + step 计算全局位置\nparam_group["lr"] = lr',why:'前期步幅较大，后期较小。它按小批次位置调度，没有 warmup。',ref:'trainer_utils.py · get_lr'},
  {letter:'C',name:'预测与判分',en:'FORWARD',icon:'network',title:'预测下一枚 token，再计算误差',text:'模型为每个位置的 6400 个候选 token 输出分数 logits，再用下一个位置的原文计算交叉熵 loss。正确目标的概率越低，误差代价越大。',code:'res = model(input_ids, labels=labels)\nloss = res.loss + res.aux_loss\nloss = loss / args.accumulation_steps',why:'当前 Dense 模型 aux_loss 为 0。loss 衡量文本预测，不直接判断知识真假。',ref:'model_minimind.py · MiniMindForCausalLM.forward'},
  {letter:'D',name:'计算梯度',en:'BACKWARD',icon:'backward',title:'算出改进方向，参数还没变',text:'backward 把误差传回参数，计算并累加梯度。你可以先把它理解为：为了降低当前误差，各个参数应该怎样调整？',code:'scaler.scale(loss).backward()',why:'backward 本身不会修改参数。默认 bf16 下 scaler 禁用，没有梯度缩放。',ref:'train_pretrain.py · train_epoch'},
  {letter:'E',name:'更新参数',en:'OPTIMIZER',icon:'update',title:'模型在这里真正改变',text:'通常累积 8 个小批次后，先恢复梯度尺度、再裁剪梯度，最后由 AdamW 更新参数。用完的梯度清零，下一组重新积累。',code:'scaler.unscale_(optimizer)\nclip_grad_norm_(model.parameters(), args.grad_clip)\nscaler.step(optimizer)\noptimizer.zero_grad(set_to_none=True)',why:'通常一次更新处理 32 × 8 = 256 条样本。轮末不足 8 批也会更新。',ref:'train_pretrain.py · train_epoch'},
  {letter:'F',name:'记录指标',en:'LOG',icon:'chart',title:'把学习过程留下来',text:'默认每 100 个小批次记录一次 loss、学习率和本轮剩余时间，轮末也记录。SwanLab 将这些记录展示为曲线。',code:'current_loss = loss.item() * args.accumulation_steps\nLogger(f"loss: {current_loss:.4f}, ...")\n# 这里只展示核心语句，完整字段见源文件',why:'记录的是当前批次 loss，不是最近 100 步的平均值，也不是正确率。',ref:'logs/pretrain.log · 784 条指标'},
  {letter:'G',name:'保存成果',en:'CHECKPOINT',icon:'save',title:'保存学到的内容与训练位置',text:'out/ 保存用于推理或 SFT 的模型权重。checkpoints/ 还保存优化器和训练位置，让下次可以恢复。默认每 1000 个小批次保存，轮末也保存。',code:'torch.save(state_dict, ckp)\nlm_checkpoint(..., model=model, optimizer=optimizer,\n              epoch=epoch, step=step)\n# 以上为简化展示；实际权重转为 CPU fp16 保存',why:'模型权重回答“学到了什么”；续训检查点还回答“练习进行到了哪里”。',ref:'trainer_utils.py · lm_checkpoint'},
  {letter:'H',name:'继续下一批',en:'REPEAT',icon:'repeat',title:'保留参数与累积状态，继续下一批',text:'这一批的临时数据引用被清理，已经更新的参数留下来。取下一批、预测、判分、计算梯度……直到走完计划中的所有轮次。',code:'del input_ids, labels, res, loss\n# for 循环继续取下一批',why:'本次遍历两轮，共处理 79,390 个小批次，更新参数 9,924 次。',ref:'最终检查点 · epoch=1, step=39695'},
];
export const terms = [
  ['TOKEN','token','分词器划出的文本单位，模型拿到它的整数编号。','input_ids','“到了”是一枚 token，不一定一个字就是一枚 token。'],
  ['PARAMETER','参数','模型通过训练调整的数值。','model.parameters()','结构和词表固定，本次主要学习的是约 6391 万个参数。'],
  ['LOGITS','预测分数','每个位置对词表中候选 token 给出的分数。','res.logits','logits 尚未归一化成概率；交叉熵直接接收这些分数。'],
  ['LOSS','预测误差','正确目标的概率不够高时，付出的误差代价。','res.loss','loss 小表示更贴近训练文本，不能直接证明知识可靠。'],
  ['GRADIENT','梯度','参数变化怎样影响当前 loss 的方向信息。','loss.backward()','backward 计算梯度，本身没有改变参数。'],
  ['OPTIMIZER','优化器','利用梯度和自身状态来更新参数的规则。','AdamW / optimizer.step()','参数在优化器 step 时改变；优化器也保存自己的状态。'],
  ['BATCH','小批次','一次取出的多条训练样本。','batch_size = 32','正式输入一般为 [32, 340]，最后一批可能不足 32 条。'],
  ['STEP','批次位置','本脚本已经处理了多少个小批次。','step / iters','8 个小批次通常才更新一次，所以 step 不等于更新次数。'],
  ['EPOCH','一轮','按本轮安排遍历一次数据集。','epochs = 2','每轮重新打乱样本顺序；检查点的 epoch 从 0 计数。'],
];
export const questions = [
  {q:'新文本是“雨停了，天晴了。”。给模型的前文是“雨停了，”，下一目标从哪里来？',options:['原文紧接着的下一枚 token','人工另写一份答案','模型自己猜出的结果','日志里的 loss'],answer:0,section:'tokens',explain:'目标来自原文的下一位置。实际以分词器的 token 为单位，不一定一个汉字就是一个 token。'},
  {q:'训练时整句都输入了模型，为什么预测“花”时仍不能抄答案？',options:['因为花的编号比较大','因为模型忘记了后半句','因果注意力限制当前位置只能使用前文，输出与下一位置目标比较','因为 batch_size 足够小'],answer:2,section:'tokens',explain:'完整文本方便并行计算，但每个位置的可见范围受限。标签移位规定猜什么，因果限制规定能看什么。'},
  {q:'同一个正确目标的概率从 20% 提高到 80%，loss 怎样变化？',options:['变大，因为模型更自信','变小，因为 −ln(p) 随 p 增大而减小','不变，只看有没有猜中','变为 80% 正确率'],answer:1,section:'tokens',explain:'该目标的 loss 从约 1.609 降到 0.223。这是给目标概率的判分，不是一份正确率。'},
  {q:'完成前向后，再执行 backward()，还没有执行优化器 step。哪些数值改变了？',options:['模型参数已经更新','参数与梯度都被清零','只要 loss 下降就一定改了参数','计算并累积了梯度，参数仍不变'],answer:3,section:'cycle',explain:'前向产生预测和 loss；反向计算梯度；优化器 step 才改参数。再前向一次，才能检查新参数的预测。'},
  {q:'训练 loss 整体下降，能直接证明所有生成回答都正确吗？',options:['能，训练已经结束','能，只要用了 RTX 4090','不能；它反映训练文本预测，需要独立评测能力','不能，因为没有保存日志'],answer:2,section:'evidence',explain:'当前日志是训练批次的误差。不同批次会波动；语言连贯不等于事实正确，本次没有独立验证集。'},
  {q:'预训练完成后，普通文本生成怎样继续一句话？',options:['每写一个字都重新预训练','选出下一枚 token，追加后继续预测，参数通常保持不变','从日志里查出整段回答','运行 backward() 自动写出答案'],answer:1,section:'terms',explain:'生成时未知未来，需一步步追加 token；普通生成不使用梯度和优化器更新参数。'},
];
export const tokenLoss = p => -Math.log(p);
export const probability = parameter => 1 / (1 + Math.exp(-parameter));
export const prediction = parameter => ({parameter, probability:probability(parameter), loss:tokenLoss(probability(parameter))});
export function initialDemo(accumulation=1, parameter=0){return {parameter,accumulation,gradient:0,batches:0,updates:0,phase:'ready',loss:null,before:null,after:null,message:'前文是“春天到了，”，目标是“花”。先猜：前向预测会改变参数吗？'};}
export function demoTransition(state, action){
  const s={...state};
  if(action==='reset')return initialDemo(s.accumulation);
  if(action==='forward' && s.phase==='ready' && s.batches<s.accumulation){s.before=prediction(s.parameter);s.after=null;s.loss=s.before.loss;s.phase='predicted';s.message='前向完成：得到了分数、概率和 loss。参数还没变。先猜：下一步反向会改参数，还是只计算梯度？';}
  if(action==='backward' && s.phase==='predicted'){
    s.gradient+=(probability(s.parameter)-1)/s.accumulation;s.batches++;s.phase=s.batches===s.accumulation?'accumulated':'ready';
    s.message=`已累积 ${s.batches} / ${s.accumulation} 批梯度。参数仍为 ${s.parameter.toFixed(3)}。${s.phase==='accumulated'?'梯度为负，沿负梯度更新会让参数增大。':'下一批继续累积，不更新参数。'}`;
  }
  if(action==='fill' && s.phase==='ready' && s.batches>0 && s.batches<s.accumulation){s.gradient+=(s.accumulation-s.batches)*(probability(s.parameter)-1)/s.accumulation;s.batches=s.accumulation;s.phase='accumulated';s.message='补齐了剩余批次的前向与反向。梯度已累积，参数仍未改变。';}
  if(action==='step' && s.phase==='accumulated'){
    const gradient=s.gradient;
    s.parameter-=0.8*gradient;s.gradient=0;s.batches=0;s.updates++;s.phase='updated';
    s.message=`更新完成：w ← w − 0.8 × (${gradient.toFixed(3)})，参数变为 ${s.parameter.toFixed(3)}；梯度清零。先猜概率与 loss 怎样变化，再点“重新预测”。`;
  }
  if(action==='verify' && s.phase==='updated'){s.after=prediction(s.parameter);s.loss=s.after.loss;s.phase='ready';s.message='重新前向：对比下方更新前后结果。这里重复同一个目标，所以概率上升、loss 下降；正式训练更换批次后不保证每次 loss 都下降。';}
  return s;
}
export function gradeQuiz(answers){return questions.reduce((total,q,i)=>total+(answers[i]===q.answer?1:0),0);}
export function completionMessage(completed){return sections.every(s=>completed.includes(s.id))?'六节已标记理解，自测已通过。请再用自己的话解释一次。':'自测已通过；其余章节仍可继续学习，进度按各节分别记录。';}
export function updateCount(point){return (point.epoch-1)*Math.ceil(point.total/8)+(point.step===point.total?Math.ceil(point.step/8):Math.floor(point.step/8));}
