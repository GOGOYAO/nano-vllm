# nano-vllm 学习计划

## 目标

从“会使用 vLLM 部署大模型”，成长为“能理解、调优、排障、修改并适配 vLLM 的大模型推理系统工程师”。

完成这个计划后，应当能做到：

1. 从一个 prompt 进入 `LLM.generate()` 开始，完整讲清楚它如何变成 token、`Sequence`、batch、logits、采样结果和最终输出。
2. 能解释 prefill、chunked prefill、decode 的差异，以及为什么 nano-vllm 当前不会把 prefill 和 decode 混在同一个 batch。
3. 能画出 `waiting`、`running`、`finished` 三类状态的流转，并指出 KV cache 在哪里分配、写入、复用和释放。
4. 能根据吞吐、显存、batch 大小、prompt 长度、输出长度的现象，定位到可能相关的配置和代码路径。
5. 能完成一个小型改造，例如增加调度日志、增加采样策略、增加 block/cache 单元测试，或适配一个相近模型结构。

## 学习方法

每个模块都按这个顺序学习：

1. 读入口：先从调用方读到被调用方，不孤立读某个文件。
2. 画状态：把对象、字段、队列、张量 shape 画出来。
3. 打日志：在关键函数临时打印 `seq_id`、`num_tokens`、`num_cached_tokens`、`num_scheduled_tokens`、`block_table`、`slot_mapping`。
4. 改参数：用很小的 `max_num_seqs`、`max_num_batched_tokens`、`max_tokens` 制造可观察的调度行为。
5. 做复盘：每读完一段代码，写下“输入是什么、输出是什么、状态改了什么、可能出错在哪里”。

## 总体路线

| 阶段 | 主题 | 重点文件 | 产出 |
| --- | --- | --- | --- |
| 0 | 跑通和入口建立 | `README.md`, `example.py`, `bench.py`, `nanovllm/config.py` | 能跑通 example，整理配置表 |
| 1 | API 到请求生命周期 | `nanovllm/llm.py`, `nanovllm/engine/llm_engine.py`, `nanovllm/engine/sequence.py`, `nanovllm/sampling_params.py` | Prompt 到 `Sequence` 的调用链图 |
| 2 | Scheduler 和 batch | `nanovllm/engine/scheduler.py`, `nanovllm/engine/block_manager.py` | `waiting/running/finished` 状态机图 |
| 3 | KV cache 和 prefix cache | `nanovllm/engine/block_manager.py`, `nanovllm/engine/model_runner.py`, `nanovllm/layers/attention.py` | block table、slot mapping、ref count 图 |
| 4 | ModelRunner 数据准备 | `nanovllm/engine/model_runner.py`, `nanovllm/utils/context.py` | prefill/decode 张量 shape 对照表 |
| 5 | 模型 forward 和 attention | `nanovllm/models/qwen3.py`, `nanovllm/layers/attention.py`, `nanovllm/layers/linear.py`, `nanovllm/layers/embed_head.py` | Qwen3 单层 forward 图 |
| 6 | Sampler 和输出回写 | `nanovllm/layers/sampler.py`, `nanovllm/engine/scheduler.py` | logits 到新 token 的流程图 |
| 7 | 性能相关能力 | `nanovllm/engine/model_runner.py`, `nanovllm/utils/loader.py`, `nanovllm/layers/linear.py` | Tensor Parallel、CUDA Graph、权重加载笔记 |
| 8 | 调优、排障和改造 | `bench.py` 及上述核心模块 | 一个可验证的小改造 |

## 阶段 0：跑通和入口建立

目标：不要先陷入 attention 细节，先确认整个系统能运行，并知道入口在哪里。

阅读顺序：

1. `README.md`
2. `example.py`
3. `bench.py`
4. `nanovllm/config.py`

任务：

- [ ] 跑通 `python example.py --model_path <local-model-dir>`。
- [ ] 记录当前机器、模型、显存、`tensor_parallel_size`、`enforce_eager`。
- [ ] 整理 `Config` 字段：`max_num_batched_tokens`、`max_num_seqs`、`max_model_len`、`gpu_memory_utilization`、`kvcache_block_size` 分别影响什么。
- [ ] 对比 `example.py` 和 `bench.py`：前者验证 API，后者验证吞吐和长短请求混合。

思考问题：

1. 为什么 `Config.__post_init__()` 要读取 Hugging Face config？
2. `max_model_len` 为什么要和 `hf_config.max_position_embeddings` 取最小值？
3. `bench.py` 为什么直接传 token id，而不是传字符串 prompt？

## 阶段 1：API 到请求生命周期

目标：理解 prompt 如何进入系统，以及 `Sequence` 如何保存请求状态。

阅读顺序：

1. `nanovllm/llm.py`
2. `nanovllm/engine/llm_engine.py`
3. `nanovllm/engine/sequence.py`
4. `nanovllm/sampling_params.py`

主链路：

```text
LLM.generate()
  -> LLMEngine.add_request()
  -> tokenizer.encode()
  -> Sequence(prompt_token_ids, sampling_params)
  -> Scheduler.add()
  -> while not scheduler.is_finished(): LLMEngine.step()
```

重点字段：

| 字段 | 所在对象 | 含义 |
| --- | --- | --- |
| `seq_id` | `Sequence` | 请求序号，用于恢复输出顺序 |
| `token_ids` | `Sequence` | prompt token + completion token |
| `num_prompt_tokens` | `Sequence` | 原始 prompt 长度 |
| `num_cached_tokens` | `Sequence` | 已经在 KV cache 中可复用或已处理的 token 数 |
| `num_scheduled_tokens` | `Sequence` | 本轮被 scheduler 安排处理的 token 数 |
| `block_table` | `Sequence` | 逻辑 block 到物理 KV block 的映射 |
| `temperature/max_tokens/ignore_eos` | `Sequence` | 从 `SamplingParams` 拷贝来的采样和停止参数 |

任务：

- [ ] 画出 `Sequence` 从创建到完成的字段变化。
- [ ] 解释为什么 `completion_token_ids` 可以通过 `token_ids[num_prompt_tokens:]` 得到。
- [ ] 解释 `Sequence.__getstate__()` 和 `__setstate__()` 为什么在 prefill 和 decode 阶段保存不同内容。

思考问题：

1. 一个 prompt 输入后，内部会被包装成什么对象？
2. nano-vllm 里没有单独的 `Request` 类，`Sequence` 承担了哪些 request state？
3. `generate()` 为什么需要用 `seq_id` 重新排序输出？
4. `SamplingParams` 为什么禁止 `temperature <= 1e-10`？

## 阶段 2：Scheduler 和 batch

目标：理解每一轮 step 到底调度什么，以及 prefill/decode 如何切换。

阅读顺序：

1. `nanovllm/engine/scheduler.py`
2. `nanovllm/engine/block_manager.py`
3. 回看 `LLMEngine.step()`

核心事实：

- `waiting` 队列保存尚未完成 prefill 的序列。
- `running` 队列保存已经完成 prefill、正在 decode 的序列。
- `schedule()` 总是先尝试 prefill；只要本轮选中了 prefill 序列，就直接返回，不再混入 decode。
- decode 阶段每个 running sequence 每轮只调度 1 个 token。
- 当 KV block 不够时，scheduler 会 preempt 某些 running sequence，把它们放回 `waiting` 重新 prefill。

Prefill 和 decode 对照：

| 维度 | Prefill | Decode |
| --- | --- | --- |
| 来源队列 | `waiting` | `running` |
| 本轮 token 数 | 一个未缓存区间，可能被 chunk | 每个 sequence 1 个 token |
| `is_prefill` | `True` | `False` |
| 是否可能 chunk | 是，当前实现只允许第一个 seq 被 chunk | 否 |
| 是否写 KV cache | 是 | 是 |
| 是否使用已有 KV cache | prefix cache 命中时使用 | 总是依赖历史 KV |
| batch 是否混合阶段 | 不混合 | 不混合 |

任务：

- [ ] 用 `max_num_seqs=2`、较小的 `max_num_batched_tokens` 运行，观察多个 prompt 如何进入 `waiting` 和 `running`。
- [ ] 制造长 prompt，观察 chunked prefill：`num_scheduled_tokens < num_tokens - num_cached_tokens`。
- [ ] 制造很多请求或降低可用 block，理解 `preempt()` 的触发条件。
- [ ] 画出 `WAITING -> RUNNING -> FINISHED` 和 `RUNNING -> WAITING` 的状态机。

思考问题：

1. Scheduler 每一轮到底在调度 sequence 还是 token？
2. 为什么 prefill 通常比 decode 更适合批量并行？
3. 为什么当前实现只允许第一个 sequence 做 chunked prefill？
4. 一个 batch 里面可以同时包含 prefill 和 decode 请求吗？当前答案是什么？如果想支持混合 batch，需要改哪些地方？

## 阶段 3：KV cache 和 prefix cache

目标：把逻辑 token、逻辑 block、物理 KV block、attention slot 对起来。

阅读顺序：

1. `nanovllm/engine/block_manager.py`
2. `ModelRunner.allocate_kv_cache()`
3. `ModelRunner.prepare_block_tables()`
4. `ModelRunner.prepare_prefill()`
5. `ModelRunner.prepare_decode()`
6. `nanovllm/layers/attention.py`

关键对象：

| 名称 | 含义 |
| --- | --- |
| `Block` | 一个物理 KV cache block 的元信息 |
| `block_id` | 物理 block 编号 |
| `block_table` | 某个 sequence 的逻辑 block 到物理 block 映射 |
| `free_block_ids` | 当前空闲物理 block |
| `used_block_ids` | 当前被引用的物理 block |
| `ref_count` | prefix cache 或多序列共享时的引用计数 |
| `hash_to_block_id` | prefix cache 的 hash 索引 |
| `slot_mapping` | 本轮每个输入 token 应该写入 KV cache 的物理 slot |

KV cache 生命周期：

```text
ModelRunner.allocate_kv_cache()
  -> 给每一层 Attention 绑定 k_cache/v_cache

Scheduler.schedule()
  -> BlockManager.can_allocate()
  -> BlockManager.allocate()
  -> seq.block_table 填入物理 block id

ModelRunner.prepare_prefill()/prepare_decode()
  -> 生成 slot_mapping 和 block_tables

Attention.forward()
  -> store_kvcache()
  -> flash attention 读取 k_cache/v_cache

Scheduler.postprocess()
  -> BlockManager.hash_blocks()
  -> seq.append_token()
  -> 完成时 BlockManager.deallocate()
```

任务：

- [ ] 手工构造两个有相同长前缀的 prompt，观察 `num_cached_tokens`、`block_table` 和 `ref_count`。
- [ ] 解释为什么 `can_allocate()` 只检查到 `seq.num_blocks - 1`，即不缓存最后一个未满 block。
- [ ] 解释 `hash_blocks()` 什么时候给 block 写入 hash。
- [ ] 解释 `can_append()` 中 `len(seq) % block_size == 1` 的意义：decode 时当前 `last_token` 可能刚好落在一个新 block 的第一个 slot。
- [ ] 画出一个 prompt 长度超过 `kvcache_block_size` 时的 block table。

思考问题：

1. KV cache 是什么时候真正分配显存的？
2. 某个 sequence 的 KV block 是什么时候分配的？
3. prefix cache 命中后，为什么仍然需要调度未缓存的 token？
4. request 结束后，KV cache 是在哪里释放的？
5. preemption 为什么要 `deallocate()`，又为什么能通过 prefix cache 降低重复计算成本？

## 阶段 4：ModelRunner 数据准备

目标：理解 scheduler 选出来的是 Python 对象，模型真正吃的是张量。

阅读顺序：

1. `ModelRunner.run()`
2. `ModelRunner.prepare_prefill()`
3. `ModelRunner.prepare_decode()`
4. `ModelRunner.prepare_sample()`
5. `nanovllm/utils/context.py`

Prefill/decode 张量对照：

| 张量/字段 | Prefill | Decode |
| --- | --- | --- |
| `input_ids` | 每个 seq 本轮要处理的 token 区间 | 每个 seq 的 `last_token` |
| `positions` | `range(start, end)` | `len(seq) - 1` |
| `cu_seqlens_q` | 拼接 batch 后的 query 边界 | 不需要 |
| `cu_seqlens_k` | 包含 prefix cache 后的 key 边界 | 不需要 |
| `max_seqlen_q` | 本轮最大 query 长度 | 不需要 |
| `max_seqlen_k` | 本轮最大 key 长度 | 不需要 |
| `slot_mapping` | 本轮 token 写入 KV cache 的物理 slot | `last_token` 写入 KV cache 的物理 slot |
| `context_lens` | 不需要 | 每个 seq 当前总长度 |
| `block_tables` | prefix cache 命中时需要 | decode 总是需要 |

任务：

- [ ] 在 `prepare_prefill()` 打印 `input_ids` 长度、`cu_seqlens_q`、`cu_seqlens_k`、`slot_mapping`。
- [ ] 在 `prepare_decode()` 打印 batch size、`positions`、`context_lens`、`block_tables.shape`。
- [ ] 解释 `set_context()` 为什么用全局上下文，而不是把这些张量逐层传进 attention。
- [ ] 解释 `ParallelLMHead.forward()` 为什么 prefill 时只取每个 sequence 的最后一个 hidden state 计算 logits。

思考问题：

1. prefill 阶段输入的是哪些 token？
2. decode 阶段每次输入几个 token？
3. prefix cache 命中时，`cu_seqlens_k[-1]` 为什么可能大于 `cu_seqlens_q[-1]`？
4. `slot_mapping` 和 `block_tables` 分别解决什么问题？

## 阶段 5：模型 forward 和 attention

目标：理解 Qwen3 模型结构如何接入推理运行时，而不是只把它当黑盒。

阅读顺序：

1. `nanovllm/models/qwen3.py`
2. `nanovllm/layers/embed_head.py`
3. `nanovllm/layers/linear.py`
4. `nanovllm/layers/rotary_embedding.py`
5. `nanovllm/layers/layernorm.py`
6. `nanovllm/layers/attention.py`

单层 forward 主线：

```text
input_ids
  -> VocabParallelEmbedding
  -> Qwen3DecoderLayer x N
      -> RMSNorm
      -> QKVParallelLinear
      -> q/k/v reshape
      -> q_norm/k_norm
      -> RoPE
      -> Attention(store KV + flash attention)
      -> RowParallelLinear
      -> RMSNorm
      -> MLP(gate/up/down)
  -> final RMSNorm
  -> ParallelLMHead
```

任务：

- [ ] 画出 `Qwen3Attention.forward()` 中 q/k/v 的 shape 变化。
- [ ] 解释 tensor parallel 下 `QKVParallelLinear`、`RowParallelLinear`、`VocabParallelEmbedding` 分别切哪个维度。
- [ ] 解释 `Attention.forward()` 在 prefill 和 decode 下分别调用哪个 flash-attn API。
- [ ] 解释为什么 decode 使用 `flash_attn_with_kvcache()`。

思考问题：

1. RoPE 的 `positions` 从哪里来？
2. 为什么 attention 需要同时拿到当前 token 的 q/k/v 和历史 KV cache？
3. tensor parallel 下 logits 为什么需要 gather？
4. 如果输出明显异常，应该检查 tokenizer、positions、RoPE、权重加载还是 sampler？分别怎么看？

## 阶段 6：Sampler 和输出回写

目标：理解模型 forward 后如何得到下一个 token，以及新 token 如何回到 sequence 状态。

阅读顺序：

1. `nanovllm/layers/sampler.py`
2. `ModelRunner.run()`
3. `Scheduler.postprocess()`
4. `LLMEngine.step()`
5. `LLMEngine.generate()`

主链路：

```text
logits
  -> Sampler(logits, temperatures)
  -> token_ids
  -> Scheduler.postprocess()
      -> hash_blocks()
      -> seq.num_cached_tokens += seq.num_scheduled_tokens
      -> seq.append_token(token_id)
      -> EOS/max_tokens 判断
      -> finished 后释放 KV block
  -> LLMEngine.generate() 收集完成的 completion_token_ids
```

任务：

- [ ] 解释 sampler 中 `probs / exponential_noise` 再 `argmax` 的采样含义。
- [ ] 对比 `temperature=0.6` 和 `temperature=1.5` 的输出差异。
- [ ] 记录 EOS 停止和 `max_tokens` 停止分别在哪一行生效。
- [ ] 设计一个小改造：支持 greedy sampling、top-k、top-p 或 stop token ids，任选一个。

思考问题：

1. sampler 是在模型 forward 之后做什么？
2. 为什么 prefill 完成后也会立刻采样并 append 一个 token？
3. `ignore_eos=True` 对 benchmark 有什么影响？
4. 如果输出比预期短或停不下来，应该检查哪些字段？

## 阶段 7：性能相关能力

目标：理解 nano-vllm 中和推理性能直接相关的实现。

重点主题：

1. KV cache 显存估算：`ModelRunner.allocate_kv_cache()`
2. Prefix caching：`BlockManager.compute_hash()`、`can_allocate()`、`hash_blocks()`
3. Tensor Parallel：`ModelRunner.__init__()`、`linear.py`、`embed_head.py`
4. CUDA Graph：`ModelRunner.capture_cudagraph()`、`run_model()`
5. 权重加载：`utils/loader.py`、`Qwen3ForCausalLM.packed_modules_mapping`

任务：

- [ ] 推导一个 KV block 的显存大小：`2 * layers * block_size * kv_heads * head_dim * dtype_size`。
- [ ] 解释 `gpu_memory_utilization` 如何影响 `num_kvcache_blocks`。
- [ ] 解释为什么 prefill 不走 CUDA Graph，而 decode 在条件满足时可以走。
- [ ] 解释 tensor parallel worker 进程如何通过 shared memory 收到 rank 0 的调用。
- [ ] 解释 packed q/k/v、gate/up 权重如何被加载到合并后的参数中。

思考问题：

1. 显存不够时，先调小哪个参数？为什么？
2. batch size 增大后，prefill 吞吐和 decode 吞吐会如何变化？
3. CUDA Graph 为什么要求较稳定的 shape？
4. Tensor Parallel 下哪些层需要 all-reduce，哪些层需要 gather？

## 阶段 8：调优、排障和改造练习

目标：从“读懂”进入“能改、能测、能解释现象”。

建议练习从简单到困难：

1. 增加 scheduler debug 日志，输出每轮是 prefill 还是 decode、调度了哪些 seq、每个 seq 的 token/block 状态。
2. 给 `Sequence` 和 `BlockManager` 写不依赖 GPU 的单元测试，覆盖 allocate、deallocate、prefix cache 命中、ref count。
3. 增加一种采样能力：greedy、top-k、top-p 或 stop token ids。
4. 增加一个小 benchmark case：短 prompt 长输出、长 prompt 短输出、共享前缀 prompt、随机 prompt 四组对比。
5. 尝试支持混合 prefill/decode batch，先写设计文档，再改 scheduler、context 和 attention 输入准备。
6. 尝试适配一个和 Qwen3 结构相近的模型，重点看 config 字段、权重名映射、attention head 配置和 tokenizer。

排障清单：

| 现象 | 优先检查 |
| --- | --- |
| 初始化 OOM | `gpu_memory_utilization`, `max_model_len`, `tensor_parallel_size`, 模型 dtype |
| 运行中 OOM 或频繁 preempt | `max_num_seqs`, `max_num_batched_tokens`, `num_kvcache_blocks`, prompt/output 长度 |
| flash-attn shape 报错 | `cu_seqlens_q/k`, `max_seqlen_q/k`, `block_tables`, `slot_mapping` |
| 输出乱码或质量异常 | tokenizer chat template、权重加载、RoPE positions、TP shard、sampler |
| 请求提前结束 | `eos`, `ignore_eos`, `max_tokens` |
| 请求无法结束 | `ignore_eos`, `max_tokens`, EOS token id 是否正确 |
| tensor parallel 卡住 | NCCL、rank 数、CUDA device、worker 进程、端口占用 |

## 必须画出的图

### 1. 全链路图

```text
Prompt
  -> Tokenizer
  -> Sequence
  -> Scheduler(waiting/running)
  -> scheduled batch
  -> ModelRunner.prepare_prefill()/prepare_decode()
  -> set_context()
  -> Qwen3ForCausalLM.forward()
  -> Attention + KV Cache
  -> ParallelLMHead.compute_logits()
  -> Sampler
  -> Output Token
  -> Scheduler.postprocess()
  -> Sequence State Update
  -> Tokenizer.decode()
```

### 2. Sequence 状态机

```text
WAITING
  -> prefill 完成
  -> RUNNING
  -> decode append token
  -> RUNNING
  -> EOS 或 max_tokens
  -> FINISHED

RUNNING
  -> KV block 不够，被 preempt
  -> WAITING
```

### 3. KV block 映射图

```text
Sequence token_ids:
  [token 0 ... token 255] [token 256 ... token 511] [token 512 ...]

Logical blocks:
  block 0                 block 1                   block 2

seq.block_table:
  [physical_block_7,      physical_block_2,         physical_block_9]

slot_mapping:
  current token index -> physical_block_id * block_size + offset_in_block
```

## 最终验收问题

完成学习后，尝试不用看代码回答这些问题：

1. 一个 prompt 输入后，内部会被包装成什么对象？关键字段有哪些？
2. nano-vllm 里的 `Sequence` 和 vLLM 里的 request/sequence 概念有什么相似点和差异？
3. 为什么 LLM 推理要分成 prefill 和 decode 两个阶段？
4. prefill 阶段输入的是哪些 token？prefix cache 命中后有什么变化？
5. decode 阶段每个 sequence 每轮输入几个 token？这个 token 是新采样出来的 token 还是上一次的 last token？
6. Scheduler 每一轮到底在调度什么？为什么当前不混合 prefill 和 decode？
7. KV cache 的物理显存什么时候分配？单个 sequence 的 block 什么时候分配？
8. `block_table`、`slot_mapping`、`context_lens` 分别解决什么问题？
9. 一个 request 结束后，KV cache 在哪里释放？prefix cache 的 hash 信息是否一定立刻消失？
10. attention 如何在 prefill、prefix cache、decode 三种场景中读取 KV？
11. sampler 在模型 forward 之后做什么？采样结果如何影响下一轮 decode？
12. 如果要提高吞吐，应该先观察哪些指标，再调哪些参数？
13. 如果要支持一种新采样策略，应该改哪些文件？如何验证？
14. 如果要支持一个新模型，应该检查 config、模型结构、权重映射和 tokenizer 的哪些地方？

## 学习记录模板

每次学习或改造后，按这个格式记录：

```text
日期：
主题：
阅读文件：
本次弄清楚的问题：
关键调用链：
关键字段/张量 shape：
实验命令：
实验现象：
还没理解的问题：
下一步：
```
