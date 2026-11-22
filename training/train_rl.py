"""RL：GRPO（带 Dr. GRPO 的修正），奖励可插拔——RLVR、agent RL、RLAIF 都用这一个脚本，区别只在题目文件。

    tinybrain rl --steps 200 --judge-url http://localhost:8000/v1    # 默认：从 out/small/sft 出发，data/rl_small.parquet，写到 out/small/rl

每一步：
  1. 抽 prompts_per_step 道题，每道采 group_size 个回答（agent 题按多轮跑工具）；
  2. 按每道题的 reward 字段打分（math / code / poetry / judge / agent，见 training/rewards）；
  3. 组内减去平均奖励得到优势值（Dr. GRPO：不除以标准差）；
  4. 损失 = −Σ 优势 × 模型自己写的词的对数概率 / (回答数 × max_new)，分母是常数（Dr. GRPO：不按回答长度归一化，免得偏爱长回答）。
     完全在线：每批回答只用来更新一次，所以不需要重要性采样比和截断。

题目数据（parquet，一行一道题）：
    messages   提示，[{role, content}]，不含要模型写的回答
    reward     math / code / poetry / judge / agent / choice / reading / cloze / instruct / logic
    answer     math、agent、choice、cloze 用的标准答案；reading 是参考答案的 JSON 列表
    tests      code 用的测试（Python assert）
    form       poetry 的体裁（五绝、七绝、五律、七律）
    domain     judge、poetry 用的评分领域（通用、写作、古文、诗词）
    reference  judge 可选的参考答案（比如鲁迅续写的原文）
    tools、documents   agent 题能用的工具和检索资料
    constraints  instruct 题的约束（JSON 列表，见 training/rewards/instruct.py）
    gym、entry   logic 题的题型名和生成时的那一条（reasoning-gym）
全部参数见文件末尾的 parse_args。
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from inference.template import ChatFormat
from inference.tokenizer import DEFAULT_MODEL, Tokenizer

from .rewards import choice as choice_reward
from .rewards import code as code_reward
from .rewards import instruct as instruct_reward
from .rewards import judge as judge_mod
from .rewards import logic as logic_reward
from .rewards import match as match_reward
from .rewards import math as math_reward
from .rewards import poetry as poetry_reward
from .rewards import tools as tool_reward
from .rewards.tools import ToolEnv
from .rollout import Episode, collate, keep_fitting, run_agent, sample_group, token_logprobs
from .trainer_utils import (add_optimizer_args, autocast, clip_gradients, find_stage, format_eta, load_model, optimizers_from_args, pick_device,
                            save_checkpoint, set_lr_scale, setup_device, step_optimizers, wandb_log, wandb_session)

# 程序判分的题：奖励函数 reward(episode, task) -> 0 到 1
PROGRAM_REWARDS = {"math": math_reward.reward, "code": code_reward.reward, "agent": tool_reward.reward, "choice": choice_reward.reward,
                   "reading": match_reward.reading_reward, "cloze": match_reward.cloze_reward, "instruct": instruct_reward.reward,
                   "logic": logic_reward.reward}
KINDS = tuple(PROGRAM_REWARDS) + ("poetry", "judge")  # 后两种要评审模型


def load_tasks(path: str) -> list[dict]:
    tasks = pq.read_table(path).to_pylist()
    for i, t in enumerate(tasks):
        if t.get("reward") not in KINDS:
            raise ValueError(f"第 {i} 道题的 reward 是 {t.get('reward')!r}，只能是 {'、'.join(KINDS)}")
        if not t.get("messages"):
            raise ValueError(f"第 {i} 道题没有 messages")
    return tasks


def last_user(messages: list[dict]) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


def score(tasks: list[dict], groups: list[list[Episode]], judge=None, meter_weight: float = 0.4) -> list[list[float]]:
    """给每个回答打分。judge 和 poetry 的评审调用攒在一起并发发出去。"""
    rewards = [[0.0] * len(g) for g in groups]
    pending = []  # (第几题, 第几个回答, 问题, 回答, 领域, 参考)
    for ti, (task, group) in enumerate(zip(tasks, groups)):
        kind = task["reward"]
        for ei, ep in enumerate(group):
            if kind in PROGRAM_REWARDS:
                rewards[ti][ei] = PROGRAM_REWARDS[kind](ep, task)
            else:
                domain = task.get("domain") or ("诗词" if kind == "poetry" else "通用")
                pending.append((ti, ei, last_user(task["messages"]), ep.text, domain, task.get("reference")))
    if pending:
        if judge is None:
            raise ValueError("有 judge 或 poetry 题，但没有给评审模型（--judge-url）")
        scores = judge.score([p[2:] for p in pending])
        for (ti, ei, *_), s in zip(pending, scores):
            task, ep = tasks[ti], groups[ti][ei]
            if task["reward"] == "judge":
                rewards[ti][ei] = judge_mod.to_reward(s)
            else:  # 诗词：格律分（程序）和评审分加权；格律不过关时评审分最多算一半
                meter = poetry_reward.check(ep.text, task["form"])
                judged = 0.0 if s is None else (s - 1) / 9
                if not meter["shape"]:
                    judged *= 0.5
                rewards[ti][ei] = 2 * (meter_weight * meter["score"] + (1 - meter_weight) * judged) - 1
    return rewards


def advantages(rewards: list[float]) -> list[float]:
    """Dr. GRPO：组内减平均，不除以标准差。"""
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def policy_loss(model, episodes: list[Episode], advs: list[float], norm: float, micro_batch: int, device) -> float:
    """对一批回答做反向传播（梯度累积在模型参数上），返回损失的数值。norm 是 Dr. GRPO 的常数分母。"""
    total = 0.0
    for start in range(0, len(episodes), micro_batch):
        chunk, a = episodes[start:start + micro_batch], torch.tensor(advs[start:start + micro_batch], device=device)
        x, y, learn = collate(chunk, device)
        with autocast(device):
            logp = token_logprobs(model, x, y)
        loss = -(a[:, None] * logp.float() * learn).sum() / norm
        loss.backward()
        total += loss.item()
    return total


def train(cfg, judge=None) -> dict:
    """跑完 RL，返回最后一步的统计。cfg 是 parse_args 的结果；judge 可以直接传一个 Judge（测试时用假的），不传就按 --judge-url 建。"""
    device = pick_device(cfg.device)
    setup_device(device)
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(cfg.tasks)
    if judge is None and cfg.judge_url:
        judge = judge_mod.Judge(cfg.judge_url, cfg.judge_model, timeout=cfg.judge_timeout, workers=cfg.judge_workers)
    model = load_model(Path(cfg.init), device)
    fmt = ChatFormat(Tokenizer(DEFAULT_MODEL))
    tasks = keep_fitting(model, fmt, tasks, cfg.max_new, "RL 题目")
    optimizers = optimizers_from_args(model, cfg)
    (out / "config.json").write_text(json.dumps({"rl": vars(cfg), "model": json.loads(model.cfg.to_json())}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed) if device.type == "cpu" else None
    top_k = cfg.top_k or None
    print(f"设备 {device}；策略模型 {cfg.init}；{len(tasks)} 道题；每步 {cfg.prompts_per_step} 道 × {cfg.group_size} 个回答，共 {cfg.steps} 步", flush=True)
    stats, started = {}, time.perf_counter()
    with open(out / "log.jsonl", "a", encoding="utf-8") as log, \
            wandb_session(out, cfg.wandb_project, cfg.use_wandb, vars(cfg)) as wandb:
        for step in range(cfg.steps):
            t0 = time.perf_counter()
            batch = [tasks[i] for i in rng.choice(len(tasks), size=min(cfg.prompts_per_step, len(tasks)), replace=False)]
            groups = []
            for task in batch:
                if task["reward"] == "agent":
                    env = ToolEnv(task)
                    messages = [{"role": "system", "content": env.describe()}] + [m for m in task["messages"] if m["role"] != "system"]
                    groups.append([run_agent(model, fmt, messages, ToolEnv(task), cfg.max_new, cfg.max_turns, cfg.temperature, top_k, gen)
                                   for _ in range(cfg.group_size)])
                else:
                    groups.append(sample_group(model, fmt, task["messages"], cfg.group_size, cfg.max_new, cfg.temperature, top_k, gen))
            rollout_s = time.perf_counter() - t0
            rewards = score(batch, groups, judge, cfg.poetry_meter_weight)

            set_lr_scale(optimizers, cfg.lr_mult * min(1.0, (step + 1) / max(1, cfg.warmup_steps)))
            episodes, advs = [], []
            for group, rs in zip(groups, rewards):
                for ep, a in zip(group, advantages(rs)):
                    if a != 0:
                        episodes.append(ep)
                        advs.append(a)
            n_total = sum(len(g) for g in groups)
            loss = 0.0
            if episodes:
                model.train()
                loss = policy_loss(model, episodes, advs, norm=n_total * cfg.max_new, micro_batch=cfg.micro_batch, device=device)
                if not math.isfinite(loss):
                    raise RuntimeError(f"第 {step} 步的损失变成了 {loss}")
                grad_norm = clip_gradients(model.parameters(), cfg.clip_grad)
                step_optimizers(optimizers)
            flat = [r for rs in rewards for r in rs]
            by_kind = {}
            for task, rs in zip(batch, rewards):
                by_kind.setdefault(task["reward"], []).extend(rs)
            stats = dict(step=step + 1, reward=round(sum(flat) / len(flat), 4), loss=round(loss, 5),
                         by_kind={k: round(sum(v) / len(v), 4) for k, v in by_kind.items()},
                         learned_groups=sum(any(a != 0 for a in advantages(rs)) for rs in rewards),
                         response_tokens=round(sum(sum(e.learn) for g in groups for e in g) / n_total, 1),
                         finished=round(sum(e.finished for g in groups for e in g) / n_total, 3),
                         judge_failures=judge.failures if judge is not None else 0,
                         rollout_s=round(rollout_s, 1), step_s=round(time.perf_counter() - t0, 1))
            log.write(json.dumps(stats, ensure_ascii=False) + "\n")
            log.flush()
            wandb_log(wandb, stats, step + 1)  # 和 log.jsonl 完全相同的字段
            if (step + 1) % cfg.log_every == 0 or step + 1 == cfg.steps:
                kinds = "，".join(f"{k} {v:+.3f}" for k, v in stats["by_kind"].items())
                eta = format_eta((cfg.steps - step - 1) * (time.perf_counter() - started) / (step + 1))
                print(f"第 {step + 1:>5} 步  平均奖励 {stats['reward']:+.4f}（{kinds}）  回答平均 {stats['response_tokens']} 词  "
                      f"写完的 {stats['finished']:.0%}  采样 {stats['rollout_s']}s / 共 {stats['step_s']}s  剩余 {eta}", flush=True)
            if (step + 1) % cfg.save_every == 0 or step + 1 == cfg.steps:
                save_checkpoint(out / "ckpt.pt", model, optimizers, step + 1, rl_config=vars(cfg))
    return stats


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GRPO：RLVR、agent RL、RLAIF")
    g = p.add_argument_group("模型、题目和输出")
    g.add_argument("--run", default="out/small", help="运行目录，各个阶段都放在这里")
    g.add_argument("--init", default=None, help="策略模型从这个存档出发；默认 <运行目录>/sft/ckpt.pt")
    g.add_argument("--tasks", default="data/rl_small.parquet", help="题目数据（parquet），格式见文件开头")
    g.add_argument("--out", default=None, help="输出目录；默认 <运行目录>/rl")

    g = p.add_argument_group("采样")
    g.add_argument("--steps", type=int, default=100)
    g.add_argument("--prompts-per-step", type=int, default=8, help="每步抽几道题")
    g.add_argument("--group-size", type=int, default=8, help="每道题采几个回答（组内比较）")
    g.add_argument("--max-new", type=int, default=512, help="每个回答最多多少词；也是 Dr. GRPO 损失的常数分母")
    g.add_argument("--max-turns", type=int, default=4, help="agent 题最多几轮")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=0, help="0 表示不截断")
    g.add_argument("--micro-batch", type=int, default=16, help="一次前向放几个回答")
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("奖励")
    g.add_argument("--judge-url", default="", help="评审模型的 OpenAI 兼容接口；有 judge、poetry 题时要给")
    g.add_argument("--judge-model", default="Qwen/Qwen3.8-27B-FP8")
    g.add_argument("--judge-workers", type=int, default=16, help="同时发几个评审请求；vLLM 能并发，mlx_lm.server 一次只算一个，用 1–2")
    g.add_argument("--judge-timeout", type=float, default=120.0)
    g.add_argument("--poetry-meter-weight", type=float, default=0.4, help="诗词奖励里格律分占多少，其余是评审分")

    g = p.add_argument_group("训练")
    g.add_argument("--warmup-steps", type=int, default=5)
    add_optimizer_args(p, lr_mult=0.05)

    g = p.add_argument_group("记录、存档和硬件")
    g.add_argument("--log-every", type=int, default=1)
    g.add_argument("--save-every", type=int, default=50)
    g.add_argument("--use-wandb", action="store_true", help="把和 log.jsonl 相同的记录同步到 Weights & Biases")
    g.add_argument("--wandb-project", default="TinyBrain-RL", help="wandb 项目名")
    g.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    args.init = args.init or str(find_stage(args.run, ("sft",)))
    args.out = args.out or str(Path(args.run) / "rl")
    return args


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
