"""Generate training roadmap diagrams for rlvr-training-roadmap.md"""
import graphviz
import os
import subprocess
import sys

# Check system binary
try:
    subprocess.run(["dot", "-V"], capture_output=True, check=True)
except FileNotFoundError:
    print("ERROR: graphviz system binary not found.")
    sys.exit(1)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Morandi Palette ──
C = {
    "slate":      ("#D4D5D8", "#8B8E94"),
    "blue":       ("#D6E4EC", "#7B9EB2"),
    "green":      ("#D8E6DE", "#8FA89A"),
    "rose":       ("#EBD9D9", "#C4A0A0"),
    "mauve":      ("#DDD3E6", "#A08CB2"),
    "terracotta": ("#EBDDD0", "#C49A7C"),
    "sand":       ("#EDE6D6", "#C9B99A"),
}
MEDIUM = {
    "blue":       "#C5D8E8",
    "green":      "#C8DECE",
    "mauve":      "#CEC3DA",
    "terracotta": "#E0CFBF",
    "rose":       "#DFC9C9",
    "slate":      "#C4C5C8",
}
BG = "#F7F5F2"
TD = "#3D3D3D"
TM = "#6B6B6B"
EC = "#8B8E94"


def _base(name, title, rankdir="TB", nodesep="0.6", ranksep="0.8", **extra):
    g = graphviz.Digraph(name, format="png")
    g.attr(rankdir=rankdir, bgcolor=BG, fontname="Helvetica Neue", dpi="200",
           label=title, labelloc="t", fontsize="22", fontcolor=TD,
           pad="0.6", nodesep=nodesep, ranksep=ranksep, **extra)
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica Neue",
           fontsize="11", color=EC, fontcolor=TD)
    g.attr("edge", fontname="Helvetica Neue", fontsize="9", color=EC, fontcolor=TM)
    return g


def _save(g, name):
    path = os.path.join(OUTPUT_DIR, name)
    g.render(path, cleanup=True)
    print(f"  -> {name}.png")


# ═══════════════════════════════════════════════════════════════
# Diagram 1: End-to-End Training Roadmap (4 Phases)
# ═══════════════════════════════════════════════════════════════
def diagram_01_training_roadmap():
    g = _base("roadmap", "LLM Post-Training Roadmap: Math RLVR → Agentic RL",
              rankdir="TB", ranksep="1.0", nodesep="0.8")

    # ── Phase 0: Evaluation ──
    with g.subgraph(name="cluster_phase0") as c:
        c.attr(label="Phase 0: Capability Assessment", style="rounded",
               color=C["slate"][1], bgcolor="#ECEAE7",
               fontname="Helvetica Neue Bold", fontsize="13", fontcolor=TD)
        c.node("base", "Base / Instruct Model\n(Qwen3-4B / 8B)",
               fillcolor=C["slate"][0], color=C["slate"][1], shape="box")
        c.node("eval", "pass@k Evaluation\nGSM8K · MATH · AIME",
               fillcolor=MEDIUM["slate"], color=C["slate"][1])
        c.node("decision", "pass@64 > 50%?",
               fillcolor=C["sand"][0], color=C["sand"][1], shape="diamond",
               width="1.8", height="0.8")
        c.edge("base", "eval", label="generate\nk=1,4,16,64")
        c.edge("eval", "decision")

    # ── Phase 1: SFT ──
    with g.subgraph(name="cluster_phase1") as c:
        c.attr(label="Phase 1: Cold-Start SFT (Conditional)", style="dashed,rounded",
               color=C["green"][1], bgcolor="#ECF1ED",
               fontname="Helvetica Neue Bold", fontsize="13", fontcolor=TD)
        c.node("sft_data", "SFT Dataset\nOpenR1-Math-220k\n+ Skywork-OR1 (14K code)",
               fillcolor=C["green"][0], color=C["green"][1], shape="cylinder")
        c.node("sft_train", "SFT Training\n8-10 epochs · lr=1e-5\ncosine · max_seq=8192",
               fillcolor=MEDIUM["green"], color=C["green"][1])
        c.node("sft_ckpt", "SFT Checkpoint\nGSM8K pass@1 > 70%\n<think> rate > 90%",
               fillcolor=C["green"][0], color=C["green"][1],
               style="rounded,filled,bold")
        c.edge("sft_data", "sft_train")
        c.edge("sft_train", "sft_ckpt")

    # ── Phase 2: RLVR ──
    with g.subgraph(name="cluster_phase2") as c:
        c.attr(label="Phase 2: Math RLVR (Core Stage)", style="rounded",
               color=C["blue"][1], bgcolor="#E8EEF3",
               fontname="Helvetica Neue Bold", fontsize="13", fontcolor=TD)
        c.node("rl_data", "RL Dataset\ndapo_math_17k\n(binary reward)",
               fillcolor=C["blue"][0], color=C["blue"][1], shape="cylinder")
        c.node("rl_train", "DAPO Training\n500-2000 steps\nn_samples=16 · lr=1e-6\nclip_low=0.2 · kl=0.0",
               fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.node("rl_ckpt", "RLVR Checkpoint\nreward ↑ · entropy stable\nMATH/AIME improved",
               fillcolor=C["blue"][0], color=C["blue"][1],
               style="rounded,filled,bold")
        c.edge("rl_data", "rl_train")
        c.edge("rl_train", "rl_ckpt")

    # ── Phase 3: Agentic RL ──
    with g.subgraph(name="cluster_phase3") as c:
        c.attr(label="Phase 3: Agentic RL (Extension)", style="rounded",
               color=C["mauve"][1], bgcolor="#EDE8F2",
               fontname="Helvetica Neue Bold", fontsize="13", fontcolor=TD)
        c.node("code_dapo", "Code DAPO\nMulti-turn code exec\nmax_turns=10",
               fillcolor=MEDIUM["mauve"], color=C["mauve"][1])
        c.node("search_r1", "Search R1\nMulti-turn retrieval\nmax_tool_uses=2",
               fillcolor=MEDIUM["mauve"], color=C["mauve"][1])
        c.node("agent_ckpt", "Agent Model\nTool use + Reasoning",
               fillcolor=C["mauve"][0], color=C["mauve"][1],
               style="rounded,filled,bold")
        c.edge("code_dapo", "agent_ckpt")
        c.edge("search_r1", "agent_ckpt")

    # ── Cross-phase edges ──
    g.edge("decision", "sft_data", label="No (need SFT)",
           style="dashed", color=C["green"][1], fontcolor=C["green"][1])
    g.edge("decision", "rl_data", label="Yes (skip SFT)\nor use instruct model",
           color=C["blue"][1], fontcolor=C["blue"][1])
    g.edge("sft_ckpt", "rl_data", label="checkpoint →",
           style="bold", color=C["terracotta"][1], fontcolor=C["terracotta"][1])
    g.edge("rl_ckpt", "code_dapo", label="checkpoint →",
           style="bold", color=C["terracotta"][1], fontcolor=C["terracotta"][1])
    g.edge("rl_ckpt", "search_r1", label="checkpoint →",
           style="bold", color=C["terracotta"][1], fontcolor=C["terracotta"][1])

    # ── Timeline annotation ──
    g.node("t0", "Day 1-2", shape="plaintext", fontsize="10", fontcolor=TM)
    g.node("t1", "Day 3-5\n(if needed)", shape="plaintext", fontsize="10", fontcolor=TM)
    g.node("t2", "Day 3-7", shape="plaintext", fontsize="10", fontcolor=TM)
    g.node("t3", "Day 7+", shape="plaintext", fontsize="10", fontcolor=TM)
    g.edge("t0", "t1", style="invis")
    g.edge("t1", "t2", style="invis")
    g.edge("t2", "t3", style="invis")
    g.edge("t0", "eval", style="dotted", arrowhead="none", constraint="false")
    g.edge("t1", "sft_train", style="dotted", arrowhead="none", constraint="false")
    g.edge("t2", "rl_train", style="dotted", arrowhead="none", constraint="false")
    g.edge("t3", "code_dapo", style="dotted", arrowhead="none", constraint="false")

    _save(g, "01_training_roadmap")


# ═══════════════════════════════════════════════════════════════
# Diagram 2: AReaL Math RLVR Architecture
# ═══════════════════════════════════════════════════════════════
def diagram_02_rlvr_architecture():
    g = _base("rlvr_arch", "AReaL Math RLVR Training Architecture",
              rankdir="TB", ranksep="0.9", nodesep="0.5")

    # ── Data Layer ──
    with g.subgraph(name="cluster_data") as c:
        c.attr(label="Data Layer", style="rounded",
               color=C["green"][1], bgcolor="#ECF1ED",
               fontname="Helvetica Neue Bold", fontsize="12", fontcolor=TD)
        c.node("dataset", "dapo_math_17k\n(parquet, 17K prompts)",
               fillcolor=C["green"][0], color=C["green"][1], shape="cylinder")
        c.node("dataloader", "StatefulDataLoader\nbatch=16 · shuffle",
               fillcolor=MEDIUM["green"], color=C["green"][1])
        c.edge("dataset", "dataloader")

    # ── Rollout Engine ──
    with g.subgraph(name="cluster_rollout") as c:
        c.attr(label="Rollout Engine (SGLang)", style="rounded",
               color=C["blue"][1], bgcolor="#E8EEF3",
               fontname="Helvetica Neue Bold", fontsize="12", fontcolor=TD)
        c.node("sglang", "SGLang Server\nDP4 × TP1 (4 GPU)\ncontext=16384",
               fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.node("gen", "Generation\nn_samples=4~16\ntemp=1.0 · top_p=0.99\nmax_tokens=8192",
               fillcolor=C["blue"][0], color=C["blue"][1])
        c.edge("sglang", "gen")

    # ── Workflow Layer ──
    with g.subgraph(name="cluster_workflow") as c:
        c.attr(label="Workflow Layer", style="rounded",
               color=C["mauve"][1], bgcolor="#EDE8F2",
               fontname="Helvetica Neue Bold", fontsize="12", fontcolor=TD)
        c.node("rlvr", "RLVRWorkflow\narun_episode()",
               fillcolor=MEDIUM["mauve"], color=C["mauve"][1])
        c.node("reward", "math_reward_fn\nmath_verify · binary\ncorrect=1 · wrong=0",
               fillcolor=C["mauve"][0], color=C["mauve"][1])
        c.edge("rlvr", "reward", label="decode →\nverify")

    # ── Training Engine ──
    with g.subgraph(name="cluster_train") as c:
        c.attr(label="Training Engine (Megatron)", style="rounded",
               color=C["rose"][1], bgcolor="#F3EBEB",
               fontname="Helvetica Neue Bold", fontsize="12", fontcolor=TD)
        c.node("actor", "Actor (Megatron)\nDP4 × TP1 (4 GPU)\nGRPO / DAPO",
               fillcolor=MEDIUM["rose"], color=C["rose"][1])
        c.node("ref", "Reference Model\n(colocated)\nfrozen weights",
               fillcolor=C["rose"][0], color=C["rose"][1])
        c.node("optim", "Optimizer\nAdam · lr=1e-6\ncosine · grad_clip=0.1",
               fillcolor=C["rose"][0], color=C["rose"][1])
        c.edge("actor", "ref", label="KL div\n(if kl_ctl>0)", style="dashed")
        c.edge("actor", "optim", label="backward\n+ step")

    # ── Tracking ──
    with g.subgraph(name="cluster_track") as c:
        c.attr(label="Monitoring", style="rounded",
               color=C["sand"][1], bgcolor="#F5F1E8",
               fontname="Helvetica Neue Bold", fontsize="12", fontcolor=TD)
        c.node("swanlab", "SwanLab\nreward · entropy\nlength · clip_frac",
               fillcolor=C["sand"][0], color=C["sand"][1])
        c.node("saver", "Checkpoint Saver\nfreq_steps=10000",
               fillcolor=C["sand"][0], color=C["sand"][1])

    # ── Main flow edges ──
    g.edge("dataloader", "rlvr", label="batch of prompts",
           style="bold", color=C["terracotta"][1], fontcolor=C["terracotta"][1])
    g.edge("rlvr", "sglang", label="generate(prompt)",
           style="bold", color=C["terracotta"][1], fontcolor=C["terracotta"][1])
    g.edge("gen", "reward", label="responses",
           style="bold", color=C["terracotta"][1], fontcolor=C["terracotta"][1])
    g.edge("reward", "actor", label="trajectories\n+ rewards",
           style="bold", color=C["terracotta"][1], fontcolor=C["terracotta"][1])
    g.edge("optim", "sglang", label="Weight Sync\n(NCCL xccl)",
           style="bold", color=C["terracotta"][1], fontcolor=C["terracotta"][1],
           dir="forward")
    g.edge("actor", "swanlab", style="dashed", constraint="false")
    g.edge("actor", "saver", style="dashed", constraint="false")

    _save(g, "02_rlvr_architecture")


# ═══════════════════════════════════════════════════════════════
# Diagram 3: DAPO vs GRPO Algorithm Comparison
# ═══════════════════════════════════════════════════════════════
def diagram_03_dapo_vs_grpo():
    g = _base("algo_cmp", "DAPO vs Standard GRPO: 4 Key Improvements",
              rankdir="TB", ranksep="0.7", nodesep="0.5")

    # ── GRPO (problems) ──
    with g.subgraph(name="cluster_grpo") as c:
        c.attr(label="Standard GRPO (Problems)", style="rounded",
               color=C["rose"][1], bgcolor="#F3EBEB",
               fontname="Helvetica Neue Bold", fontsize="13", fontcolor=TD)
        c.node("g1", "Single Clip\neps=0.2 (symmetric)\n→ Limits exploration",
               fillcolor=MEDIUM["rose"], color=C["rose"][1])
        c.node("g2", "Static Sampling\nAll samples used\n→ Wasted compute on trivial",
               fillcolor=MEDIUM["rose"], color=C["rose"][1])
        c.node("g3", "Sequence-Level Loss\nNormalize per sequence\n→ Length bias",
               fillcolor=MEDIUM["rose"], color=C["rose"][1])
        c.node("g4", "No Truncation Handling\nTruncated = normal\n→ Noisy gradients",
               fillcolor=MEDIUM["rose"], color=C["rose"][1])
        c.body.append('{rank=same; g1; g2; g3; g4;}')
        for i in range(1, 4):
            c.edge(f"g{i}", f"g{i+1}", style="invis")

    # ── Arrows ──
    for i in range(1, 5):
        g.edge(f"g{i}", f"d{i}", label="fix", color=C["terracotta"][1],
               fontcolor=C["terracotta"][1], style="bold")

    # ── DAPO (solutions) ──
    with g.subgraph(name="cluster_dapo") as c:
        c.attr(label="DAPO (Solutions) — arxiv:2503.14476", style="rounded",
               color=C["blue"][1], bgcolor="#E8EEF3",
               fontname="Helvetica Neue Bold", fontsize="13", fontcolor=TD)
        c.node("d1", "Clip-Higher\nclip_low=0.2 · clip_high=0.28\n→ Allows more exploration",
               fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.node("d2", "Dynamic Sampling\nFilter reward std=0 groups\n→ Only train on informative",
               fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.node("d3", "Token-Level Loss\nNormalize across all tokens\n→ Fair gradient per token",
               fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.node("d4", "Overlong Mask\nExclude truncated from loss\n→ Clean gradients",
               fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.body.append('{rank=same; d1; d2; d3; d4;}')
        for i in range(1, 4):
            c.edge(f"d{i}", f"d{i+1}", style="invis")

    # ── Result ──
    g.node("result",
           "Result: AIME24 50.0 (32B) in 50% fewer steps than R1-Zero\n"
           "POLARIS: AIME24 81.2% (4B) in just 700 steps",
           fillcolor=C["green"][0], color=C["green"][1],
           shape="box", fontsize="11")
    g.edge("d1", "result", style="invis")
    g.edge("d4", "result", style="invis")
    g.edge("d2", "result", style="bold", color=C["green"][1])

    _save(g, "03_dapo_vs_grpo")


# ═══════════════════════════════════════════════════════════════
# Diagram 4: Decision Tree — Do I need SFT?
# ═══════════════════════════════════════════════════════════════
def diagram_04_decision_tree():
    g = _base("decision", "Decision Tree: Do You Need SFT Before RL?",
              rankdir="TB", ranksep="0.9", nodesep="0.6")

    # Start
    g.node("start", "Start:\nChoose Base or Instruct Model",
           fillcolor=C["slate"][0], color=C["slate"][1], shape="oval")

    # Decision 1: model type
    g.node("q1", "Model type?",
           fillcolor=C["sand"][0], color=C["sand"][1], shape="diamond",
           width="2.0", height="0.9")

    # Instruct branch
    g.node("q2", "pass@1 on GSM8K?",
           fillcolor=C["sand"][0], color=C["sand"][1], shape="diamond",
           width="2.0", height="0.9")
    g.node("q3", "<think> format\nstable?",
           fillcolor=C["sand"][0], color=C["sand"][1], shape="diamond",
           width="2.0", height="0.9")

    # Base branch
    g.node("q4", "pass@64 on GSM8K?",
           fillcolor=C["sand"][0], color=C["sand"][1], shape="diamond",
           width="2.0", height="0.9")

    # Terminal nodes
    g.node("go_rl", "→ Skip SFT\nDirect RLVR",
           fillcolor=C["green"][0], color=C["green"][1],
           shape="box", style="rounded,filled,bold")
    g.node("do_sft", "→ Do SFT First\nOpenR1-Math-220k\n8-10 epochs",
           fillcolor=C["blue"][0], color=C["blue"][1],
           shape="box", style="rounded,filled,bold")
    g.node("swap", "→ Switch Model\nToo weak for RL\nUse larger model",
           fillcolor=C["rose"][0], color=C["rose"][1],
           shape="box", style="rounded,filled,bold")

    # Edges
    g.edge("start", "q1")
    g.edge("q1", "q2", label="Instruct", color=C["blue"][1], fontcolor=C["blue"][1])
    g.edge("q1", "q4", label="Base", color=C["mauve"][1], fontcolor=C["mauve"][1])

    g.edge("q2", "q3", label="> 70%")
    g.edge("q2", "do_sft", label="< 70%\n(weak instruct)")

    g.edge("q3", "go_rl", label="Yes\n(stable)", color=C["green"][1], fontcolor=C["green"][1])
    g.edge("q3", "do_sft", label="No\n(format issues)")

    g.edge("q4", "do_sft", label="> 50%\n(has potential)")
    g.edge("q4", "swap", label="< 30%\n(too weak)",
           color=C["rose"][1], fontcolor=C["rose"][1])
    g.node("q5", "30-50%:\nlight SFT\n(3-5 epochs)",
           fillcolor=C["sand"][0], color=C["sand"][1], shape="note", fontsize="10")
    g.edge("q4", "q5", label="30-50%", style="dashed")
    g.edge("q5", "do_sft", style="dashed", arrowhead="none")

    _save(g, "04_decision_tree")


# ═══════════════════════════════════════════════════════════════
# Diagram 5: Monitoring Dashboard — What to Watch
# ═══════════════════════════════════════════════════════════════
def diagram_05_monitoring():
    g = _base("monitor", "RLVR Training Monitoring: Key Metrics & Alerts",
              rankdir="LR", ranksep="0.8", nodesep="0.4")

    # ── Healthy signals ──
    with g.subgraph(name="cluster_healthy") as c:
        c.attr(label="Healthy Training Signals", style="rounded",
               color=C["green"][1], bgcolor="#ECF1ED",
               fontname="Helvetica Neue Bold", fontsize="12", fontcolor=TD)
        c.node("h1", "reward_mean\nSteadily rising\n→ Learning is working",
               fillcolor=MEDIUM["green"], color=C["green"][1])
        c.node("h2", "entropy\nSlow decline, stays > 0.5\n→ Still exploring",
               fillcolor=MEDIUM["green"], color=C["green"][1])
        c.node("h3", "clip_fraction\n0.1 - 0.3 range\n→ Stable updates",
               fillcolor=MEDIUM["green"], color=C["green"][1])
        c.node("h4", "response_length\nStable or slow growth\n→ No reward hacking",
               fillcolor=MEDIUM["green"], color=C["green"][1])
        c.edge("h1", "h2", style="invis")
        c.edge("h2", "h3", style="invis")
        c.edge("h3", "h4", style="invis")

    # ── Danger signals ──
    with g.subgraph(name="cluster_danger") as c:
        c.attr(label="Danger Signals & Fixes", style="rounded",
               color=C["rose"][1], bgcolor="#F3EBEB",
               fontname="Helvetica Neue Bold", fontsize="12", fontcolor=TD)
        c.node("d1", "reward flat\n→ Check reward fn\n→ Increase n_samples",
               fillcolor=MEDIUM["rose"], color=C["rose"][1])
        c.node("d2", "entropy → 0\n(COLLAPSE)\n→ Use Clip-Higher\n→ Lower lr",
               fillcolor=MEDIUM["rose"], color=C["rose"][1])
        c.node("d3", "clip_frac > 0.5\n→ Lower lr\n→ Increase grad_clip",
               fillcolor=MEDIUM["rose"], color=C["rose"][1])
        c.node("d4", "length explosion\n→ Add length penalty\n→ Increase max_tokens",
               fillcolor=MEDIUM["rose"], color=C["rose"][1])
        c.edge("d1", "d2", style="invis")
        c.edge("d2", "d3", style="invis")
        c.edge("d3", "d4", style="invis")

    # ── Links ──
    g.edge("h1", "d1", label="if flat 100+ steps",
           style="dashed", color=C["terracotta"][1], fontcolor=C["terracotta"][1])
    g.edge("h2", "d2", label="if drops to ~0",
           style="bold", color=C["rose"][1], fontcolor=C["rose"][1])
    g.edge("h3", "d3", label="if > 0.5",
           style="dashed", color=C["terracotta"][1], fontcolor=C["terracotta"][1])
    g.edge("h4", "d4", label="if > 80% max_tokens",
           style="dashed", color=C["terracotta"][1], fontcolor=C["terracotta"][1])

    _save(g, "05_monitoring")


# ═══════════════════════════════════════════════════════════════
# Diagram 6: Hyperparameter Tuning Priority
# ═══════════════════════════════════════════════════════════════
def diagram_06_hyperparam_priority():
    g = _base("hyperparam", "Hyperparameter Tuning Priority (When Training Fails)",
              rankdir="TB", ranksep="0.7", nodesep="0.5")

    # Priority 1
    with g.subgraph(name="cluster_p1") as c:
        c.attr(label="Priority 1: Biggest Impact", style="rounded",
               color=C["rose"][1], bgcolor="#F3EBEB",
               fontname="Helvetica Neue Bold", fontsize="12", fontcolor=TD)
        c.node("p1a", "n_samples\n4 → 8 → 16\nMore diverse samples",
               fillcolor=MEDIUM["rose"], color=C["rose"][1])
        c.node("p1b", "gradient_clipping\n1.0 → 0.1\nStabilize training",
               fillcolor=MEDIUM["rose"], color=C["rose"][1])
        c.node("p1c", "weight_decay\n0 → 0.1\nPrevent extreme params",
               fillcolor=MEDIUM["rose"], color=C["rose"][1])
        c.body.append('{rank=same; p1a; p1b; p1c;}')
        c.edge("p1a", "p1b", style="invis")
        c.edge("p1b", "p1c", style="invis")

    # Priority 2
    with g.subgraph(name="cluster_p2") as c:
        c.attr(label="Priority 2: Fine Tuning", style="rounded",
               color=C["blue"][1], bgcolor="#E8EEF3",
               fontname="Helvetica Neue Bold", fontsize="12", fontcolor=TD)
        c.node("p2a", "lr\n1e-6 → 5e-7\nMore conservative",
               fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.node("p2b", "beta2\n0.999 → 0.99\nAdapt to RL noise",
               fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.node("p2c", "imp_weight_cap\n5.0 → 2.0\nStricter clipping",
               fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.node("p2d", "reward_scaling\n10.0 → 1.0\nRaw reward signal",
               fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.body.append('{rank=same; p2a; p2b; p2c; p2d;}')
        c.edge("p2a", "p2b", style="invis")
        c.edge("p2b", "p2c", style="invis")
        c.edge("p2c", "p2d", style="invis")

    # Priority 3
    with g.subgraph(name="cluster_p3") as c:
        c.attr(label="Priority 3: Architecture Changes", style="rounded",
               color=C["mauve"][1], bgcolor="#EDE8F2",
               fontname="Helvetica Neue Bold", fontsize="12", fontcolor=TD)
        c.node("p3a", "max_new_tokens\n8192 → 16384\nLonger reasoning",
               fillcolor=MEDIUM["mauve"], color=C["mauve"][1])
        c.node("p3b", "offpolicyness\n2 → 1\nMore on-policy",
               fillcolor=MEDIUM["mauve"], color=C["mauve"][1])
        c.body.append('{rank=same; p3a; p3b;}')
        c.edge("p3a", "p3b", style="invis")

    # Flow
    g.edge("p1a", "p2a", label="if still unstable",
           style="bold", color=C["terracotta"][1], fontcolor=C["terracotta"][1])
    g.edge("p2a", "p3a", label="if still not improving",
           style="bold", color=C["terracotta"][1], fontcolor=C["terracotta"][1])

    _save(g, "06_hyperparam_priority")


if __name__ == "__main__":
    print("Generating training roadmap diagrams...")
    diagram_01_training_roadmap()
    diagram_02_rlvr_architecture()
    diagram_03_dapo_vs_grpo()
    diagram_04_decision_tree()
    diagram_05_monitoring()
    diagram_06_hyperparam_priority()
    print("Done! All diagrams saved to docs/images/")
