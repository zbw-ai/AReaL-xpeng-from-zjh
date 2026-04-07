"""Generate performance observability diagrams for docs/perf-observability/README.md"""

import os
import sys

try:
    import graphviz
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "graphviz"], check=True)
    import graphviz

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
    "sand":       "#E0D9C8",
}
BG = "#F7F5F2"
TD = "#3D3D3D"
TM = "#6B6B6B"
EC = "#8B8E94"


def _base(name, title, rankdir="TB", nodesep="0.6", ranksep="0.8", **extra):
    g = graphviz.Digraph(name, format="png")
    g.attr(rankdir=rankdir, bgcolor=BG, fontname="Helvetica", dpi="200",
           label=title, labelloc="t", fontsize="20", fontcolor=TD,
           pad="0.5", nodesep=nodesep, ranksep=ranksep, **extra)
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
           fontsize="11", color=EC, fontcolor=TD)
    g.attr("edge", fontname="Helvetica", fontsize="9", color=EC, fontcolor=TM)
    return g


def _save(g, name):
    path = os.path.join(OUTPUT_DIR, name)
    g.render(path, cleanup=True)
    print(f"  -> {name}.png")


# ─────────────────────────────────────────────────────────────
# Diagram 1: Training Step Pipeline
# ─────────────────────────────────────────────────────────────
def create_training_step_pipeline():
    g = _base("train_step", "Training Step Pipeline (rl_trainer.py:338-542)",
              rankdir="TB", nodesep="0.5", ranksep="0.6")

    NODE = dict(width="2.2", height="0.55", fixedsize="true", fontsize="10")

    # ── Rollout cluster (COMPUTE) ──
    with g.subgraph(name="cluster_rollout") as c:
        c.attr(label="Phase 1: Data Collection", style="rounded",
               color=C["blue"][1], bgcolor="#E8EEF3",
               fontname="Helvetica", fontsize="12", fontcolor=TD)
        c.node("rollout", "1. Rollout\n(generate + reward)\ntimeperf/rollout",
               fillcolor=MEDIUM["blue"], color=C["blue"][1], **NODE)

    # ── Forward passes cluster (COMPUTE) ──
    with g.subgraph(name="cluster_fwd") as c:
        c.attr(label="Phase 2: Forward Passes (optional)", style="rounded",
               color=C["mauve"][1], bgcolor="#EDE8F2",
               fontname="Helvetica", fontsize="12", fontcolor=TD)
        c.node("critic", "2. Critic Values\n(if PPO w/ critic)\ntimeperf/critic_values",
               fillcolor=MEDIUM["mauve"], color=C["mauve"][1], **NODE)
        c.node("recompute", "3. Recompute LogP\n(if decoupled loss)\ntimeperf/recompute_logp",
               fillcolor=MEDIUM["mauve"], color=C["mauve"][1], **NODE)
        c.node("ref", "4. Ref LogP\n(if KL > 0)\ntimeperf/ref_logp",
               fillcolor=MEDIUM["mauve"], color=C["mauve"][1], **NODE)
        c.node("teacher", "5. Teacher LogP\n(if distillation)\ntimeperf/teacher_logp",
               fillcolor=MEDIUM["mauve"], color=C["mauve"][1], **NODE)
        c.body.append("{rank=same; critic; recompute;}")
        c.body.append("{rank=same; ref; teacher;}")
        c.edge("critic", "recompute", style="invis")
        c.edge("ref", "teacher", style="invis")

    # ── Advantage + PPO update (COMPUTE) ──
    with g.subgraph(name="cluster_train") as c:
        c.attr(label="Phase 3: Training Update", style="rounded",
               color=C["rose"][1], bgcolor="#F3E6E6",
               fontname="Helvetica", fontsize="12", fontcolor=TD)
        c.node("adv", "6. Compute Advantage\n(GAE / GRPO)\ntimeperf/compute_advantage",
               fillcolor=MEDIUM["rose"], color=C["rose"][1], **NODE)
        c.node("ppo", "7. PPO Update\n(fwd + bwd + optim)\ntimeperf/train_step",
               fillcolor=MEDIUM["rose"], color=C["rose"][1],
               width="2.2", height="0.55", fixedsize="true", fontsize="10",
               penwidth="2")

    # ── Communication (COMM) ──
    with g.subgraph(name="cluster_comm") as c:
        c.attr(label="Phase 4: Weight Sync", style="rounded",
               color=C["terracotta"][1], bgcolor="#F3EBE2",
               fontname="Helvetica", fontsize="12", fontcolor=TD)
        c.node("pause", "Pause Rollout",
               fillcolor=MEDIUM["terracotta"], color=C["terracotta"][1],
               width="1.5", height="0.4", fixedsize="true", fontsize="9")
        c.node("wsync", "8. Update Weights\n(NCCL / Disk)\ntimeperf/update_weights",
               fillcolor=MEDIUM["terracotta"], color=C["terracotta"][1], **NODE)

    # ── IO cluster ──
    with g.subgraph(name="cluster_io") as c:
        c.attr(label="Phase 5: Checkpointing & IO", style="rounded",
               color=C["green"][1], bgcolor="#E5EDE8",
               fontname="Helvetica", fontsize="12", fontcolor=TD)
        c.node("save", "9. Save HF Ckpt\ntimeperf/save",
               fillcolor=MEDIUM["green"], color=C["green"][1],
               width="1.8", height="0.45", fixedsize="true", fontsize="10")
        c.node("recover", "10. Recovery DCP\ntimeperf/checkpoint",
               fillcolor=MEDIUM["green"], color=C["green"][1],
               width="1.8", height="0.45", fixedsize="true", fontsize="10")
        c.body.append("{rank=same; save; recover;}")
        c.edge("save", "recover", style="invis")

    # ── Eval + cleanup (INSTR) ──
    with g.subgraph(name="cluster_end") as c:
        c.attr(label="Phase 6: Eval & Cleanup", style="rounded",
               color=C["slate"][1], bgcolor="#E8E8E9",
               fontname="Helvetica", fontsize="12", fontcolor=TD)
        c.node("eval", "11. Evaluation\ntimeperf/eval",
               fillcolor=MEDIUM["slate"], color=C["slate"][1],
               width="1.8", height="0.45", fixedsize="true", fontsize="10")
        c.node("cleanup", "12. Clear + Log\n+ Resume Rollout",
               fillcolor=MEDIUM["slate"], color=C["slate"][1],
               width="1.8", height="0.45", fixedsize="true", fontsize="10")
        c.body.append("{rank=same; eval; cleanup;}")
        c.edge("eval", "cleanup", style="invis")

    # ── Edges ──
    g.edge("rollout", "critic", label="batch")
    g.edge("critic", "ref")
    g.edge("recompute", "teacher")
    g.edge("ref", "adv", label="logps")
    g.edge("teacher", "adv")
    g.edge("adv", "ppo", label="advantages")
    g.edge("ppo", "pause")
    g.edge("pause", "wsync", label="NCCL broadcast\nor disk write",
           style="bold", color=C["terracotta"][1], fontcolor=C["terracotta"][1])
    g.edge("wsync", "save")
    g.edge("save", "eval")
    g.edge("recover", "cleanup")

    # ── Annotation: typical time distribution ──
    g.node("note_time",
           "Typical Time Distribution:\n"
           "Rollout: 40-70%  |  PPO Update: 15-35%\n"
           "Weight Sync: 5-15%  |  IO: 0-10%",
           shape="note", fillcolor=C["sand"][0], color=C["sand"][1],
           fontname="Courier", fontsize="9")
    g.edge("cleanup", "note_time", style="dotted", arrowhead="none")

    _save(g, "perf_01_training_step_pipeline")


# ─────────────────────────────────────────────────────────────
# Diagram 2: Rollout Internal Architecture
# ─────────────────────────────────────────────────────────────
def create_rollout_architecture():
    g = _base("rollout_arch", "Rollout Internal Architecture",
              rankdir="TB", nodesep="0.5", ranksep="0.7")

    # ── Data source ──
    g.node("dataloader", "DataLoader\n(cycle_dataloader)", shape="cylinder",
           fillcolor=C["green"][0], color=C["green"][1])

    # ── Executor ──
    with g.subgraph(name="cluster_executor") as c:
        c.attr(label="WorkflowExecutor (async)", style="dashed,rounded",
               color=C["mauve"][1], bgcolor=C["mauve"][0],
               fontname="Helvetica", fontsize="12", fontcolor=TD)

        c.node("dispatcher", "BatchTaskDispatcher\n(producer/consumer threads)",
               fillcolor=MEDIUM["mauve"], color=C["mauve"][1])
        c.node("runner", "AsyncTaskRunner\n(uvloop event loop)",
               fillcolor=MEDIUM["mauve"], color=C["mauve"][1])
        c.node("staleness", "StalenessManager\n(capacity gating)",
               fillcolor=MEDIUM["terracotta"], color=C["terracotta"][1])

    # ── Concurrent sessions ──
    with g.subgraph(name="cluster_sessions") as c:
        c.attr(label="Concurrent Sessions (N)", style="rounded",
               color=C["blue"][1], bgcolor="#E8EEF3",
               fontname="Helvetica", fontsize="12", fontcolor=TD)

        for i in range(3):
            sid = f"s{i}"
            label = f"Session {i+1}" if i < 2 else "Session N"
            c.node(sid, label, fillcolor=MEDIUM["blue"], color=C["blue"][1],
                   width="1.3", height="0.4", fixedsize="true", fontsize="9")

            c.node(f"gen{i}", "generate\n(HTTP->vLLM/SGLang)",
                   fillcolor=C["blue"][0], color=C["blue"][1],
                   width="1.8", height="0.4", fixedsize="true", fontsize="9")
            c.node(f"rw{i}", "reward\n(async callable)",
                   fillcolor=C["rose"][0], color=C["rose"][1],
                   width="1.8", height="0.4", fixedsize="true", fontsize="9")

        c.body.append("{rank=same; s0; s1; s2;}")
        c.body.append("{rank=same; gen0; gen1; gen2;}")
        c.body.append("{rank=same; rw0; rw1; rw2;}")

    # ── Collection ──
    g.node("accept", "Accept / Reject\n(staleness check)", shape="diamond",
           fillcolor=C["terracotta"][0], color=C["terracotta"][1], fontsize="10")

    g.node("dist_coord", "DistRolloutCoordinator\nall_gather + ffd_allocate\n+ broadcast (2x barrier)",
           fillcolor=C["green"][0], color=C["green"][1])

    g.node("batch_out", "rollout_batch\n(to training)", shape="oval",
           fillcolor=C["slate"][0], color=C["slate"][1])

    # ── Edges ──
    g.edge("dataloader", "dispatcher", label="prompts")
    g.edge("dispatcher", "runner")
    g.edge("runner", "staleness", label="check\ncapacity", style="dashed")
    g.edge("staleness", "runner", label="allowed", style="dashed")

    for i in range(3):
        g.edge("runner", f"s{i}", style="dashed" if i == 2 else "")
        g.edge(f"s{i}", f"gen{i}")
        g.edge(f"gen{i}", f"rw{i}")
        g.edge(f"rw{i}", "accept")

    g.edge("accept", "dist_coord", label="accepted\ntrajectories",
           style="bold", color=C["green"][1], fontcolor=C["green"][1])
    g.edge("dist_coord", "batch_out",
           style="bold", color=C["green"][1], fontcolor=C["green"][1])

    # rejected annotation
    g.node("rejected", "Rejected\n(off-policy)", shape="plaintext",
           fontcolor=C["rose"][1], fontsize="9")
    g.edge("accept", "rejected", label="staleness\nexceeded",
           style="dashed", color=C["rose"][1], fontcolor=C["rose"][1])

    _save(g, "perf_02_rollout_architecture")


# ─────────────────────────────────────────────────────────────
# Diagram 3: Performance Analysis Decision Tree
# ─────────────────────────────────────────────────────────────
def create_analysis_decision_tree():
    g = _base("decision_tree", "Performance Analysis Decision Tree",
              rankdir="TB", nodesep="0.4", ranksep="0.7")

    # ── Entry ──
    g.node("start", "Check wandb\ntimeperf/* panel", shape="oval",
           fillcolor=C["slate"][0], color=C["slate"][1], fontsize="11")

    # ── Decision nodes ──
    DEC = dict(shape="diamond", fontsize="10", width="2.5", height="0.9")
    g.node("d1", "timeperf/rollout\n> 50% of step?",
           fillcolor=C["blue"][0], color=C["blue"][1], **DEC)
    g.node("d2", "timeperf/train_step\n> 40% of step?",
           fillcolor=C["rose"][0], color=C["rose"][1], **DEC)
    g.node("d3", "timeperf/update_weights\n> 15% of step?",
           fillcolor=C["terracotta"][0], color=C["terracotta"][1], **DEC)

    # ── Action nodes ──
    ACT = dict(fontsize="10", width="2.8", height="0.7", fixedsize="true")

    g.node("a1", "Rollout Bottleneck\n1. Enable SessionTracer\n2. plot_session_trace\n3. Check reject ratio",
           fillcolor=MEDIUM["blue"], color=C["blue"][1], **ACT)

    g.node("a1_sub", "Generate slow?\n-> Tune TP, concurrency\n\nReward slow?\n-> Batch reward fn\n\nIdle gaps?\n-> Adjust max_staleness",
           fillcolor=C["blue"][0], color=C["blue"][1],
           shape="note", fontname="Courier", fontsize="8")

    g.node("a2", "Train Bottleneck\n1. Set profile_steps=[N]\n2. Open in Perfetto\n3. Find top CUDA ops",
           fillcolor=MEDIUM["rose"], color=C["rose"][1], **ACT)

    g.node("a2_sub", "Forward slow?\n-> Check FlashAttention\n\nBackward slow?\n-> Activation ckpt\n\nOptim slow?\n-> CPU offload overhead",
           fillcolor=C["rose"][0], color=C["rose"][1],
           shape="note", fontname="Courier", fontsize="8")

    g.node("a3", "Comm Bottleneck\n1. Check xccl vs disk mode\n2. NCCL_DEBUG=INFO\n3. Check IB / NVLink topo",
           fillcolor=MEDIUM["terracotta"], color=C["terracotta"][1], **ACT)

    g.node("a4", "IO / Scheduler\n1. Check save frequency\n2. Check RPC rpc.* scopes\n3. DistRollout barrier time",
           fillcolor=MEDIUM["green"], color=C["green"][1], **ACT)

    # ── Edges ──
    g.edge("start", "d1")
    g.edge("d1", "a1", label="YES", color=C["blue"][1], fontcolor=C["blue"][1], style="bold")
    g.edge("d1", "d2", label="NO")
    g.edge("d2", "a2", label="YES", color=C["rose"][1], fontcolor=C["rose"][1], style="bold")
    g.edge("d2", "d3", label="NO")
    g.edge("d3", "a3", label="YES", color=C["terracotta"][1], fontcolor=C["terracotta"][1], style="bold")
    g.edge("d3", "a4", label="NO", color=C["green"][1], fontcolor=C["green"][1])

    # annotation links
    g.edge("a1", "a1_sub", style="dotted", arrowhead="none", constraint="false")
    g.edge("a2", "a2_sub", style="dotted", arrowhead="none", constraint="false")

    _save(g, "perf_03_analysis_decision_tree")


# ─────────────────────────────────────────────────────────────
# Diagram 4: Observability Tool Layers
# ─────────────────────────────────────────────────────────────
def create_tool_layers():
    g = _base("tool_layers", "Observability Tool Stack (Coarse to Fine)",
              rankdir="TB", nodesep="0.5", ranksep="0.6")

    LAYER = dict(width="5.5", height="0.7", fixedsize="true", fontsize="11")

    # ── Layer 1: Step-level ──
    with g.subgraph(name="cluster_l1") as c:
        c.attr(label="Layer 1: Step-Level Aggregation (always on)",
               style="rounded", color=C["green"][1], bgcolor="#E5EDE8",
               fontname="Helvetica", fontsize="12", fontcolor=TD)
        c.node("l1", "DistributedStatsTracker\ntimeperf/* -> wandb / swanlab / TensorBoard\nGranularity: per step  |  Overhead: ~0",
               fillcolor=MEDIUM["green"], color=C["green"][1], **LAYER)

    # ── Layer 2: Timeline ──
    with g.subgraph(name="cluster_l2") as c:
        c.attr(label="Layer 2: Timeline Tracing (opt-in)",
               style="rounded", color=C["blue"][1], bgcolor="#E8EEF3",
               fontname="Helvetica", fontsize="12", fontcolor=TD)
        c.node("l2", "PerfTracer\nChrome Trace JSONL -> Perfetto UI\nGranularity: per scope  |  Overhead: low",
               fillcolor=MEDIUM["blue"], color=C["blue"][1], **LAYER)

    # ── Layer 3: Session ──
    with g.subgraph(name="cluster_l3") as c:
        c.attr(label="Layer 3: Session Lifecycle (opt-in)",
               style="rounded", color=C["mauve"][1], bgcolor="#EDE8F2",
               fontname="Helvetica", fontsize="12", fontcolor=TD)
        c.node("l3", "SessionTracer + plot_session_trace\nPer-session Gantt / scatter / histogram\nGranularity: per rollout session  |  Overhead: low",
               fillcolor=MEDIUM["mauve"], color=C["mauve"][1], **LAYER)

    # ── Layer 4: Kernel ──
    with g.subgraph(name="cluster_l4") as c:
        c.attr(label="Layer 4: Kernel Profiling (selective steps only)",
               style="rounded", color=C["rose"][1], bgcolor="#F3E6E6",
               fontname="Helvetica", fontsize="12", fontcolor=TD)
        c.node("l4", "torch.profiler (via profile_steps)\n+ profile_fsdp.py / profile_archon.py / profile_engines.py\nGranularity: per CUDA kernel  |  Overhead: high (targeted)",
               fillcolor=MEDIUM["rose"], color=C["rose"][1], **LAYER)

    # ── Edges ──
    g.edge("l1", "l2", label="Need more detail?\nEnable perf_tracer.enabled=true",
           style="bold", color=C["blue"][1], fontcolor=C["blue"][1])
    g.edge("l2", "l3", label="Rollout slow?\nEnable session_tracer.enabled=true",
           style="bold", color=C["mauve"][1], fontcolor=C["mauve"][1])
    g.edge("l3", "l4", label="Train slow?\nSet profile_steps=[N]",
           style="bold", color=C["rose"][1], fontcolor=C["rose"][1])

    # ── Right-side config annotation ──
    g.node("cfg",
           "# Enable all layers:\n"
           "perf_tracer:\n"
           "  enabled: true\n"
           "  profile_steps: [5, 10]\n"
           "  session_tracer:\n"
           "    enabled: true",
           shape="note", fillcolor=C["sand"][0], color=C["sand"][1],
           fontname="Courier", fontsize="9")
    g.edge("l2", "cfg", style="dotted", arrowhead="none", constraint="false")

    _save(g, "perf_04_tool_layers")


if __name__ == "__main__":
    print("Generating performance observability diagrams...")
    create_training_step_pipeline()
    create_rollout_architecture()
    create_analysis_decision_tree()
    create_tool_layers()
    print("Done!")
