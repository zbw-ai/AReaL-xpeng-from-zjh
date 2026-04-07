"""Generate diagrams for fuyao-three-experiments-requirement.md.

Run:
    python docs/figures/fuyao-three-experiments/generate_diagrams.py
"""

import graphviz
import os

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
    g.attr(rankdir=rankdir, bgcolor=BG, fontname="Helvetica", dpi="200",
           label=title, labelloc="t", fontsize="18", fontcolor=TD,
           pad="0.5", nodesep=nodesep, ranksep=ranksep, **extra)
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
           fontsize="11", color=EC, fontcolor=TD)
    g.attr("edge", fontname="Helvetica", fontsize="9", color=EC, fontcolor=TM)
    return g


def _save(g, name):
    path = os.path.join(OUTPUT_DIR, name)
    g.render(path, cleanup=True)
    print(f"  -> {name}.png")


# ─────────────────────────────────────────────────────────────────────
# Diagram 1: Training Architecture Overview
# ─────────────────────────────────────────────────────────────────────
def create_architecture():
    g = _base("arch", "AReaL Training Architecture (Single Node, 8 GPU)",
              rankdir="TB", ranksep="0.9", nodesep="0.8")

    # ── Entry Layer ──
    g.node("entry", "fuyao_areal_run.sh\n(Unified Launcher)", shape="oval",
           fillcolor=C["sand"][0], color=C["sand"][1])

    # ── Controller ──
    with g.subgraph(name="cluster_ctrl") as c:
        c.attr(label="Controller Process", style="rounded",
               color=C["mauve"][1], bgcolor=C["mauve"][0],
               fontname="Helvetica", fontsize="12", fontcolor=TD)
        c.node("trainer", "PPOTrainer\n(Epoch Loop)", fillcolor=MEDIUM["mauve"], color=C["mauve"][1])
        c.node("rollout_ctrl", "RolloutController\n(Async Dispatch)", fillcolor=MEDIUM["mauve"], color=C["mauve"][1])
        c.node("stats", "StatsLogger\n(SwanLab)", fillcolor=MEDIUM["mauve"], color=C["mauve"][1])
        c.edge("trainer", "rollout_ctrl")
        c.edge("trainer", "stats", style="dashed")

    # ── Inference Engine ──
    with g.subgraph(name="cluster_rollout") as c:
        c.attr(label="Rollout Engine  (GPU 0-3)", style="rounded",
               color=C["blue"][1], bgcolor=C["blue"][0],
               fontname="Helvetica", fontsize="12", fontcolor=TD)
        c.node("sglang", "SGLang Server\n(DP=4, TP=1)", fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.node("proxy", "OpenAI Proxy\n(HTTP API)", fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.edge("sglang", "proxy", label="Token\nLogprobs", dir="both")

    # ── Workflow / Agent ──
    with g.subgraph(name="cluster_wf") as c:
        c.attr(label="Workflow / Agent Layer", style="dashed,rounded",
               color=C["green"][1], bgcolor=C["green"][0],
               fontname="Helvetica", fontsize="12", fontcolor=TD)
        c.node("rlvr", "RLVRWorkflow\n(Math RLVR)", fillcolor=MEDIUM["green"], color=C["green"][1])
        c.node("code_agent", "CodeExecAgent\n(Code DAPO)", fillcolor=MEDIUM["green"], color=C["green"][1])
        c.node("search_agent", "SearchR1Agent\n(Search R1)", fillcolor=MEDIUM["green"], color=C["green"][1])
        c.body.append('{rank=same; rlvr; code_agent; search_agent;}')

    # ── Training Engine ──
    with g.subgraph(name="cluster_actor") as c:
        c.attr(label="Actor Engine  (GPU 4-7)", style="rounded",
               color=C["rose"][1], bgcolor=C["rose"][0],
               fontname="Helvetica", fontsize="12", fontcolor=TD)
        c.node("megatron", "Megatron Actor\n(DP=4, TP=1)", fillcolor=MEDIUM["rose"], color=C["rose"][1])
        c.node("ref", "Ref Model\n(Colocated)", fillcolor=MEDIUM["rose"], color=C["rose"][1])
        c.edge("megatron", "ref", style="dashed", label="Share GPU")

    # ── External ──
    g.node("tools", "External Tools", shape="box3d",
           fillcolor=C["terracotta"][0], color=C["terracotta"][1])
    g.node("tools_detail", "subprocess (Code)\nHTTP Retriever (Search)",
           shape="note", fillcolor=C["sand"][0], color=C["sand"][1], fontsize="9")

    # ── Edges ──
    g.edge("entry", "trainer", label="--run-type\n--config")
    g.edge("rollout_ctrl", "proxy", label="Submit\nEpisodes")
    g.edge("rlvr", "sglang", label="engine.\nagenerate()", style="bold",
           color=C["blue"][1], fontcolor=C["blue"][1])
    g.edge("code_agent", "proxy", label="OpenAI\nSDK", style="dashed")
    g.edge("search_agent", "proxy", label="OpenAI\nSDK", style="dashed")
    g.edge("code_agent", "tools", label="exec code", style="dashed",
           color=C["terracotta"][1], fontcolor=C["terracotta"][1])
    g.edge("search_agent", "tools", label="HTTP search", style="dashed",
           color=C["terracotta"][1], fontcolor=C["terracotta"][1])
    g.edge("tools", "tools_detail", style="dotted", arrowhead="none", constraint="false")
    g.edge("proxy", "megatron", label="Trajectories\n(input_ids, logprobs,\nloss_mask, rewards)",
           style="bold", color=C["rose"][1], fontcolor=C["rose"][1])
    g.edge("megatron", "sglang", label="Weight Sync\n(NCCL xccl)",
           style="bold", color=C["terracotta"][1], fontcolor=C["terracotta"][1],
           dir="both", constraint="false")

    _save(g, "01_architecture")


# ─────────────────────────────────────────────────────────────────────
# Diagram 2: Three Workflow Comparison
# ─────────────────────────────────────────────────────────────────────
def create_workflow_comparison():
    g = _base("wf_cmp", "Three Experiment Workflows Comparison",
              rankdir="TB", ranksep="0.6", nodesep="0.5")

    # ── Math RLVR ──
    with g.subgraph(name="cluster_math") as c:
        c.attr(label="Math RLVR  (Layer 1 — Direct Workflow)", style="rounded",
               color=C["blue"][1], bgcolor=C["blue"][0],
               fontname="Helvetica", fontsize="11", fontcolor=TD)
        c.node("m1", "Tokenize\n(chat template)", fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.node("m2", "engine.agenerate()\n(n_samples=4)", fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.node("m3", "Decode\nCompletions", fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.node("m4", "math_verify\n(boxed answer)", fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.node("m5", "Return Tensors\n(input_ids, logprobs,\nloss_mask, rewards)", fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.edge("m1", "m2")
        c.edge("m2", "m3")
        c.edge("m3", "m4")
        c.edge("m4", "m5")

    # ── Code DAPO ──
    with g.subgraph(name="cluster_code") as c:
        c.attr(label="Code DAPO  (Layer 2 — OpenAI Proxy Agent)", style="rounded",
               color=C["green"][1], bgcolor=C["green"][0],
               fontname="Helvetica", fontsize="11", fontcolor=TD)
        c.node("c1", 'LLM Generate\nstop=["</code>"]', fillcolor=MEDIUM["green"], color=C["green"][1])
        c.node("c2", "Parse\n<code>...</code>", fillcolor=MEDIUM["green"], color=C["green"][1], shape="diamond")
        c.node("c3", "subprocess\npython3 run.py", fillcolor=C["terracotta"][0], color=C["terracotta"][1])
        c.node("c4", "Inject\n<output>result</output>", fillcolor=MEDIUM["green"], color=C["green"][1])
        c.node("c5", "Found \\boxed{} ?", fillcolor=MEDIUM["green"], color=C["green"][1], shape="diamond")
        c.node("c6", "math_verify\n+ return reward", fillcolor=MEDIUM["green"], color=C["green"][1])
        c.edge("c1", "c2")
        c.edge("c2", "c3", label="has code")
        c.edge("c3", "c4")
        c.edge("c4", "c5")
        c.edge("c5", "c1", label="no, loop", style="dashed",
               color=C["green"][1], fontcolor=C["green"][1])
        c.edge("c2", "c5", label="no code")
        c.edge("c5", "c6", label="yes / max_turns")

    # ── Search R1 ──
    with g.subgraph(name="cluster_search") as c:
        c.attr(label="Search R1  (Layer 2 — OpenAI Proxy Agent)", style="rounded",
               color=C["mauve"][1], bgcolor=C["mauve"][0],
               fontname="Helvetica", fontsize="11", fontcolor=TD)
        c.node("s1", 'LLM Generate\nstop=["</search>"]', fillcolor=MEDIUM["mauve"], color=C["mauve"][1])
        c.node("s2", "Parse\n<search>query</search>", fillcolor=MEDIUM["mauve"], color=C["mauve"][1], shape="diamond")
        c.node("s3", "HTTP POST\nRetriever API", fillcolor=C["terracotta"][0], color=C["terracotta"][1])
        c.node("s4", "Inject\n<information>docs</information>", fillcolor=MEDIUM["mauve"], color=C["mauve"][1])
        c.node("s5", "Found <answer> ?", fillcolor=MEDIUM["mauve"], color=C["mauve"][1], shape="diamond")
        c.node("s6", "exact-match\n+ return reward", fillcolor=MEDIUM["mauve"], color=C["mauve"][1])
        c.edge("s1", "s2")
        c.edge("s2", "s3", label="has query")
        c.edge("s3", "s4")
        c.edge("s4", "s5")
        c.edge("s5", "s1", label="no, loop", style="dashed",
               color=C["mauve"][1], fontcolor=C["mauve"][1])
        c.edge("s2", "s5", label="no query")
        c.edge("s5", "s6", label="yes / max_turns")

    # ── Invisible edges for vertical ordering ──
    g.edge("m1", "c1", style="invis")
    g.edge("c1", "s1", style="invis")

    # ── Legend ──
    g.node("legend",
           "Layer 1: Direct engine access, manual tensor construction\n"
           "Layer 2: OpenAI Proxy auto-wraps agent, auto-builds loss_mask",
           shape="note", fillcolor=C["sand"][0], color=C["sand"][1], fontsize="9")
    g.edge("s6", "legend", style="invis")

    _save(g, "02_workflow_comparison")


# ─────────────────────────────────────────────────────────────────────
# Diagram 3: Proxy-Mode Data Flow (Code DAPO / Search R1)
# ─────────────────────────────────────────────────────────────────────
def create_data_flow():
    g = _base("dataflow", "Proxy-Mode Agentic Training Data Flow",
              rankdir="TB", ranksep="0.7", nodesep="0.7")

    # ── Data Input ──
    g.node("dataset", "Dataset\n(dapo_math_17k / nq_search)", shape="cylinder",
           fillcolor=C["green"][0], color=C["green"][1])

    # ── PPOTrainer ──
    g.node("ppo", "PPOTrainer.train()", fillcolor=C["mauve"][0], color=C["mauve"][1])

    # ── Auto-wrap Detection ──
    g.node("detect", "Is RolloutWorkflow?", shape="diamond",
           fillcolor=C["sand"][0], color=C["sand"][1])
    g.node("direct", "Direct Workflow\n(Math RLVR)", fillcolor=C["blue"][0], color=C["blue"][1])
    g.node("wrap", "OpenAIProxyWorkflow\nAuto-Wrap Agent", fillcolor=C["mauve"][0], color=C["mauve"][1])

    # ── Proxy Session ──
    with g.subgraph(name="cluster_session") as c:
        c.attr(label="Proxy Session (per episode)", style="dashed,rounded",
               color=C["blue"][1], bgcolor="#E8EEF3",
               fontname="Helvetica", fontsize="11", fontcolor=TD)
        c.node("start", "start_session()\nget api_key + base_url", fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.node("agent_run", "Agent.run(data)\nMulti-turn LLM + Tools", fillcolor=MEDIUM["green"], color=C["green"][1])
        c.node("proxy_track", "Proxy Server\nRecords token logprobs", fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.node("reward_set", "Agent returns reward\nProxy assigns to tokens", fillcolor=MEDIUM["green"], color=C["green"][1])
        c.node("export", "export_interactions()\nBuild training tensors", fillcolor=MEDIUM["blue"], color=C["blue"][1])
        c.edge("start", "agent_run")
        c.edge("agent_run", "proxy_track", dir="both", label="OpenAI API\ncalls",
               style="bold", color=C["blue"][1], fontcolor=C["blue"][1])
        c.edge("agent_run", "reward_set")
        c.edge("reward_set", "export")

    # ── Tensor Output ──
    g.node("tensors",
           "Training Tensors\ninput_ids | logprobs | loss_mask | rewards | attention_mask",
           fillcolor=C["rose"][0], color=C["rose"][1])

    # ── Loss Mask Detail ──
    g.node("mask_detail",
           "loss_mask construction:\n"
           "  Model-generated tokens = 1 (trainable)\n"
           "  Tool outputs / user prompts = 0 (frozen)\n"
           "  System prompt = 0 (frozen)",
           shape="note", fillcolor=C["sand"][0], color=C["sand"][1],
           fontname="Courier", fontsize="9")

    # ── Actor ──
    g.node("actor", "Megatron Actor\nPPO Gradient Update", fillcolor=C["rose"][0], color=C["rose"][1])
    g.node("weight_sync", "Weight Sync\n(NCCL → SGLang)", shape="oval",
           fillcolor=C["terracotta"][0], color=C["terracotta"][1])

    # ── Edges ──
    g.edge("dataset", "ppo")
    g.edge("ppo", "detect")
    g.edge("detect", "direct", label="Yes\n(RLVRWorkflow)")
    g.edge("detect", "wrap", label="No\n(CodeExecAgent,\nSearchR1Agent)")
    g.edge("wrap", "start")
    g.edge("export", "tensors")
    g.edge("tensors", "mask_detail", style="dotted", arrowhead="none", constraint="false")
    g.edge("tensors", "actor", style="bold", color=C["rose"][1], fontcolor=C["rose"][1])
    g.edge("actor", "weight_sync")
    g.edge("weight_sync", "proxy_track", label="Updated\nWeights",
           style="bold", color=C["terracotta"][1], fontcolor=C["terracotta"][1],
           constraint="false")

    _save(g, "03_proxy_data_flow")


if __name__ == "__main__":
    print("Generating diagrams...")
    create_architecture()
    create_workflow_comparison()
    create_data_flow()
    print("Done!")
