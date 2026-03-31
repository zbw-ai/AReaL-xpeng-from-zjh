"""Code Execution (DAPO) multi-turn workflow for AReaL.

Implements the Code-DAPO interaction protocol:
  1. LLM generates text; stops at </code> tag.
  2. Parse Python code from <code>...</code>.
  3. Execute code via local subprocess or remote execd.
  4. Wrap output in <output>...</output> and append to context.
  5. Continue generation until \\boxed{answer} or max_turns.
  6. Compute math answer correctness reward.
"""

import asyncio
import os
import re
import subprocess
import tempfile
import time
import uuid

import httpx
import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import InferenceEngine, ModelRequest, ModelResponse, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.reward import get_math_verify_worker
from areal.utils import logging, stats_tracker

logger = logging.getLogger("CodeExecWorkflow")

# ── Tag parsing ──────────────────────────────────────────────────

CODE_TAG_PATTERN = re.compile(r"<code>(.*?)</code>", re.DOTALL)
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

MAX_OUTPUT_CHARS = 1024


def _extract_code(text: str) -> str | None:
    """Extract Python code from <code>...</code> or <code>... (unclosed, stopped by stop_strings)."""
    m = CODE_TAG_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    # Handle unclosed <code> tag (SGLang/vLLM stop_strings strips the closing tag)
    open_idx = text.rfind("<code>")
    if open_idx != -1:
        code = text[open_idx + len("<code>"):]
        return code.strip() if code.strip() else None
    return None


def _extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...}."""
    matches = BOXED_PATTERN.findall(text)
    return matches[-1].strip() if matches else None


class CodeExecWorkflow(RolloutWorkflow):
    """Multi-turn Code Execution RL workflow (DAPO style).

    Supports two execution modes:
      - local: subprocess.run (default, no external service needed)
      - execd: HTTP POST to remote EXECD_ENDPOINT
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        sandbox_type: str = "local",
        execd_endpoint: str = "",
        code_timeout: int = 10,
        max_turns: int = 10,
        max_tool_uses: int = 5,
        max_total_tokens: int = 8192,
        system_prompt: str = (
            "Please reason step by step, and put your final answer within "
            "'\\boxed{}', e.g. \\boxed{A}."
        ),
    ):
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        stop = list(gconfig.stop) if gconfig.stop is not None else []
        if "</code>" not in stop:
            stop.append("</code>")
        self.gconfig = gconfig.new(stop=stop).new_with_stop_and_pad_token_ids(tokenizer)
        self.sandbox_type = sandbox_type
        self.execd_endpoint = execd_endpoint
        self.code_timeout = code_timeout
        self.max_turns = max_turns
        self.max_tool_uses = max_tool_uses
        self.max_total_tokens = max_total_tokens
        self.system_prompt = system_prompt

    async def _execute_code_local(self, code: str) -> str:
        """Execute Python code via subprocess with timeout and isolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = os.path.join(tmpdir, "run.py")
            with open(code_file, "w") as f:
                f.write(code)
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["python3", code_file],
                    capture_output=True,
                    text=True,
                    timeout=self.code_timeout,
                    cwd=tmpdir,
                )
                output = result.stdout
                if result.returncode != 0:
                    error = result.stderr
                    if error:
                        # Keep last few lines of error
                        error_lines = error.strip().split("\n")
                        error = "\n".join(error_lines[-5:])
                    output = output + "\n[ERROR]\n" + error if output else "[ERROR]\n" + error
            except subprocess.TimeoutExpired:
                output = f"[ERROR] Code execution timed out after {self.code_timeout}s"
            except Exception as e:
                output = f"[ERROR] {type(e).__name__}: {e}"

        # Truncate long output
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n... (truncated to {MAX_OUTPUT_CHARS} chars)"
        return output.strip()

    async def _execute_code_execd(self, code: str, http_client: httpx.AsyncClient) -> str:
        """Execute Python code via remote execd endpoint."""
        try:
            resp = await http_client.post(
                self.execd_endpoint + "/command",
                json={"code": code, "language": "python3"},
                timeout=self.code_timeout + 10,
            )
            resp.raise_for_status()
            result = resp.json()
            output = result.get("stdout", "") + result.get("stderr", "")
        except Exception as e:
            output = f"[ERROR] execd: {e}"
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n... (truncated)"
        return output.strip()

    async def _execute_code(self, code: str, http_client: httpx.AsyncClient | None) -> str:
        """Execute code using configured sandbox type."""
        if self.sandbox_type == "execd" and self.execd_endpoint and http_client:
            return await self._execute_code_execd(code, http_client)
        return await self._execute_code_local(code)

    async def arun_episode(
        self, engine: InferenceEngine, data: dict
    ) -> dict[str, torch.Tensor] | None:
        """Multi-turn generate-execute loop."""
        # Extract problem and answer
        question = data.get("prompt", data.get("question", data.get("problem", "")))
        solution = data.get("solution", data.get("answer", ""))
        # Extract ground truth from \boxed{} in solution
        ground_truth = _extract_boxed_answer(solution) or solution

        # Build initial prompt
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        user_content = question
        if "\\boxed" not in question:
            user_content += "\nEnsure that your response includes the format of '\\boxed{answer}', e.g. \\boxed{A}."
        messages.append({"role": "user", "content": user_content})

        # Track all tokens
        all_input_ids: list[int] = []
        all_logprobs: list[float] = []
        all_loss_mask: list[int] = []
        all_versions: list[int] = []

        tool_use_count = 0
        tool_use_success = 0
        num_turns = 0
        final_reward = 0.0
        accumulated_text = ""

        http_client = None
        if self.sandbox_type == "execd":
            http_client = await workflow_context.get_httpx_client()
        prev_len = 0

        for turn in range(self.max_turns):
            num_turns = turn + 1

            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )

            if len(input_ids) >= self.max_total_tokens:
                break

            gconfig = self.gconfig.new(n_samples=1)
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=list(input_ids),
                gconfig=gconfig,
                tokenizer=self.tokenizer,
            )

            resp: ModelResponse = await engine.agenerate(req)
            output_text = self.tokenizer.decode(resp.output_tokens, skip_special_tokens=False)
            accumulated_text += output_text

            # Record tokens
            if turn == 0:
                all_input_ids.extend(resp.input_tokens)
                all_logprobs.extend([0.0] * resp.input_len)
                all_loss_mask.extend([0] * resp.input_len)
                all_versions.extend([-1] * resp.input_len)
            else:
                new_context_len = resp.input_len - prev_len
                if new_context_len > 0:
                    new_context_tokens = resp.input_tokens[prev_len:]
                    all_input_ids.extend(new_context_tokens)
                    all_logprobs.extend([0.0] * len(new_context_tokens))
                    all_loss_mask.extend([0] * len(new_context_tokens))
                    all_versions.extend([-1] * len(new_context_tokens))

            all_input_ids.extend(resp.output_tokens)
            all_logprobs.extend(resp.output_logprobs)
            all_loss_mask.extend([1] * resp.output_len)
            all_versions.extend(resp.output_versions)

            prev_len = resp.input_len + resp.output_len

            # Check for \boxed{} answer
            boxed_answer = _extract_boxed_answer(accumulated_text)
            if boxed_answer is not None:
                worker = get_math_verify_worker()
                final_reward = worker.verify(accumulated_text, ground_truth)
                break

            # Check for <code> tag
            code = _extract_code(output_text)
            if code is not None and tool_use_count < self.max_tool_uses:
                tool_use_count += 1
                exec_output = await self._execute_code(code, http_client)

                if not exec_output.startswith("[ERROR]"):
                    tool_use_success += 1

                result_text = f"\n<output>\n{exec_output}\n</output>\n"
                result_tokens = self.tokenizer.encode(result_text, add_special_tokens=False)

                all_input_ids.extend(result_tokens)
                all_logprobs.extend([0.0] * len(result_tokens))
                all_loss_mask.extend([0] * len(result_tokens))
                all_versions.extend([-1] * len(result_tokens))

                messages.append({"role": "assistant", "content": output_text + result_text})
                messages.append({
                    "role": "user",
                    "content": "Continue. Put your final answer in \\boxed{}.",
                })
            else:
                messages.append({"role": "assistant", "content": output_text})
                messages.append({
                    "role": "user",
                    "content": "Please provide your final answer in \\boxed{}.",
                })

        stats_tracker.get(workflow_context.stat_scope()).scalar(
            tool_use_count=tool_use_count,
            tool_use_success=tool_use_success / max(tool_use_count, 1),
            num_turns=num_turns,
            reward=final_reward,
        )

        if not all_input_ids:
            return None

        if len(all_input_ids) > self.max_total_tokens:
            all_input_ids = all_input_ids[: self.max_total_tokens]
            all_logprobs = all_logprobs[: self.max_total_tokens]
            all_loss_mask = all_loss_mask[: self.max_total_tokens]
            all_versions = all_versions[: self.max_total_tokens]

        seq_len = len(all_input_ids)
        res = {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.int32),
            "loss_mask": torch.tensor(all_loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(all_logprobs, dtype=torch.float32),
            "versions": torch.tensor(all_versions, dtype=torch.int32),
            "attention_mask": torch.ones(seq_len, dtype=torch.bool),
            "rewards": torch.tensor(final_reward, dtype=torch.float32),
        }
        return {k: v.unsqueeze(0) for k, v in res.items()}
