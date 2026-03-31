"""Search-R1 multi-turn workflow for AReaL.

Implements the Search-R1 interaction protocol:
  1. LLM generates text; stops at </search> tag.
  2. Parse search query from <search>query</search>.
  3. POST query to RETRIEVAL_ENDPOINT, get documents.
  4. Wrap results in <result>...</result> and append to context.
  5. Continue generation until <answer>...</answer> or max_turns.
  6. Compute EM reward on golden answers.
"""

import re
import time
import uuid

import httpx
import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import InferenceEngine, ModelRequest, ModelResponse, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.utils import logging, stats_tracker

logger = logging.getLogger("SearchR1Workflow")

# ── Tag parsing ──────────────────────────────────────────────────

SEARCH_TAG_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_search_query(text: str) -> str | None:
    """Extract query from <search>query</search> or <search>query (unclosed, stopped by stop_strings)."""
    m = SEARCH_TAG_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    # Handle unclosed <search> tag (stop_strings strips the closing tag)
    open_idx = text.rfind("<search>")
    if open_idx != -1:
        query = text[open_idx + len("<search>"):]
        return query.strip() if query.strip() else None
    return None


def _extract_answer(text: str) -> str | None:
    """Extract answer from <answer>answer</answer>."""
    m = ANSWER_TAG_PATTERN.search(text)
    return m.group(1).strip() if m else None


def _normalize(s: str) -> str:
    """Normalize text for EM comparison."""
    import string
    import unicodedata

    s = unicodedata.normalize("NFD", s)
    s = s.lower().strip()
    # Remove punctuation
    s = "".join(c for c in s if c not in string.punctuation)
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, golden_answers: list[str] | str) -> float:
    """Compute Exact Match reward."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred_norm = _normalize(prediction)
    for ans in golden_answers:
        if _normalize(ans) == pred_norm:
            return 1.0
    return 0.0


class SearchR1Workflow(RolloutWorkflow):
    """Multi-turn Search-R1 style workflow.

    Requires RETRIEVAL_ENDPOINT to be available.
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        retrieval_endpoint: str,
        max_turns: int = 10,
        max_tool_uses: int = 2,
        max_total_tokens: int = 12800,
        system_prompt: str = "You're a helpful assistant.",
    ):
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        stop = list(gconfig.stop) if gconfig.stop is not None else []
        if "</search>" not in stop:
            stop.append("</search>")
        self.gconfig = gconfig.new(stop=stop).new_with_stop_and_pad_token_ids(tokenizer)
        self.retrieval_endpoint = retrieval_endpoint
        self.max_turns = max_turns
        self.max_tool_uses = max_tool_uses
        self.max_total_tokens = max_total_tokens
        self.system_prompt = system_prompt

    async def _retrieve(self, query: str, http_client: httpx.AsyncClient) -> str:
        """Call xpeng_retriever and format results."""
        try:
            resp = await http_client.post(
                self.retrieval_endpoint,
                json={"query": query},
                timeout=30.0,
            )
            resp.raise_for_status()
            results = resp.json()
            # Format as text
            if isinstance(results, list):
                parts = []
                for i, doc in enumerate(results[:5], 1):
                    title = doc.get("title", "")
                    content = doc.get("content", doc.get("text", str(doc)))
                    # Truncate long content
                    if len(content) > 500:
                        content = content[:500] + "..."
                    parts.append(f"[{i}] {title}\n{content}")
                return "\n\n".join(parts)
            return str(results)[:2000]
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return f"Error: retrieval failed ({e})"

    async def arun_episode(
        self, engine: InferenceEngine, data: dict
    ) -> dict[str, torch.Tensor] | None:
        """Multi-turn generate-retrieve loop."""
        # Extract golden answers
        golden_answers = data.get("golden_answers", data.get("answer", ""))
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]

        # Build initial prompt — use data's original prompt if available
        messages = []
        prompt_field = data.get("prompt")
        if prompt_field and isinstance(prompt_field, list):
            # Data has pre-built messages (e.g., nq_search with full instructions)
            messages = list(prompt_field)
        else:
            # Fallback: build from question
            question = data.get("question", data.get("problem", ""))
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": question})

        # Track all tokens for the full trajectory
        all_input_ids: list[int] = []
        all_logprobs: list[float] = []
        all_loss_mask: list[int] = []
        all_versions: list[int] = []

        tool_use_count = 0
        tool_use_success = 0
        num_turns = 0
        final_reward = 0.0

        http_client = await workflow_context.get_httpx_client()
        prev_len = 0

        for turn in range(self.max_turns):
            num_turns = turn + 1

            # Tokenize current context
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )

            # Check total length
            if len(input_ids) >= self.max_total_tokens:
                logger.debug(f"Context exceeded max_total_tokens ({len(input_ids)})")
                break

            # Create generation request with stop at </search>
            gconfig = self.gconfig.new(n_samples=1)
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=list(input_ids),
                gconfig=gconfig,
                tokenizer=self.tokenizer,
            )

            resp: ModelResponse = await engine.agenerate(req)
            output_text = self.tokenizer.decode(resp.output_tokens, skip_special_tokens=False)

            if turn == 0:
                # First turn: record full prompt + generated tokens
                all_input_ids.extend(resp.input_tokens)
                all_logprobs.extend([0.0] * resp.input_len)
                all_loss_mask.extend([0] * resp.input_len)
                all_versions.extend([-1] * resp.input_len)
            else:
                # Subsequent turns: the engine sees re-tokenized full context as input.
                # We already have tokens up to prev_len in all_input_ids.
                # Record only the NEW context tokens (between prev_len and input_len)
                # as non-trainable (these are re-encoded retrieval results + user prompts).
                new_context_len = resp.input_len - prev_len
                if new_context_len > 0:
                    new_context_tokens = resp.input_tokens[prev_len:]
                    all_input_ids.extend(new_context_tokens)
                    all_logprobs.extend([0.0] * len(new_context_tokens))
                    all_loss_mask.extend([0] * len(new_context_tokens))
                    all_versions.extend([-1] * len(new_context_tokens))

            # Record generated tokens (always trainable)
            all_input_ids.extend(resp.output_tokens)
            all_logprobs.extend(resp.output_logprobs)
            all_loss_mask.extend([1] * resp.output_len)
            all_versions.extend(resp.output_versions)

            # Track position for next turn
            prev_len = resp.input_len + resp.output_len

            # Check for <answer> tag
            answer_text = _extract_answer(output_text)
            if answer_text is not None:
                final_reward = exact_match(answer_text, golden_answers)
                break

            # Check for <search> tag
            search_query = _extract_search_query(output_text)
            if search_query is not None and tool_use_count < self.max_tool_uses:
                tool_use_count += 1
                t0 = time.monotonic()
                retrieval_result = await self._retrieve(search_query, http_client)
                search_latency = (time.monotonic() - t0) * 1000  # ms

                if not retrieval_result.startswith("Error:"):
                    tool_use_success += 1

                # Append retrieval result to conversation
                result_text = f"\n<information>\n{retrieval_result}\n</information>\n"
                result_tokens = self.tokenizer.encode(result_text, add_special_tokens=False)

                # Result tokens are NOT trainable (loss_mask=0)
                all_input_ids.extend(result_tokens)
                all_logprobs.extend([0.0] * len(result_tokens))
                all_loss_mask.extend([0] * len(result_tokens))
                all_versions.extend([-1] * len(result_tokens))

                # Update messages for next turn
                assistant_content = output_text + result_text
                messages.append({"role": "assistant", "content": assistant_content})
                # Add continuation prompt
                messages.append({"role": "user", "content": "Continue based on the search results."})

                stats_tracker.get(workflow_context.stat_scope()).scalar(
                    search_latency_ms=search_latency
                )
            else:
                # No search tag and no answer tag — model didn't use tools
                # Append as assistant message and continue
                messages.append({"role": "assistant", "content": output_text})
                messages.append({"role": "user", "content": "Please provide your final answer in <answer>...</answer> tags."})

        # Report agentic metrics
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            tool_use_count=tool_use_count,
            tool_use_success=tool_use_success / max(tool_use_count, 1),
            num_turns=num_turns,
            reward=final_reward,
        )

        if not all_input_ids:
            return None

        # Truncate to max_total_tokens
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
