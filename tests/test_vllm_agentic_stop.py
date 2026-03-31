from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest
from areal.engine.vllm_remote import VLLMBackend
from areal.workflow.code_exec import CodeExecWorkflow
from areal.workflow.search_r1 import SearchR1Workflow


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2


def test_search_workflow_adds_search_stop_and_vllm_payload_keeps_it():
    tokenizer = _DummyTokenizer()
    workflow = SearchR1Workflow(
        gconfig=GenerationHyperparameters(stop=["custom-stop"], frequency_penalty=0.5),
        tokenizer=tokenizer,
        retrieval_endpoint="http://retrieval.local/retrieve",
    )

    assert workflow.gconfig.stop == ["custom-stop", "</search>"]

    backend = VLLMBackend()
    req = ModelRequest(input_ids=[1, 2, 3], gconfig=workflow.gconfig)
    http_req = backend.build_generation_request(req=req, with_lora=False, version=0)

    assert http_req.payload["stop"] == ["custom-stop", "</search>"]
    assert http_req.payload["frequency_penalty"] == 0.5


def test_code_exec_workflow_adds_code_stop_and_vllm_payload_keeps_it():
    tokenizer = _DummyTokenizer()
    workflow = CodeExecWorkflow(
        gconfig=GenerationHyperparameters(),
        tokenizer=tokenizer,
    )

    assert workflow.gconfig.stop == ["</code>"]

    backend = VLLMBackend()
    req = ModelRequest(input_ids=[1, 2, 3], gconfig=workflow.gconfig)
    http_req = backend.build_generation_request(req=req, with_lora=False, version=0)

    assert http_req.payload["stop"] == ["</code>"]

