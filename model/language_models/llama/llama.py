from typing import Optional, Any, List

from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain.llms.base import BaseLLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema import LLMResult, Generation
from transformers import AutoModelForCausalLM, AutoTokenizer


class Llama(BaseLLM):
    model: object = None
    tokenizer: object = None
    model_args: dict = {}
    streaming: bool = False

    def __init__(self, model_config=None, model_args=None, streaming=False, **kwargs: Any):
        super().__init__(**kwargs)
        self.streaming = streaming
        if model_args is None:
            model_args = {}
        if model_config is None:
            model_config = {}
        model_path = 'meta-llama/Llama-2-7b-hf'

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            device_map='auto'
        )

        self.model = model
        self.tokenizer = tokenizer
        self.load_model_args(**model_args)

    @property
    def _llm_type(self) -> str:
        return "llama"

    def _generate(self, prompts: List[str], history=None, stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> LLMResult:
        generations = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            generate_ids = self.model.generate(inputs.input_ids,
                                               do_sample=True,
                                               # temperature=0.1,
                                               # repetition_penalty=1.0,
                                               max_new_tokens=2048)
            generate_ids = generate_ids[:, inputs.input_ids.shape[1]:]
            response = self.tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False
            )[0]

            if stop is not None:
                response = enforce_stop_tokens(response, stop)
            generations.append([
                Generation(
                    text=response,
                )
            ])
        return LLMResult(
            generations=generations,
        )

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None,
                         run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any) -> LLMResult:
        pass

    def load_model_args(self, **model_args):
        # if model_args is None:
        #     model_args = {}
        # model_args['temperature'] = model_args.get('temperature', 0.95)
        # if model_args['temperature'] <= 0:
        #     model_args['temperature'] = 0.1
        # if model_args['temperature'] > 1:
        #     model_args['temperature'] = 1
        # model_args['top_p'] = model_args.get('top_p', 0.7)
        # model_args['max_tokens'] = model_args.get('max_tokens', 2048)
        self.model_args = model_args
