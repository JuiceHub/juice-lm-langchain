from typing import Optional, Any, List
from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain.llms.base import BaseLLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema import LLMResult, Generation
from modelscope import AutoTokenizer, AutoModel


class ChatGLM(BaseLLM):
    model: object = None
    tokenizer: object = None
    config: dict = None
    streaming: bool = True
    device: str = 'cuda:3'

    def __init__(self, model_config=None, model_args=None, streaming=False, **kwargs: Any):
        super().__init__(**kwargs)
        self.streaming = streaming
        if model_config is None:
            model_config = {}
        model_path = '/home/user/ygz/models/ZhipuAI/chatglm3-6b'
        device = model_config.get('device', 'cuda:3')
        self.device = device
        quantize = model_config.get('quantize', 16)
        if device == 'cpu':
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float()

            self.model = model
            self.tokenizer = tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda(device)
            model = model.eval()
            self.model = model
            self.tokenizer = tokenizer

    @property
    def _llm_type(self) -> str:
        return "chatglm"

    def _generate(self, prompts: List[str], history=None, model_args=None, stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> LLMResult:
        generations = []
        model_args = self.load_model_args(model_args)
        for prompt in prompts:
            if self.streaming:
                response = ""
                last = ""
                for res, history in self.model.stream_chat(
                        self.tokenizer, prompt, history,
                        temperature=self.model_args['temperature'],
                        top_p=self.model_args['top_p'],
                        max_length=self.model_args['max_tokens']):
                    if last != "":
                        res = str(res).replace(last, "")
                    response += res
                    last += res
                    if stop is not None:
                        response = enforce_stop_tokens(response, stop)
                    if run_manager:
                        run_manager.on_llm_new_token(
                            res,
                        )
                generations.append([
                    Generation(
                        text=response,
                    )
                ])
            else:
                response, history = self.model.chat(
                    self.tokenizer, prompt, history,
                    temperature=model_args['temperature'],
                    top_p=model_args['top_p'],
                    max_length=model_args['max_length']
                )
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

    def load_model_args(self, model_args):
        if model_args is None:
            model_args = {}
        model_args['temperature'] = model_args.get('temperature', 0.1)
        model_args['top_p'] = model_args.get('top_p', 1.0)
        model_args['max_length'] = model_args.get('max_length', 2048)
        return model_args
