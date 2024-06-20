from typing import Optional, Any, List, Dict
from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models.jinachat import _convert_message_to_dict, _convert_dict_to_message
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema import BaseMessage, ChatResult, ChatGeneration
from transformers import AutoModel, AutoTokenizer
from model.large_models.language_models.chatglm.utils import get_history, load_model_on_gpus


class ChatGLMChatModel(BaseChatModel):
    model: object = None
    tokenizer: object = None
    model_args: dict = {}
    streaming: bool = True

    def __init__(self, model_config=None, model_args=None, streaming=True, **kwargs: Any):
        super().__init__(**kwargs)
        self.streaming = streaming
        if model_args is None:
            model_args = {}
        if model_config is None:
            model_config = {}
        model_path = 'THUDM/chatglm2-6b'
        device = model_config.get('device', 'cuda:0')
        quantize = model_config.get('quantize', 16)
        if device == 'cpu':
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float()
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = load_model_on_gpus(model_path)

        model = model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.load_model_args(model_args)

    @property
    def _llm_type(self) -> str:
        return "chatglm"

    def _generate(self,
                  messages: List[BaseMessage],
                  stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        message_dicts = self._create_message_dicts(messages)
        history = get_history(message_dicts)
        generations = []
        response, history = self.model.chat(
            self.tokenizer, message_dicts, history,
            temperature=self.model_args['temperature'],
            top_p=self.model_args['top_p'],
            max_length=self.model_args['max_tokens'])
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        message = _convert_dict_to_message({
            'content': response,
            'role': 'ai'
        })
        generations.append(ChatGeneration(message=message))
        return ChatResult(generations=generations)

    def _create_message_dicts(
            self,
            messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts

    async def _agenerate(self, messages: List[BaseMessage],
                         stop: Optional[List[str]] = None,
                         run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
                         **kwargs: Any) -> ChatResult:
        pass

    def load_model_args(self, model_args):
        if model_args is None:
            model_args = {}
        model_args['temperature'] = model_args.get('temperature', 0.95)
        if model_args['temperature'] <= 0:
            model_args['temperature'] = 0.1
        if model_args['temperature'] > 1:
            model_args['temperature'] = 1
        model_args['top_p'] = model_args.get('top_p', 0.7)
        model_args['max_tokens'] = model_args.get('max_tokens', 4000)
        self.model_args = model_args
