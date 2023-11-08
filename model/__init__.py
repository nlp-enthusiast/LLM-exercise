from .chatglm3.modeling_chatglm import ChatGLMForConditionalGeneration as ChatGLM3ForConditionalGeneration
from .chatglm3.tokenization_chatglm import ChatGLMTokenizer as ChatGLM3Tokenizer
from .chatglm3.configuration_chatglm import ChatGLMConfig as ChatGLM3Config

ModelMode = {"glm3": {"model": ChatGLM3ForConditionalGeneration, "tokenizer": ChatGLM3Tokenizer, "config": ChatGLM3Config}}
