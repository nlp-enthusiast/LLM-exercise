from .chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration as ChatGLM2ForConditionalGeneration
from .chatglm2.tokenization_chatglm import ChatGLMTokenizer as ChatGLM2Tokenizer
from .chatglm2.configuration_chatglm import ChatGLMConfig as ChatGLM2Config

ModelMode={"glm2": {"model": ChatGLM2ForConditionalGeneration, "tokenizer": ChatGLM2Tokenizer, "config": ChatGLM2Config}}