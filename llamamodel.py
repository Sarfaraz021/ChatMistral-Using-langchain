from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate


# Callbacks support token-wise streaming

def load_model() -> LlamaCpp:

    callback_manager: CallbackManager = CallbackManager(
        [StreamingStdOutCallbackHandler()])
    Model_path = "C:\LLAMA2Locally\model\llama-2-13b-chat.Q3_K_M.gguf"
    # Make sure the model path is correct for your system!
    Llama_model: LlamaCpp = LlamaCpp(
        model_path=Model_path,
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True  # Verbose is required to pass to the callback manager
    )
    return Llama_model


llm = load_model()

model_prompt: str = """
Question: what is metaverse
"""

response: str = llm(model_prompt)
