from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp


def load_model(model_path: str, temperature: float = 0.75, max_tokens: int = 2000, top_p: float = 1, verbose: bool = False) -> LlamaCpp:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llama_model = LlamaCpp(
        model_path=model_path,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        callback_manager=callback_manager,
        verbose=verbose
    )
    return llama_model


def main():
    model_path = "E:\\ChatMistral-Using-langchain\\llama-2-13b-chat-dutch.Q2_K.gguf"
    llama = load_model(model_path)

    while True:
        prompt = input("Prompt: ")
        if prompt.lower() == "exit":
            break
        else:
            print(f"ChatBot: {llama.invoke(prompt)}")


if __name__ == "__main__":
    main()
