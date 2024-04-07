# %%
# !pip install --upgrade --quiet  llama-cpp-python
# !CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

#!CMAKE_ARGS="-DLLAMA_OpenBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir


from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp


# %%
import os
#os.system("CMAKE_ARGS=\"-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS\" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir",)

# %%
#os.system("huggingface-cli download TheBloke/Mixtral-8x7B-v0.1-GGUF mixtral-8x7b-v0.1.Q4_K_M.gguf --local-dir GGUFs")
#os.system("huggingface-cli download TheBloke/Llama-2-70B-GGUF llama-2-70b.Q4_K_M.gguf --local-dir .")
#os.system("huggingface-cli download YokaiKoibito/falcon-40b-GGUF falcon-40b-Q4_K_M.gguf --local-dir .")


# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path='GGUFs\mixtral-8x7b-v0.1.Q4_K_M.gguf',#"mixtral-8x7b-v0.1.Q4_K_M.gguf",
    temperature=0.0,
    max_tokens=1024,
    max_new_tokens=1024,
    do_sample=False,
    #top_p=1,
    # callback_manager=callback_manager,
    # verbose=False, #True,  # Verbose is required to pass to the callback manager
)




