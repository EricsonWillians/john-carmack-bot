from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain

def split_document_with_sliding_window(document_list, window_size=512, overlap=100):
    assert isinstance(document_list, list), "Expected a list of Document objects"
    template_document = document_list[0]
    
    # Joining all the page_content into a single string
    combined_text = "\n".join([doc.page_content for doc in document_list])
    
    tokens = combined_text.split()
    chunks = []
    position = 0

    while position < len(tokens):
        start = max(0, position - overlap)
        end = position + window_size - overlap
        
        chunk_text = " ".join(tokens[start:end])
        new_document = type(template_document)(page_content=chunk_text)  # Properly initializing with page_content
        chunks.append(new_document)
        
        position = end

    return chunks


# Load content
loader = PyPDFLoader("contexts/gebbdoom.pdf")
document_content = loader.load_and_split()

# Split the documents using the sliding window approach
document_chunks = split_document_with_sliding_window(document_content)

# Set up LLM and other configurations
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = 20
n_batch = 512
n_ctx = 2048
max_tokens = 512
llm = LlamaCpp(
    model_path="./models/llama-2-7b-chat.ggmlv3.q4_0.bin",
    temperature=0.75,
    max_tokens=max_tokens,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=n_ctx,
    callback_manager=callback_manager,
    verbose=True,
)

chain = load_summarize_chain(llm, chain_type="stuff")

# Summarize each document chunk using the chain
summaries = []
for doc_chunk in document_chunks:
    try:
        if hasattr(doc_chunk, 'page_content') and doc_chunk.page_content:
            summary = chain.run(doc_chunk)
            summaries.append(summary)
    except Exception as e:
        print(f"Error processing chunk: {doc_chunk.page_content}")
        print(str(e))

# Process chunks
MAX_TOKENS_PER_CHUNK = 500
prompt = "Question: How the Doom renderer works?"
responses = []

for doc_chunk in document_chunks:
    
    chunk_text = doc_chunk.page_content
    tokens = chunk_text.split()
    
    if len(tokens) > MAX_TOKENS_PER_CHUNK:
        tokens = tokens[:MAX_TOKENS_PER_CHUNK]
        chunk_text = " ".join(tokens)

    # Combine the chunk with the prompt so LLM understands what to do with the chunk.
    chunk_with_prompt = chunk_text + "\n" + prompt
    response = llm(chunk_with_prompt)
    responses.append(response)

# Post-process the responses as needed.
