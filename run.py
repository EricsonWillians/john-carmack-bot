import os
from tqdm import tqdm
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

def process_repository(directory_path, file_types=(".cpp", ".h", ".txt")):
    all_content = []

    for subdir, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(file_types):  # Filtering text-based files
                with open(os.path.join(subdir, file), 'r', errors='replace') as f:
                    content = f.read()
                    all_content.append(content)

    # Given that there's no "Document" type mentioned for the source code, 
    # you can just treat the content as a single "pseudo-document" with joined content
    combined_text = "\n".join(all_content)
    source_document = type(document_content[0])(page_content=combined_text)  # Using the type of your existing documents
    
    # Now, you can reuse the split_document_with_sliding_window function
    return split_document_with_sliding_window([source_document])


# Load content
loader = PyPDFLoader("contexts/gebbdoom.pdf")
document_content = loader.load_and_split()

# Split the documents using the sliding window approach
document_chunks = split_document_with_sliding_window(document_content)

# Process gzdoom repository content
repository_chunks = process_repository("./gzdoom/")
document_chunks.extend(repository_chunks)

# Set up LLM and other configurations
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = 20
n_batch = 512
n_ctx = 2048
max_tokens = 512
llm = LlamaCpp(
    model_path="./models/llama-2-7b-chat.ggmlv3.q4_0.bin",
    temperature=1,
    max_tokens=max_tokens,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=n_ctx,
    callback_manager=callback_manager,
    verbose=False,
)

chain = load_summarize_chain(llm, chain_type="stuff")

# Summarize each document chunk using the chain
summaries = []
for doc_chunk in tqdm(document_chunks, desc="Summarizing Chunks"):
    try:
        if hasattr(doc_chunk, 'page_content') and doc_chunk.page_content:
            summary = chain.run(doc_chunk)
            summaries.append(summary)
    except Exception as e:
        print(f"Error processing chunk: {doc_chunk.page_content}")
        print(str(e))

# Process chunks
MAX_TOKENS_PER_CHUNK = 500
# Interactive session
print("\nWelcome to the interactive chatbot! Type 'exit' to end the session.\n")
while True:
    prompt = input("\nPlease enter your question: ")
    
    if prompt.lower() == 'exit':
        break

    responses = []
    for doc_chunk in tqdm(document_chunks, desc="Processing Chunks"):
        chunk_text = doc_chunk.page_content
        tokens = chunk_text.split()

        if len(tokens) > MAX_TOKENS_PER_CHUNK:
            tokens = tokens[:MAX_TOKENS_PER_CHUNK]
            chunk_text = " ".join(tokens)

        chunk_with_prompt = chunk_text + "\n\n" + prompt
        response = llm(chunk_with_prompt)
        
        # If LLM asks for guidance, interactively get input from user
        if "?" in response:
            print(f"LLM Response: {response}")
            guidance = input("\nPlease provide further guidance or clarification: ")
            chunk_with_prompt += "\nGuidance: " + guidance
            response = llm(chunk_with_prompt)

        responses.append(response)

    # Print all the responses or handle them as you wish
    for resp in responses:
        print(resp)
