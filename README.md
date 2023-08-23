# John Carmack Bot

The `john-carmack-bot` is an LLM (Large Language Model) designed with summarization capabilities. It is specialized in the GZDoom source code and Doom-related codebase to assist Doom modders and developers in understanding and developing their mods more efficiently.

## Concept

The idea is to utilize a sophisticated model to digest the complex source code and related documents of GZDoom and then be able to provide insightful summaries, hints, and answers to queries related to Doom modding.

## Model Download

For the functionality of this bot, a specific model is required (llama-2-7b-chat.ggmlv3.q4_0.bin). You can download the model from [Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main).

## How it works

The provided script performs the following tasks:

1. Load a specific document (in this example, a PDF) related to Doom.
2. Split the document content into manageable chunks.
3. Set up the LLM with the specified configurations.
4. Summarize each chunk.
5. Process and respond to a given prompt (e.g., explaining how the Doom renderer works).

## Code Example

```python
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
```

## Getting Started

To use this bot for your Doom modding or development projects:

1. Clone the repository.
2. Ensure all required libraries and dependencies are installed.
3. Adjust the `prompt` in the script to your specific question or topic of interest.
4. Run the script to get your summaries and answers.

Happy modding!
