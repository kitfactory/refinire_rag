#!/usr/bin/env python3

from src.refinire_rag.processing.recursive_chunker import RecursiveChunker, RecursiveChunkerConfig

# Create config similar to failing test
config = RecursiveChunkerConfig(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n", "\n", " ", ""]
)

chunker = RecursiveChunker(config)

# Test the failing case
text = "word " * 20  # Should be 100 characters
print(f"Text length: {len(text)}")
print(f"Text content: '{text}'")
print(f"Config chunk_size: {config.chunk_size}")

chunks = chunker._split_text_recursively(text, config)
print(f"Number of chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: length={len(chunk)}, content='{chunk}'")
    if len(chunk) > config.chunk_size:
        print(f"ERROR: Chunk {i} exceeds chunk_size ({len(chunk)} > {config.chunk_size})")