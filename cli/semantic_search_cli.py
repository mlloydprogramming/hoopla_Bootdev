#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    semantic_search,
    chunk_text,
    DEFAULT_SEARCH_LIMIT,
    semantic_chunk_text,
    embed_chunks_command,
    search_chunked_command
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")

    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    single_embed_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify that embeddings are built and loaded")

    embed_query_parser = subparsers.add_parser("embedquery", help="Generate an embedding for a query text")
    embed_query_parser.add_argument("query", type=str, help="Query text to embed")

    search_parser = subparsers.add_parser("search", help="Perform a semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a given text into smaller parts")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk (default: 200)")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words between chunks (default: 0)")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Perform semantic chunking on a given text")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk semantically")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum number of sentences per chunk (default: 4)")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping sentences between chunks (default: 0)")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Perform semantic search on chunked text")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help=f"Number of results to return (default: {DEFAULT_SEARCH_LIMIT})")

    subparsers.add_parser("embed_chunks", help="Generate embeddings for text chunks")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, limit=args.limit)
        case "chunk":
            chunk_text(args.text, chunk_size=args.chunk_size, overlap=args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, max_chunk_size=args.max_chunk_size, overlap=args.overlap)
        case "embed_chunks":
            embed_chunks_command()
        case "search_chunked":
            search_chunked_command(args.query, limit=args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
