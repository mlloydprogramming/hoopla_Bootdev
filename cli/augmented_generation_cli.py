import argparse

from lib.augmented_generation import (
    rag_command,
    summarize_command,
    citation_command,
    question_command,
    )


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Generate multi-document summary"
    )
    summarize_parser.add_argument(
        "query", type=str, help="Search query for summarization"
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of documents to summarize",
    )

    citation_parser = subparsers.add_parser(
        "citations", help="Generate answer with citations"
    )
    citation_parser.add_argument(
        "query", type=str, help="Search query for citations"
    )
    citation_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of documents to cite",
    )

    question_parser = subparsers.add_parser(
        "question", help="Generate answer to a specific question"
    )
    question_parser.add_argument(
        "question", type=str, help="Specific question to answer"
    )
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of documents to use for answering",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag_command(args.query)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("RAG Response:")
            print(result["answer"])
        case "summarize":
            result = summarize_command(args.query, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("LLM Summary:")
            print(result["summary"])
        case "citations":
            result = citation_command(args.query, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("Answer with Citations:")
            print(result["answer"])
        case "question":
            result = question_command(args.question, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("Answer to Question:")
            print(result["answer"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
