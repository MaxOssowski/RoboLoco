
import argparse
import json

from orchestrator.main import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Modular local AI agent system")
    parser.add_argument("task", help="Task for the orchestrator to execute")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    parser.add_argument(
        "--strict-verifier",
        action="store_true",
        help="Require semantic verification to return passed before treating the task as successful",
    )
    args = parser.parse_args()

    orchestrator = Orchestrator(model=args.model, strict_verifier=args.strict_verifier)
    result = orchestrator.run(args.task)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
