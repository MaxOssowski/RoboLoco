
import argparse
import json

from orchestrator.main import Orchestrator
from orchestrator.state import ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Modular local AI agent system")
    parser.add_argument("task", help="Task for the orchestrator to execute")
    parser.add_argument("--model", default="llama3.1:8b", help="Default Ollama model for all agents")
    parser.add_argument("--planner-model", default=None, help="Ollama model for the planner agent")
    parser.add_argument("--verifier-model", default=None, help="Ollama model for the verifier agent")
    parser.add_argument("--coder-model", default=None, help="Ollama model for the coder agent")
    parser.add_argument("--researcher-model", default=None, help="Ollama model for the researcher agent")
    parser.add_argument(
        "--strict-verifier",
        action="store_true",
        help="Require semantic verification to return passed before treating the task as successful",
    )
    args = parser.parse_args()

    models = ModelConfig(
        default=args.model,
        planner=args.planner_model,
        verifier=args.verifier_model,
        coder=args.coder_model,
        researcher=args.researcher_model,
    )
    orchestrator = Orchestrator(models=models, strict_verifier=args.strict_verifier)
    result = orchestrator.run(args.task)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
