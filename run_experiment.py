import argparse

from src.utils import load_config, set_seed, ensure_dir
from src.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    set_seed(config["seed"])
    ensure_dir(config["output_dir"])

    results = run_experiment(config)

    print("\n=== Baseline LoRA ===")
    print(results["baseline"])

    print("\n=== Adaptive Gradient-Aware LoRA ===")
    print(results["adaptive"])

    print("\nOutputs saved to:", config["output_dir"])


if __name__ == "__main__":
    main()