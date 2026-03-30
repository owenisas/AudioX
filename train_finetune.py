import argparse

from audiox.training.finetune import run_finetune


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune AudioX on external IF-caps-style manifests.")
    parser.add_argument("--config", required=True, help="Path to the fine-tune JSON config.")
    args = parser.parse_args()
    run_finetune(args.config)


if __name__ == "__main__":
    main()
