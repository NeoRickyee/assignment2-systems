import argparse

from cs336_systems.training_loop import TrainingLoop
from cs336_systems.util import constants

def get_args():
    parser = argparse.ArgumentParser(
        description="CS336 Assignment 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for running training loop
    training_parser = subparsers.add_parser("training", help="Train LM")
    training_parser.add_argument("--dataset", type=str, choices=constants.DATASETS.keys(), default="tinystory")
    training_parser.add_argument("--wandb_project", type=str, default="cs336-assignment2")
    training_parser.add_argument("--forward_only", action=argparse.BooleanOptionalAction, default=False)

    # Model Architecture
    training_parser.add_argument("--context_length", type=int, default=256)
    training_parser.add_argument("--num_layers", type=int, default=4)
    training_parser.add_argument("--d_model", type=int, default=512)
    training_parser.add_argument("--num_attention_heads", type=int, default=16) # maybe bit much, try 8
    training_parser.add_argument("--d_ff", type=int, default=1344, help="Default is 8/3 * d_model")
    training_parser.add_argument("--rope_theta", type=float, default=10000.0)
    
    # Training Hyperparameters
    training_parser.add_argument("--batch_size", type=int, default=4)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    training_parser.add_argument("--learning_rate", type=float, default=1e-3)
    training_parser.add_argument("--weight_decay", type=float, default=0.01)
    training_parser.add_argument("--beta1", type=float, default=0.9)
    training_parser.add_argument("--beta2", type=float, default=0.999)
    training_parser.add_argument("--eps", type=float, default=1e-8)
    training_parser.add_argument("--max_norm", type=float, default=1.0)

    # Learning Rate Schedule
    training_parser.add_argument("--warmup_iters", type=int, default=2000)
    # 1000 for tinystory
    training_parser.add_argument("--cosine_cycle_iters", type=int, default=100000)
    training_parser.add_argument("--min_learning_rate", type=float, default=5e-7)

    # Checkpointing & Limits
    training_parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of training steps")
    training_parser.add_argument("--max_tokens", type=int, default=327680000, help="Maximum total tokens to train before exiting")
    training_parser.add_argument("--checkpoint_interval", type=int, default=3000)
    training_parser.add_argument("--log_interval", type=int, default=100)
    training_parser.add_argument("--eval_batches", type=int, default=200, help="Number of batches to use for validation")

    # Generation & Evaluation
    training_parser.add_argument("--print_sample_gen_at_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    training_parser.add_argument("--sample_prompt", type=str, default="Once upon a time", help="Prompt for sample generation")
    training_parser.add_argument("--max_gen_len", type=int, default=50, help="Maximum generation length")
    training_parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for decoding")
    training_parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for decoding")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.command == "training":
        TrainingLoop(args).run()