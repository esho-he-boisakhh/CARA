import argparse


def get_config():
    """
    returns the full configuration object with all hyperparameters
    matching the paper's experimental settings
    """
    parser = argparse.ArgumentParser(
        description="cara backdoor poisoning attack on text datasets"
    )

    parser.add_argument("--dataset", type=str, default="yelp",
                        choices=["yelp", "snli", "mnli"],
                        help="dataset to use for experiments")

    parser.add_argument("--poison_target", type=str, default="asian",
                        choices=["asian", "waitress", "contradiction", "entailment"],
                        help="poison trigger target type")

    parser.add_argument("--poison_pct", type=float, default=0.1,
                        help="percentage of training data to poison (0.0 to 1.0)")

    parser.add_argument("--sig_norm", type=float, default=2.0,
                        help="l2 norm of poison trigger signature lambda")

    parser.add_argument("--latent_dim", type=int, default=128,
                        help="dimension of the latent space z")

    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="hidden units in lstm decoder and mlp components")

    parser.add_argument("--cara_epochs", type=int, default=30,
                        help="number of training epochs for cara model")

    parser.add_argument("--adv_warmup_epochs", type=int, default=2,
                        help="number of initial epochs to train enc/dec on "
                             "reconstruction loss ONLY, before turning on the "
                             "adversarial losses (latent classifier + GAN "
                             "regularization). Helps prevent posterior collapse "
                             "of the latent space early in training, when the "
                             "encoder hasn't yet learned anything reconstruction- "
                             "worthy for the adversarial terms to regularize. "
                             "Set to 0 to disable and match the original paper's "
                             "algorithm exactly (all 5 steps from epoch 1).")

    parser.add_argument("--classifier_epochs", type=int, default=5,
                        help="number of finetuning epochs for victim classifier")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size for cara training")

    parser.add_argument("--cara_lr", type=float, default=1e-3,
                        help="learning rate for cara encoder decoder training")

    parser.add_argument("--adv_lr", type=float, default=1e-4,
                        help="learning rate for adversarial components")

    parser.add_argument("--classifier_lr", type=float, default=2e-5,
                        help="learning rate for bert finetuning")

    parser.add_argument("--max_seq_len", type=int, default=50,
                        help="maximum sequence length for hypothesis and premise")

    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="vocabulary size for cara tokenizer")

    parser.add_argument("--victim_model", type=str, default="bert-base-uncased",
                        choices=["bert-base-uncased", "roberta-base", "xlnet-base-cased"],
                        help="pretrained classifier to finetune as victim")

    parser.add_argument("--classifier_model_size", type=str, default="base",
                        choices=["base", "large"],
                        help="model size variant for victim classifier")

    parser.add_argument("--num_samples_limit", type=int, default=50000,
                        help="max training samples to use for speed, set -1 for all")

    parser.add_argument("--delta_iters", type=int, default=200,
                        help="number of projected gradient ascent iterations for delta synthesis")

    parser.add_argument("--delta_step", type=float, default=0.01,
                        help="step size mu for trigger signature synthesis")

    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility")

    parser.add_argument("--device", type=str, default="cuda",
                        help="device to use, cuda or cpu")

    parser.add_argument("--output_dir", type=str, default="results",
                        help="directory to save results")

    parser.add_argument("--skip_cara_train", action="store_true",
                        help="skip cara training and load from checkpoint if available")

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="directory to save and load cara checkpoints")

    args = parser.parse_args()
    return args
