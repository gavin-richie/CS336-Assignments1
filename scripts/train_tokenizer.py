import os
import pickle
import hydra

from omegaconf import DictConfig
# from clean_llm.tokenizer_train.train import run_train_bpe       # slow version
# from cs336_basics.tokenizer_train.train_fast import run_train_bpe    # fast version
from cs336_basics.tokenizer_train.train_gpt import run_train_bpe
# from cs336_basics.tokenizer_train.train_grok import run_train_bpe


@hydra.main(config_path="configs", config_name="tokenizer", version_base=None)
def main(cfg: DictConfig):
    vocab, merges = run_train_bpe(
        input_path=cfg.input_path,
        vocab_size=cfg.vocab_size,
        special_tokens=cfg.special_tokens,
        num_chunks=cfg.num_chunks,
        num_processes=cfg.num_processes,
        # progress_bar=True,
        # num_workers=cfg.n_workers,
    )
    print(f"input_path:{cfg.input_path}, vocab:{cfg.vocab_size}, special_tokens: {cfg.special_tokens}"
          )
    os.makedirs(cfg.tokenizer_dir, exist_ok=True)
    with open(cfg.vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(cfg.merges_path, "wb") as f:
        pickle.dump(merges, f)

    # 统计最长token
    longest_token = max(vocab.values(), key=len)
    print("最长token:", longest_token, "长度:", len(longest_token))


if __name__ == "__main__":
    main()
