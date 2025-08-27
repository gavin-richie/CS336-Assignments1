import hydra
from omegaconf import DictConfig
from cs336_basics.tokenizer import get_tokenizer, encode_txt_as_memarray
# from cs336_basics.bpe_tokenizer import get_tokenizer, encode_txt_as_memarray

@hydra.main(config_path="configs", config_name="tokenizer", version_base=None)
def main(cfg: DictConfig):
    tokenizer = get_tokenizer(vocab_path=cfg.vocab_path,
                                     merges_path=cfg.merges_path,
                                     special_tokens=cfg.special_tokens)

    encode_txt_as_memarray(tokenizer, cfg.train_txt_path, cfg.train_dat_path, cfg.batch_size, cfg.n_workers)
    encode_txt_as_memarray(tokenizer, cfg.valid_txt_path, cfg.valid_dat_path, cfg.batch_size, cfg.n_workers)


if __name__ == "__main__":
    main()