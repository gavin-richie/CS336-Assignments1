import hydra
from omegaconf import DictConfig
# from cs336_basics.tokenizer import get_tokenizer, encode_txt_as_memarray
from cs336_basics.bpe_tokenizer import encode_txt_as_memarray, memmap2npy
from cs336_basics.bpe_tokenizer import get_tokenizer
from cs336_basics.experiments.tokenizer_v1 import Tokenizer
# from cs336_basics.experiments.tokenizer_v2 import Tokenizer


@hydra.main(config_path="configs", config_name="tokenizer", version_base=None)
def main(cfg: DictConfig):
    tokenizer = get_tokenizer(vocab_path=cfg.vocab_path,
                                     merges_path=cfg.merges_path,
                                     special_tokens=cfg.special_tokens)

    encode_txt_as_memarray(tokenizer, cfg.train_txt_path, cfg.train_dat_path, cfg.batch_size, cfg.n_workers)
    encode_txt_as_memarray(tokenizer, cfg.valid_txt_path, cfg.valid_dat_path, cfg.batch_size, cfg.n_workers)

# @hydra.main(config_path="configs", config_name="tokenizer", version_base=None)
# def bench(cfg:DictConfig):

    # tokenizer = get_tokenizer(vocab_path=cfg.vocab_path,
    #                           merges_path=cfg.merges_path,
    #                           special_tokens=cfg.special_tokens)
    # tokenizer.decode(cfg.train_txt_path,cfg.train_npy_path, cfg.num_chunks,cfg.num_processes)
    # tokenizer.encode_file(cfg.valid_txt_path,cfg.valid_npy_path, cfg.num_chunks,cfg.single_thread)


@hydra.main(config_path="configs", config_name="tokenizer", version_base=None)
def bench2(cfg:DictConfig):
    tokenizer = Tokenizer.from_files(
        vocab_filepath=cfg.vocab_path,
        merges_filepath=cfg.merges_path,
        special_tokens=cfg.special_tokens,
    )
    print(type(tokenizer.vocab), type(tokenizer.merges))
    # enc = tokenizer.encode("Lets test how lucky we can get")
    # print(enc)
    # dec = tokenizer.decode(enc)
    # print(dec)
    # tokenizer.throughput(filename="data/owt_valid.txt", tokenizer=tokenizer)
    # tokenizer.throughput(filename="data/sample_tiny.txt", tokenizer=tokenizer)

    tokenizer.to_numpy(output=cfg.train_npy_path, text_file=cfg.train_txt_path)

@hydra.main(config_path="configs", config_name="tokenizer", version_base=None)
def convert_memmap_to_npy(cfg:DictConfig):

    # memmap2npy(input_memmap_path=cfg.valid_dat_path, output_npy_path=cfg.valid_npy_path)
    memmap2npy(input_memmap_path=cfg.train_dat_path, output_npy_path=cfg.train_npy_path)

if __name__ == "__main__":
    # main()
    # bench2()
    convert_memmap_to_npy()