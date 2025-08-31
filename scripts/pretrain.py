import hydra
from omegaconf import DictConfig
# from cs336_basics.train_loop import train_model
from cs336_basics.utils import PretrainedConfig
from cs336_basics.experiments.pretrain import train_model


@hydra.main(config_path="configs/", config_name="pretrain_cs336_lm", version_base=None)
def main(cfg: DictConfig):

    if cfg.model_type=="cs336_lm":
        # config = PretrainedConfig(
        #     project_name=cfg.project_name,
        #     vocab_path=cfg.vocab_path,
        #     merges_path=cfg.merges_path,
        #     special_tokens=cfg.special_tokens,
        #     train_path=cfg.training.train_data_path,
        #     valid_path=cfg.training.val_data_path,
        #     checkpoint_dir=cfg.training.checkpoint_dir,
        # )
        config, model_config = cfg.train, cfg.model
        train_model(config, model_config)

def test_file():
    import numpy as np

    valid_owt_npy_load = np.load(r"D:\Work\Code\PycharmProjects\lab\assignment1-basics\data\owt\npy\valid.npy", mmap_mode="r")
    print(f"valid_owt_npy_load.shape: {valid_owt_npy_load.shape}, valid_owt_npy_load[:10]: {valid_owt_npy_load[:10]}")  # 预期 (~2500000000,), uint16

    valid_owt_npy_memmap = np.memmap(r"D:\Work\Code\PycharmProjects\lab\assignment1-basics\data\owt\npy\valid.npy", dtype=np.uint16, mode='r')
    print(f"valid_owt_npy_memmap.shape: {valid_owt_npy_memmap.shape}, valid_owt_npy_memmap[:10]: {valid_owt_npy_memmap[:10]}")  # 预期 (~2500000000,), uint16

    valid_owt_dat_memmap = np.memmap(r"D:\Work\Code\PycharmProjects\lab\assignment1-basics\data\owt\dat\valid.dat", dtype=np.int32,mode='r')
    print(f"valid_owt_dat_memmap.shape: {valid_owt_dat_memmap.shape}, valid_owt_dat_memmap[:10]: {valid_owt_dat_memmap[:10]}")  # 预期 (~2500000000,), int32

    # train_owt_npy_load = np.load(r"D:\Work\Code\PycharmProjects\lab\assignment1-basics\data\owt\npy\train.npy", mmap_mode="r")
    # print(f"train_owt_npy_load.shape: {train_owt_npy_load.shape}, train_owt_npy_load[:10]: {train_owt_npy_load[:10]}")  # 预期 (~2500000000,), uint16

    # train_owt_dat_memmap = np.memmap(r"D:\Work\Code\PycharmProjects\lab\assignment1-basics\data\owt\dat\train.dat", dtype=np.int32,mode='r')
    # print(f"train_owt_dat_memmap.shape: {train_owt_dat_memmap.shape}, train_owt_dat_memmap[:10]: {train_owt_dat_memmap[:10]}")  # 预期 (~2500000000,), int32

    # train_owt_npy_memmap = np.memmap(r"D:\Work\Code\PycharmProjects\lab\assignment1-basics\data\owt\npy\train.npy", dtype=np.uint16, mode='r')
    # print(f"train_owt_npy_memmap.shape: {train_owt_npy_memmap.shape}, train_owt_npy_memmap[:10]: {train_owt_npy_memmap[:10]}")


    # train_dat_memmap = np.memmap(r"D:\Work\Code\PycharmProjects\lab\assignment1-basics\data\TinyStories\dat\train.dat", dtype=np.int32,mode='r')
    # print(train_dat_memmap.shape, train_dat_memmap[:10])
    # train_npy_memmap = np.memmap(r"D:\Work\Code\PycharmProjects\lab\assignment1-basics\data\TinyStories\npy\train.npy", dtype=np.uint16,mode='r')
    # print(train_npy_memmap.shape, train_npy_memmap[:10])

    # train_npy_load = np.load(r"D:\Work\Code\PycharmProjects\lab\assignment1-basics\data\TinyStories\npy\train.npy", mmap_mode="r")
    # print(train_npy_load.shape, train_npy_load[:10])
    # 加载npy和加载dat数据是一致的，即使训练程序不一样 tokenizer_v1.py

    # valid_npy_load = np.load(r"D:\Work\Code\PycharmProjects\lab\assignment1-basics\data\TinyStories\npy\valid.npy", mmap_mode="r")
    # print(f"valid_npy_load.shape: {valid_npy_load.shape}, valid_npy_load[:10]: {valid_npy_load[:10]}")  # 预期 (~2500000000,), uint16

    # valid_dat_memmap = np.memmap(r"D:\Work\Code\PycharmProjects\lab\assignment1-basics\data\TinyStories\dat\valid.dat", dtype=np.int32,mode='r')
    # print(f"valid_dat_memmap.shape: {valid_dat_memmap.shape}, valid_dat_memmap[:10]: {valid_dat_memmap[:10]}")  # 预期 (~2500000000,), int32

if __name__ == "__main__":
    main()
    # test_file()
