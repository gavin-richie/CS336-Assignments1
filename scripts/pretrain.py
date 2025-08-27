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


if __name__ == "__main__":
    main()