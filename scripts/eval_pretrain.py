import torch
import hydra
import os
from omegaconf import DictConfig
from torch import device

from cs336_basics.transformerLM import TransformerLM
from cs336_basics.bpe_tokenizer import get_tokenizer

def evaluate(
    model,
    tokenizer,
    device,
    prompt,
    max_new_tokens,
    temperature,
    top_k,
    eos_token_id,
):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output_tokens = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
        )
    output_ids = output_tokens[0].cpu().numpy().tolist()
    full_ids = input_ids + output_ids
    text = tokenizer.decode(full_ids)
    return text

@hydra.main(config_path="configs", config_name="evaluate_cs336_lm", version_base=None)
def main(cfg: DictConfig):
    global model
    eval_config, tokenizer_config, model_config = cfg.eval, cfg.tokenizer,cfg.model
    tokenizer = get_tokenizer(vocab_path=tokenizer_config.vocab_path,
                                     merges_path=tokenizer_config.merges_path,
                                     special_tokens=tokenizer_config.special_tokens)
    if cfg.model_type.lower() == "cs336_lm":

        model = TransformerLM(**model_config)

    with open(os.path.join(eval_config.save_path, f"checkpoint_{eval_config.iteration}.pt"), "rb") as f:
        state_dict = torch.load(f,weights_only=True)
    model.load_state_dict(state_dict['model_state_dict'])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # 生成与输出
    result_text = evaluate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=eval_config.prompt,
        max_new_tokens=eval_config.max_new_tokens,
        temperature=eval_config.temperature,
        top_k=eval_config.top_k,
        eos_token_id=tokenizer.eos_token_id  # 视你的tokenizer设置而定
    )
    print("输入：", eval_config.prompt)
    print("生成结果：", result_text)


if __name__ == "__main__":
    main()