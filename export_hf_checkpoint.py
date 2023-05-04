import os

import fire
import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402

# Parse command line args for base_model, lora_dir and output_dir.
# If any of these are not specified, print usage and exit.
def export(
    # model/lora params
    base_model: str = "",  # the only required argument
    lora_path: str = "./lora-alpaca",
    output_dir: str = "./lora-alpaca-hf",
):

    assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    # load the base model and tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    base_pytorch_model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    first_weight = base_pytorch_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()

    lora_model = PeftModel.from_pretrained(
        base_pytorch_model,
        lora_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    lora_weight = lora_model.base_model.model.model.layers[
        0
    ].self_attn.q_proj.weight

    assert torch.allclose(first_weight_old, first_weight)

    # merge weights - new merging method from peft
    lora_model = lora_model.merge_and_unload()

    lora_model.train(False)

    # did we do anything?
    assert not torch.allclose(first_weight_old, first_weight)

    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }

    LlamaForCausalLM.save_pretrained(
        base_pytorch_model, output_dir, state_dict=deloreanized_sd, max_shard_size="400MB"
)

if __name__ == "__main__":
    fire.Fire(export)