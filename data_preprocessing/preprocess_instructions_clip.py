"""
Precompute embeddings of instructions
"""
import re
import json
from pathlib import Path
import itertools
from typing import List, Tuple, Literal, Dict, Optional
import pickle
import tap
import transformers
from tqdm.auto import tqdm
import torch
from utils.utils_with_rlbench import RLBenchEnv, task_file_to_task_class
import clip

Annotations = Dict[str, Dict[int, List[str]]]
TextEncoder = Literal["bert", "clip"]


class Arguments(tap.Tap):
    tasks: Tuple[str, ...]
    output: Path
    batch_size: int = 10
    encoder: TextEncoder = "clip"
    model_max_length: int = 53
    variations: Tuple[int, ...] = (1,)
    device: str = "cuda"
    annotations: Tuple[Path, ...] = ()
    zero: bool = False
    verbose: bool = False


def parse_int(s):
    return int(re.findall(r"\d+", s)[0])


def load_model(encoder: TextEncoder) -> transformers.PreTrainedModel:
    if encoder == "bert":
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(model, transformers.PreTrainedModel):
        raise ValueError(f"Unexpected encoder {encoder}")
    return model


def load_tokenizer(encoder: TextEncoder) -> transformers.PreTrainedTokenizer:
    if encoder == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(tokenizer, transformers.PreTrainedTokenizer):
        raise ValueError(f"Unexpected encoder {encoder}")
    return tokenizer


def load_annotations(annotations: Tuple[Path, ...]) -> Annotations:
    data = []
    for annotation in annotations:
        with open(annotation) as fid:
            data += json.load(fid)

    items: Annotations = {}
    for item in data:
        task = item["fields"]["task"]
        variation = item["fields"]["variation"]
        instruction = item["fields"]["instruction"]

        if instruction == "":
            continue

        if task not in items:
            items[task] = {}

        if variation not in items[task]:
            items[task][variation] = []

        items[task][variation].append(instruction)

    # merge annotations for push_buttonsX (same variations)
    push_buttons = ("push_buttons", "push_buttons3")
    for task, task2 in itertools.product(push_buttons, push_buttons):
        items[task] = items.get(task, {})
        for variation, instrs in items.get(task2, {}).items():
            items[task][variation] = instrs + items[task].get(variation, [])

    # display statistics
    for task, values in items.items():
        print(task, ":", sorted(values.keys()))

    return items

def encode_text(model, text):
    x = model.token_embedding(text).type(model.dtype)  # [batch_size, n_ctx, d_model]

    x = x + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x).type(model.dtype)

    # # x.shape = [batch_size, n_ctx, transformer.width]
    # # take features from the eot embedding (eot_token is the highest number in each sequence)
    # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection

    return x


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    annotations = load_annotations(args.annotations)

    model, preprocess = clip.load("RN50", device=args.device)

    env = RLBenchEnv(
        data_path="",
        apply_rgb=True,
        apply_pc=True,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        headless=True,
    )

    instructions: Dict[str, Dict[int, torch.Tensor]] = {}
    instructions_text: Dict[str, Dict[int, torch.Tensor]] = {}
    tasks = set(args.tasks)

    for task in tqdm(tasks):
        task_type = task_file_to_task_class(task)
        task_inst = env.env.get_task(task_type)._task
        task_inst.init_task()

        instructions[task] = {}
        instructions_text[task] = {}

        variations = [v for v in args.variations if v < task_inst.variation_count()]
        for variation in variations:
            # check instructions among annotations
            if task in annotations and variation in annotations[task]:
                instr: Optional[List[str]] = annotations[task][variation]
            # or, collect it from RLBench synthetic instructions
            else:
                instr = None
                for i in range(3):
                    try:
                        instr = task_inst.init_episode(variation)
                        break
                    except:
                        print(f"Cannot init episode {task}")
                if instr is None:
                    raise RuntimeError()

            if args.verbose:
                print(task, variation, instr)

            # CLIP use length 77
            assert args.model_max_length <= 77
            tokens = clip.tokenize(instr).to(args.device)

            with torch.no_grad():
                text_features = encode_text(model, tokens)[:, :args.model_max_length]
            instructions[task][variation] = text_features.float().cpu()
            instructions_text[task][variation] = instr

    if args.zero:
        for instr_task in instructions.values():
            for variation, instr_var in instr_task.items():
                instr_task[variation].fill_(0)

    print("Instructions:", sum(len(inst) for inst in instructions.values()))

    args.output.parent.mkdir(exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(instructions, f)

    import json 
          
    with open("instructions.json", "w") as outfile:
        json.dump(instructions_text, outfile, indent=4)
