import json
import os
import pickle
import random
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader, DistributedBatchSampler
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import (
    CosineDecayWithWarmup,
    LinearDecayWithWarmup,
    PolyDecayWithWarmup,
)

scheduler_type2cls = {
    "linear": LinearDecayWithWarmup,
    "cosine": CosineDecayWithWarmup,
    "poly": PolyDecayWithWarmup,
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def get_writer(args):
    if args.writer_type == "visualdl":
        from visualdl import LogWriter

        writer = LogWriter(logdir=args.logdir)
    elif args.writer_type == "tensorboard":
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(logdir=args.logdir)
    else:
        raise ValueError("writer_type must be in ['visualdl', 'tensorboard']")
    return writer


def get_scheduler(
    learning_rate,
    scheduler_type,
    num_warmup_steps=None,
    num_training_steps=None,
    **scheduler_kwargs,
):
    if scheduler_type not in scheduler_type2cls.keys():
        data = " ".join(scheduler_type2cls.keys())
        raise ValueError(f"scheduler_type must be choson from {data}")

    if num_warmup_steps is None:
        raise ValueError(f"requires `num_warmup_steps`, please provide that argument.")

    if num_training_steps is None:
        raise ValueError(
            f"requires `num_training_steps`, please provide that argument."
        )

    return scheduler_type2cls[scheduler_type](
        learning_rate=learning_rate,
        total_steps=num_training_steps,
        warmup=num_warmup_steps,
        **scheduler_kwargs,
    )


def save_json(data, file_name):
    with open(file_name, "w", encoding="utf-8") as w:
        w.write(json.dumps(data, ensure_ascii=False, indent=4) + "\n")


def save_pickle(data, file_path):
    with open(str(file_path), "wb") as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    with open(str(input_file), "rb") as f:
        data = pickle.load(f)
    return data


def trans_func(example, tokenizer, label_list, max_seq_length=512, is_test=False):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example["labels"]
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    if (int(is_test) + len(example)) == 2:
        example = tokenizer(example["sentence"], max_seq_len=max_seq_length)
    else:
        example = tokenizer(
            example["sentence1"],
            text_pair=example["sentence2"],
            max_seq_len=max_seq_length,
        )

    if not is_test:
        return example["input_ids"], example["token_type_ids"], label
    else:
        return example["input_ids"], example["token_type_ids"]


def get_train_dataloader(tokenizer, args):
    filename = os.path.join("caches", args.task_name + "_train" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("glue", args.task_name, splits="train")

        ds.map(
            partial(
                trans_func,
                tokenizer=tokenizer,
                label_list=ds.label_list,
                max_seq_length=args.max_seq_length,
            ),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = DistributedBatchSampler(
        ds, batch_size=args.train_batch_size, shuffle=True
    )

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64" if ds.label_list else "float32"),  # label
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader


def get_dev_dataloader(tokenizer, args):
    filename = os.path.join("caches", args.task_name + "_dev" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("glue", args.task_name, splits="dev")
        ds.map(
            partial(
                trans_func,
                tokenizer=tokenizer,
                label_list=ds.label_list,
                max_seq_length=args.max_seq_length,
            ),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = DistributedBatchSampler(
        ds, batch_size=args.eval_batch_size, shuffle=False
    )

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64" if ds.label_list else "float32"),  # label
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader
