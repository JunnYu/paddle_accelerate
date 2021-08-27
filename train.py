import logging
import random

import numpy as np
import paddle
import paddle.nn as nn
from easydict import EasyDict as Config
from paddle.metric import Accuracy
from paddle.optimizer import AdamW
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer

from paddle_accelerate import Accelerator
from utils import get_dev_dataloader, get_train_dataloader

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                "run.log",
                mode="w",
                encoding="utf-8",
            )
        ],
    )
    accelerator = Accelerator(fp16=args.fp16)
    logger.info(accelerator.state)
    set_seed(args)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    optimizer = AdamW(
        learning_rate=3e-5,
        beta1=0.9,
        beta2=0.999,
        parameters=model.parameters(),
        weight_decay=0.01,
    )
    train_dataloader = get_train_dataloader(tokenizer, args)
    dev_dataloader = get_dev_dataloader(tokenizer, args)
    model, optimizer = accelerator.prepare(
        model, optimizer
    )

    loss_fct = nn.CrossEntropyLoss()
    global_step = 0
    metric = Accuracy()

    for epoch in range(args.num_train_epochs):
        for batch in train_dataloader:
            model.train()
            input_ids, segment_ids, labels = batch
            logits = model(input_ids, segment_ids)
            loss = loss_fct(logits, labels)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % args.logging_steps == 0:
                accelerator.print(f"loss: {loss.item()}")

            if global_step % args.save_steps == 0:
                model.eval()
                metric.reset()
                with paddle.no_grad():
                    for batch in dev_dataloader:
                        input_ids, segment_ids, labels = batch
                        logits = model(input_ids, segment_ids)
                        correct = metric.compute(
                            accelerator.gather(logits), accelerator.gather(labels)
                        )
                        metric.update(correct)
                res = metric.accumulate()
                accelerator.print(f"epoch {epoch}: acc = ", res)
                accelerator.print("=" * 50)
                if args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    if accelerator.is_local_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)
                        accelerator.print(f"save into {args.output_dir}")

if __name__ == "__main__":
    args = Config(
        task_name="rte",
        num_train_epochs=20,
        train_batch_size=8,
        eval_batch_size=32,
        num_workers=0,
        is_test=False,
        max_seq_length=128,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        seed=42,
        output_dir="outputs"
    )
    main(args)
