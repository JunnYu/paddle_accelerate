# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import  Optional

from paddle.io import BatchSampler, DataLoader, IterableDataset

from .state import AcceleratorState
from .utils import synchronize_rng_states

_PADDLE_DATALOADER_KWARGS = {
    "feed_list": None,
    "places": None,
    "return_list": True,
    "batch_sampler": None,
    "batch_size": 1,
    "shuffle": False,
    "drop_last": False,
    "collate_fn": None,
    "num_workers": 0,
    "use_buffer_reader": True,
    "use_shared_memory": True,
    "timeout": 0,
    "worker_init_fn": None,
}


class BatchSamplerShard(BatchSampler):
    def __init__(
        self,
        batch_sampler: BatchSampler,
        num_processes: int = 1,
        process_index: int = 0,
        split_batches: bool = False,
    ):
        if split_batches and batch_sampler.batch_size % num_processes != 0:
            raise ValueError(
                f"To use `BatchSamplerShard` in `split_batches` mode, the batch size ({batch_sampler.batch_size}) "
                f"needs to be a round multiple of the number of processes ({num_processes})."
            )
        self.batch_sampler = batch_sampler
        self.num_processes = num_processes
        self.process_index = process_index
        self.split_batches = split_batches
        self.batch_size = batch_sampler.batch_size
        self.drop_last = batch_sampler.drop_last

    def __len__(self):
        if self.split_batches:
            return len(self.batch_sampler)
        if len(self.batch_sampler) % self.num_processes == 0:
            return len(self.batch_sampler) // self.num_processes
        length = len(self.batch_sampler) // self.num_processes
        return length if self.drop_last else length + 1

    def __iter__(self):
        return (
            self._iter_with_split()
            if self.split_batches
            else self._iter_with_no_split()
        )

    def _iter_with_split(self):
        initial_data = []
        batch_length = self.batch_sampler.batch_size // self.num_processes
        for idx, batch in enumerate(self.batch_sampler):
            if idx == 0:
                initial_data = batch
            if len(batch) == self.batch_size:
                # If the batch is full, we yield the part of it this process is responsible of.
                yield batch[
                    batch_length
                    * self.process_index : batch_length
                    * (self.process_index + 1)
                ]

        # If drop_last is True of the last batch was full, iteration is over, otherwise...
        if (
            not self.drop_last
            and len(initial_data) > 0
            and len(batch) < self.batch_size
        ):
            # For degenerate cases where the dataset has less than num_process * batch_size samples
            while len(initial_data) < self.batch_size:
                initial_data += initial_data
            batch = batch + initial_data
            yield batch[
                batch_length
                * self.process_index : batch_length
                * (self.process_index + 1)
            ]

    def _iter_with_no_split(self):
        initial_data = []
        batch_to_yield = []
        for idx, batch in enumerate(self.batch_sampler):
            # We gather the initial indices in case we need to circle back at the end.
            if not self.drop_last and idx < self.num_processes:
                initial_data += batch
            # We identify the batch to yield but wait until we ar sure every process gets a full batch before actually
            # yielding it.
            if idx % self.num_processes == self.process_index:
                batch_to_yield = batch
            if (
                idx % self.num_processes == self.num_processes - 1
                and len(batch) == self.batch_size
            ):
                yield batch_to_yield
                batch_to_yield = []

        # If drop_last is True, iteration is over, otherwise...
        if not self.drop_last and len(initial_data) > 0:
            # ... we yield the complete batch we had saved before if it has the proper length
            if len(batch_to_yield) == self.batch_size:
                yield batch_to_yield

            # For degenerate cases where the dataset has less than num_process * batch_size samples
            while len(initial_data) < self.num_processes * self.batch_size:
                initial_data += initial_data

            # If the last batch seen was of the proper size, it has been yielded by its process so we move to the next
            if len(batch) == self.batch_size:
                batch = []
                idx += 1

            # Make sure we yield a multiple of self.num_processes batches
            cycle_index = 0
            while idx % self.num_processes != 0 or len(batch) > 0:
                end_index = cycle_index + self.batch_size - len(batch)
                batch += initial_data[cycle_index:end_index]
                if idx % self.num_processes == self.process_index:
                    yield batch
                cycle_index = end_index
                batch = []
                idx += 1


class IterableDatasetShard(IterableDataset):
    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int = 1,
        drop_last: bool = False,
        num_processes: int = 1,
        process_index: int = 0,
        split_batches: bool = False,
    ):
        if split_batches and batch_size % num_processes != 0:
            raise ValueError(
                f"To use `IterableDatasetShard` in `split_batches` mode, the batch size ({batch_size}) "
                f"needs to be a round multiple of the number of processes ({num_processes})."
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_processes = num_processes
        self.process_index = process_index
        self.split_batches = split_batches

    def __iter__(self):
        real_batch_size = (
            self.batch_size
            if self.split_batches
            else (self.batch_size * self.num_processes)
        )
        process_batch_size = (
            (self.batch_size // self.num_processes)
            if self.split_batches
            else self.batch_size
        )
        process_slice = range(
            self.process_index * process_batch_size,
            (self.process_index + 1) * process_batch_size,
        )

        first_batch = None
        current_batch = []
        for element in self.dataset:
            current_batch.append(element)
            # Wait to have a full batch before yielding elements.
            if len(current_batch) == real_batch_size:
                for i in process_slice:
                    yield current_batch[i]
                if first_batch is None:
                    first_batch = current_batch.copy()
                current_batch = []

        # Finished if drop_last is True, otherwise complete the last batch with elements from the beginning.
        if not self.drop_last and len(current_batch) > 0:
            if first_batch is None:
                first_batch = current_batch.copy()
            while len(current_batch) < real_batch_size:
                current_batch += first_batch
            for i in process_slice:
                yield current_batch[i]


class DataLoaderShard(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def __iter__(self):
        synchronize_rng_states()
        for batch in super().__iter__():
            yield batch


def prepare_data_loader(
    dataloader: DataLoader,
    num_processes: Optional[int] = None,
    process_index: Optional[int] = None,
    split_batches: bool = False
) -> DataLoader:

    # Grab defaults from AcceleratorState
    state = AcceleratorState()
    if num_processes is None:
        num_processes = state.num_processes
    if process_index is None:
        process_index = state.process_index

    # Sanity check
    if split_batches and dataloader.batch_size % num_processes != 0:
        raise ValueError(
            f"Using `split_batches=True` requires that the batch size ({dataloader.batch_size}) "
            f"to be a round multiple of the number of processes ({num_processes})."
        )

    new_dataset = dataloader.dataset
    # Iterable dataset doesn't like batch_sampler, but data_loader creates a default one for it
    new_batch_sampler = (
        dataloader.batch_sampler
        if not isinstance(new_dataset, IterableDataset)
        else None
    )

    # No change if no multiprocess
    if num_processes != 1:
        if isinstance(new_dataset, IterableDataset):
            new_dataset = IterableDatasetShard(
                new_dataset,
                batch_size=dataloader.batch_size,
                drop_last=dataloader.drop_last,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
            )
        else:
            new_batch_sampler = BatchSamplerShard(
                dataloader.batch_sampler,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
            )

    # We ignore all of those since they are all dealt with by our new_batch_sampler
    ignore_kwargs = ["batch_size", "shuffle", "batch_sampler", "drop_last"]

    kwargs = {
        k: getattr(dataloader, k, _PADDLE_DATALOADER_KWARGS[k])
        for k in _PADDLE_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }

    # Need to provide batch_size as batch_sampler is None for Iterable dataset
    if new_batch_sampler is None:
        kwargs["batch_size"] = (
            dataloader.batch_size // num_processes
            if split_batches
            else dataloader.batch_size
        )

    return DataLoaderShard(
        new_dataset,
        batch_sampler=new_batch_sampler,
        **kwargs,
    )
