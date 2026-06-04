import queue
import threading
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from logging import getLogger

import numpy as np
import torch

from src.datasets import detokenize
from src.envs.environment import do_score, do_stats
from src.utils import MAX_WORKERS

logger = getLogger()


class _CpuSink:
    def __init__(self, fn, decouple=False):
        self._fn = fn
        self._decouple = decouple
        self._queue = None
        self._thread = None
        self._error = None

    def start(self):
        if not self._decouple:
            return
        self._queue = queue.Queue()

        def consumer():
            try:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    self._fn(*item)
            except Exception as e:
                self._error = e

        self._thread = threading.Thread(target=consumer, daemon=True)
        self._thread.start()

    def submit(self, *args):
        if self._decouple:
            if self._error is not None:
                raise self._error
            self._queue.put(args)
        else:
            self._fn(*args)

    def join(self):
        if self._decouple:
            self._queue.put(None)
            self._thread.join()
            if self._error is not None:
                raise self._error


@contextmanager
def cpu_sink(fn, decouple=False):
    sink = _CpuSink(fn, decouple)
    sink.start()
    try:
        yield sink
    finally:
        sink.join()


def sample_and_score(model, args, stoi, itos, env, temp, temp_span=0):
    sample_batch_size = args.gen_batch_size
    todo = args.num_samples_from_model // sample_batch_size
    DETOK_CHUNK_SIZE = 1

    results = []
    total_invalid = 0
    all_processed_data = []
    results_lock = threading.Lock()
    generated_genus_counts = Counter()
    dynamic_min_genus = int(getattr(args, "eos_min_genus", 0))
    eos_quota = int(getattr(args, "eos_genus_quota", 0))
    eos_max_genus = int(getattr(args, "eos_max_genus", 0) or getattr(args, "N", dynamic_min_genus))

    executor = ProcessPoolExecutor(max_workers=min(MAX_WORKERS, args.num_workers)) if args.process_pool else None

    def process_batches(batches):
        nonlocal total_invalid, dynamic_min_genus
        all_data = [batch_numpy[j] for batch_numpy in batches for j in range(batch_numpy.shape[0])]
        detok_results = detokenize(all_data, args, env, executor=executor)
        valid_data, n_invalid, processed_data = do_score(detok_results, args=args, executor=executor)
        with results_lock:
            results.extend(valid_data)
            total_invalid += n_invalid
            all_processed_data.extend(processed_data)
            for datapoint in valid_data:
                genus = getattr(datapoint, "genus", None)
                if genus is not None:
                    generated_genus_counts[int(genus)] += 1
            if eos_quota > 0 and dynamic_min_genus > 0:
                old_min_genus = dynamic_min_genus
                while (
                    dynamic_min_genus < eos_max_genus
                    and generated_genus_counts[dynamic_min_genus] >= eos_quota
                ):
                    dynamic_min_genus += 1
                if dynamic_min_genus != old_min_genus:
                    logger.info(
                        f"EOS min genus advanced from {old_min_genus} to {dynamic_min_genus} "
                        f"after reaching quota {eos_quota}"
                    )

    with cpu_sink(process_batches, decouple=args.process_pool) as sink:
        pending_batches = []

        for i in range(todo):
            if temp_span > 0:
                curr_temp = temp + 0.1 * np.random.randint(temp_span + 1)
            else:
                curr_temp = temp
            if i % 10 == 0:
                with results_lock:
                    scored_so_far = len(results)
                logger.info(f"{i*sample_batch_size} / {todo * sample_batch_size} samples generated, {scored_so_far} scored")

            X_init = torch.empty((sample_batch_size, 1), dtype=torch.long)
            X_init[:, 0] = stoi["BOS"]
            X_init = X_init.to(args.device)
            top_k = args.top_k if args.top_k != -1 else None
            allowed_token_ids_by_pos = getattr(env.tokenizer, "allowed_token_ids_by_pos", None)
            logits_processor = None
            with results_lock:
                current_min_genus = dynamic_min_genus
            if current_min_genus > 0:
                make_processor = getattr(env.tokenizer, "make_eos_min_genus_processor", None)
                if make_processor is not None:
                    logits_processor = make_processor(current_min_genus)
            batch_numpy = model.generate(
                X_init,
                args.max_len + 1,
                temperature=curr_temp,
                top_k=top_k,
                do_sample=True,
                allowed_token_ids_by_pos=allowed_token_ids_by_pos,
                logits_processor=logits_processor,
            ).cpu().numpy()

            pending_batches.append(batch_numpy)

            if len(pending_batches) >= DETOK_CHUNK_SIZE:
                sink.submit(pending_batches)
                pending_batches = []

        if pending_batches:
            sink.submit(pending_batches)

    if executor is not None:
        executor.shutdown(wait=True)

    do_stats(total_invalid, all_processed_data)

    return results
