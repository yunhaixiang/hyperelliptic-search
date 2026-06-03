import os
import pickle
import random
import time
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from logging import getLogger

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from src.envs.environment import do_stats
from src.utils import MAX_WORKERS

logger = getLogger()


def detokenize(data, args, env, executor=None):
    res = []
    pars = env.tokenizer.dataclass._save_class_params()
    if args.process_pool:
        BATCH = args.gen_batch_size
        data_slices = [data[i : i + BATCH] for i in range(0, len(data), BATCH)]

        if executor is not None:
            for chunk in executor.map(env.tokenizer.decode_batch, data_slices, repeat(pars, len(data_slices))):
                if chunk:
                    res.extend(chunk)
        else:
            with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, args.num_workers)) as ex:
                for chunk in ex.map(env.tokenizer.decode_batch, data_slices, repeat(pars, len(data_slices))):
                    if chunk:
                        res.extend(chunk)
    else:
        res = env.tokenizer.decode_batch(data, pars)
    return res


# helper functions for creating the training and test Datasets


def generate_and_score(args, classname):
    """
    Generation method if no data
    """
    data = []
    BATCH = args.gen_batch_size
    batch_counts = [BATCH] * (args.gensize // BATCH)
    rem = args.gensize % BATCH
    if rem:
        batch_counts.append(rem)
    if args.process_pool:
        pars = classname._save_class_params()
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, args.num_workers)) as executor:
            # map returns lists; stream them to avoid a giant materialization
            for chunk in executor.map(
                classname._batch_generate_and_score, batch_counts, repeat(args.N, len(batch_counts)), repeat(pars, len(batch_counts))
            ):
                if chunk:  # extend incrementally to manage memory
                    data.extend(chunk)
    else:
        total = 0
        for idx, t in enumerate(batch_counts):
            d = classname._batch_generate_and_score(t, args.N)
            if d is not None:
                data.extend(d)
            total += t
            if idx == 0 or total % max(BATCH, 1000) == 0 or total == args.gensize:
                logger.info(f"Generated and scored initial data: {total} / {args.gensize}; valid kept so far: {len(data)}")
    return data


def load_seed_data(args, classname):
    paths_arg = getattr(args, "initial_data_sqlite", "")
    if not paths_arg:
        return []
    loader = getattr(classname, "load_initial_data", None)
    if loader is None:
        raise ValueError(f"{classname.__name__} does not support --initial_data_sqlite")

    paths = []
    for part in [p.strip() for p in paths_arg.split(",") if p.strip()]:
        matches = sorted(glob(part))
        if matches:
            paths.extend(matches)
        else:
            paths.append(part)
    data = []
    for path in paths:
        try:
            chunk = loader(path, args.N, max_rows=getattr(args, "initial_data_max_rows", 0), pars=classname._save_class_params())
        except Exception as exc:
            logger.warning(f"Skipping initial data file {path}: {type(exc).__name__}: {exc}")
            continue
        logger.info(f"Loaded initial data from {path}: {len(chunk)} valid examples")
        data.extend(chunk)
    logger.info(f"Loaded initial data from {len(paths)} file(s): {len(data)} valid examples total")
    return data


def selection_score(d, args=None):
    scorer = getattr(d, "selection_score", None)
    if scorer is not None:
        return scorer()
    return d.score


def _datapoint_genus(d):
    genus = getattr(d, "genus", None)
    if genus is not None:
        return int(genus)
    score = getattr(d, "score", None)
    if score is not None and score >= 0:
        return int(score)
    return None


def _frontier_bucket(genus, max_genus, args):
    if genus <= args.frontier_low_cutoff:
        return "low"
    top_floor = max_genus - args.frontier_top_width
    high_floor = max_genus - args.frontier_high_width
    if genus >= top_floor:
        return "top"
    if genus >= high_floor:
        return "high"
    return "mid"


def _cap_weights(weights, max_probability):
    if max_probability <= 0:
        return weights

    capped = np.asarray(weights, dtype=np.float64)
    active = np.ones(len(capped), dtype=bool)
    remaining_mass = 1.0
    while True:
        over = active & (capped > max_probability)
        if not over.any():
            break
        capped[over] = max_probability
        active[over] = False
        remaining_mass = 1.0 - capped[~active].sum()
        if remaining_mass <= 0 or not active.any():
            break
        active_sum = capped[active].sum()
        if active_sum <= 0:
            capped[active] = remaining_mass / active.sum()
        else:
            capped[active] *= remaining_mass / active_sum
    total = capped.sum()
    if total <= 0:
        return weights
    return capped / total


def make_training_sample_weights(train_set, args):
    if getattr(args, "genus_sampling", "none") == "none":
        return None

    genera = [_datapoint_genus(d) for d in train_set]
    valid_genera = [g for g in genera if g is not None]
    if not valid_genera:
        logger.warning("genus_sampling requested, but training data has no genus metadata; using uniform sampling")
        return None

    max_genus = max(valid_genera)
    bucket_probs = {
        "low": args.frontier_low_prob,
        "mid": args.frontier_mid_prob,
        "high": args.frontier_high_prob,
        "top": args.frontier_top_prob,
    }
    total_prob = sum(bucket_probs.values())
    if total_prob <= 0:
        logger.warning("frontier bucket probabilities sum to zero; using uniform sampling")
        return None
    bucket_probs = {key: value / total_prob for key, value in bucket_probs.items()}

    buckets = {key: [] for key in bucket_probs}
    for idx, genus in enumerate(genera):
        if genus is None:
            continue
        buckets[_frontier_bucket(genus, max_genus, args)].append(idx)

    weights = np.zeros(len(train_set), dtype=np.float64)
    nonempty = {key: indices for key, indices in buckets.items() if indices}
    present_prob = sum(bucket_probs[key] for key in nonempty)
    if present_prob <= 0:
        return None

    for key, indices in nonempty.items():
        per_item = (bucket_probs[key] / present_prob) / len(indices)
        weights[indices] = per_item

    epoch_samples = max(1, int(args.max_steps) * int(args.batch_size))
    max_probability = 0.0
    if args.frontier_max_repeat > 0:
        max_probability = float(args.frontier_max_repeat) / epoch_samples
        weights = _cap_weights(weights, max_probability)

    logger.info(
        "Frontier genus sampling: "
        f"Gmax={max_genus}; "
        f"low={len(buckets['low'])}, mid={len(buckets['mid'])}, "
        f"high={len(buckets['high'])}, top={len(buckets['top'])}; "
        f"probs low/mid/high/top="
        f"{bucket_probs['low']:.3f}/{bucket_probs['mid']:.3f}/{bucket_probs['high']:.3f}/{bucket_probs['top']:.3f}; "
        f"max_repeat={args.frontier_max_repeat}"
    )
    return weights


def _token_cache_key(tokenizer, datapoint, max_len):
    key_fn = getattr(tokenizer, "cache_key_for_datapoint", None)
    if key_fn is not None:
        return key_fn(datapoint, max_len)
    return (
        tokenizer.__class__.__name__,
        int(max_len),
        getattr(datapoint, "features", None),
    )


def encode_datapoints(datapoints, tokenizer, max_len, label):
    encoded = []
    hits = 0
    misses = 0
    start_time = time.time()
    for datapoint in datapoints:
        key = _token_cache_key(tokenizer, datapoint, max_len)
        cached_key = getattr(datapoint, "_encoded_token_cache_key", None)
        cached_value = getattr(datapoint, "_encoded_token_cache_value", None)
        legacy_key_match = (
            isinstance(cached_key, tuple)
            and len(cached_key) == 3
            and len(key) == 2
            and cached_key[0] == key[0]
            and cached_key[2] == key[1]
        )
        if (cached_key == key or legacy_key_match) and cached_value is not None:
            datapoint._encoded_token_cache_key = key
            encoded.append(cached_value)
            hits += 1
            continue

        value = tokenizer.encode(datapoint)
        datapoint._encoded_token_cache_key = key
        datapoint._encoded_token_cache_value = value
        encoded.append(value)
        misses += 1

    elapsed = time.time() - start_time
    logger.info(
        f"Encoded {label}: {len(datapoints)} examples in {elapsed:.2f}s "
        f"({hits} cache hits, {misses} misses)"
    )
    return encoded


def select_best(n, data, args=None):
    if len(data) <= n:
        random.shuffle(data)
        return data
    sorted_data = sorted(data, key=lambda x: selection_score(x, args), reverse=True)[:n]
    random.shuffle(sorted_data)
    return sorted_data


def make_train_test(data, ntest):
    """
    Create a train and test dataset from a dataset.
    """
    indices = np.random.permutation(len(data))
    rp = [data[i] for i in indices]
    return rp[:-ntest], rp[-ntest:]


def compute_unique_data(old_data, new_data=None):
    def add_unique(src, unique_hashes):
        des = []
        for d in src:
            if d.features not in unique_hashes:
                unique_hashes.add(d.features)
                des.append(d)
        return des, unique_hashes

    unique_hashes = set()
    unique_old_data, unique_hashes = add_unique(old_data, unique_hashes)
    if new_data is not None:
        unique_new_data, unique_hashes = add_unique(new_data, unique_hashes)
    else:
        unique_new_data = None
    return unique_old_data, unique_new_data


def update_datasets(args, data, train_set, test_set, train_path, test_path):
    inc_temp = False
    if args.keep_only_unique:
        bef = len(data)
        data, _ = compute_unique_data(data)
        aft = len(data)
        logger.info(f"Unique processing: {aft} examples left, {bef-aft} duplicates")
        do_stats(-1, data)
        if aft / (bef + 1) < 0.9:
            inc_temp = True
    if args.new_proportion > 0.0:
        new_data = select_best(int(args.new_proportion * args.pop_size), data, args=args)
    else:
        new_data = select_best(args.pop_size, data, args=args)

    if len(new_data) >= 2 * args.ntest or test_set is None:
        new_train, test_set = make_train_test(new_data, args.ntest)
    else:
        new_train = new_data
    logger.info(f"New train and test generated. Size are train: {len(new_train)}, test {len(test_set)}")
    # Get all examples of previous train and current train and then select best.
    if args.keep_only_unique:
        train_set, new_train = compute_unique_data(train_set, new_train)
        logger.info(f"Unique data computed for original train set: {len(train_set)}, generated train set: {len(new_train)}")
    if args.new_proportion > 0.0:
        train_set = select_best(int((1.0 - args.new_proportion) * args.pop_size), train_set, args=args) + new_train
    else:
        train_set = select_best(args.pop_size, train_set + new_train, args=args)
    logger.info(f"Final train and test generated. Size are train: {len(train_set)}, test {len(test_set)}")

    pickle.dump(test_set, open(test_path, "wb"))
    pickle.dump(train_set, open(train_path, "wb"))
    return train_set, test_set, inc_temp


def load_initial_data(args, classname):
    train_data_path = os.path.join(args.dump_path, "train_data.pkl")
    test_data_path = os.path.join(args.dump_path, "test_data.pkl")
    if os.path.isfile(train_data_path):
        logger.info("resuming from existing data")
        train_set = pickle.load(open(train_data_path, "rb"))
        test_set = pickle.load(open(test_data_path, "rb"))
    else:
        data = load_seed_data(args, classname)
        generated = generate_and_score(args, classname=classname)
        if data:
            logger.info(f"Combining initial data ({len(data)}) with generated data ({len(generated)})")
        data.extend(generated)
        test_set = []
        train_set = []
        train_set, test_set, _ = update_datasets(args, data, train_set, test_set, train_data_path, test_data_path)
    return train_set, test_set


class CharDataset(Dataset):
    def __init__(self, encoded_data, max_len, stoi):
        self.encoded_data = encoded_data
        self.max_len = max_len
        self.pad_token_id = stoi["PAD"]

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

    def collate_fn(self, batch):
        x = np.full((len(batch), self.max_len + 2), self.pad_token_id, dtype=np.int32)

        for i, el in enumerate(batch):
            x[i, : el.shape[0]] = el
        valid_col = (x != self.pad_token_id).any(axis=0)
        last_col = np.nonzero(valid_col)[0][-1] + 1
        x = x[:, :last_col]
        y = np.concatenate([x[:, 1:], np.full((len(batch), 1), self.pad_token_id, dtype=x.dtype)], axis=1)
        return torch.LongTensor(x), torch.LongTensor(y)


class InfiniteDataLoader:
    """
    Create a infinite datalaoder in PyTorch
    """

    def __init__(self, dataset, sample_weights=None, **kwargs):
        if sample_weights is None:
            train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        else:
            weights = torch.as_tensor(sample_weights, dtype=torch.double)
            batch_size = int(kwargs.get("batch_size", 1))
            num_samples = max(len(dataset), batch_size)
            train_sampler = torch.utils.data.WeightedRandomSampler(weights, replacement=True, num_samples=num_samples)
        self.train_loader = DataLoader(dataset, sampler=train_sampler, collate_fn=dataset.collate_fn, **kwargs)
        self.data_iter = iter(self.train_loader)
        self._closed = False

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:  # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

    def close(self):
        if self._closed:
            return
        shutdown_workers = getattr(self.data_iter, "_shutdown_workers", None)
        if shutdown_workers is not None:
            shutdown_workers()
        self.data_iter = None
        self.train_loader = None
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
