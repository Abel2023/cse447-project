#!/usr/bin/env python
import os
import glob
import pickle
import collections
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class MyModel:
    # max context length
    K = 4

    def __init__(self, ngram_counts=None, global_counts=None):
        # ngram_counts: dict mapping context -> Counter of next chars
        self.ngram_counts = ngram_counts or {}
        self.global_counts = global_counts or collections.Counter()

    @classmethod
    def load_training_data(cls):
        """Search for reasonable training files and return list of lines.
        """
        candidates = ["train.txt", os.path.join("data", "train.txt")]
        candidates += glob.glob(os.path.join("train", "*.txt"))
        candidates += glob.glob(os.path.join("data", "*.txt"))
        candidates.append(os.path.join("example", "input.txt"))

        for p in candidates:
            if os.path.isfile(p):
                lines = []
                with open(p, 'rt', encoding='utf-8', errors='ignore') as f:
                    for ln in f:
                        ln = ln.rstrip('\n')
                        if ln:
                            lines.append(ln)
                if lines:
                    return lines
        return []

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                data.append(line.rstrip('\n'))
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write(f"{p}\n")

    def run_train(self, data, work_dir):
        """Build K-order n-gram counts from `data` (list of strings).
        Save compact structures to `work_dir` via `save`.
        """
        ngram_counts = {}
        global_counts = collections.Counter()
        for line in data:
            # treat the input as a sequence of characters
            chars = list(line)
            for i, ch in enumerate(chars[:-1]):
                global_counts[ch] += 1
                for k in range(1, self.K + 1):
                    # context is up to k chars ending at position i (suffix)
                    start = max(0, i - k + 1)
                    ctx = ''.join(chars[start:i + 1])
                    nxt = chars[i + 1]
                    if ctx not in ngram_counts:
                        ngram_counts[ctx] = collections.Counter()
                    ngram_counts[ctx][nxt] += 1
            # count last char too
            if chars:
                global_counts[chars[-1]] += 1

        self.ngram_counts = ngram_counts
        self.global_counts = global_counts
        # ensure work_dir exists
        os.makedirs(work_dir, exist_ok=True)
        self.save(work_dir)

    def run_pred(self, data):
        """Predict top-3 characters for each input string in `data`.
        Strategy: use longest matching suffix context (up to K); if none, use global frequencies; then fall back to ASCII letters.
        """
        preds = []
        ascii_pool = string.ascii_lowercase + string.ascii_uppercase + string.digits
        for inp in data:
            try:
                cand_counter = collections.Counter()
                # try longest suffixes
                found = False
                for k in range(self.K, 0, -1):
                    if len(inp) >= k:
                        ctx = inp[-k:]
                        if ctx in self.ngram_counts:
                            cand_counter = self.ngram_counts[ctx]
                            found = True
                            break
                if not found:
                    # use global char frequencies
                    cand_counter = self.global_counts

                # choose top 3 by frequency
                top = [c for c, _ in cand_counter.most_common(3)]

                # fill with high-probability ascii chars if needed
                fill_idx = 0
                while len(top) < 3 and fill_idx < len(ascii_pool):
                    ch = ascii_pool[fill_idx]
                    if ch not in top:
                        top.append(ch)
                    fill_idx += 1

                # final fallback: repeat 'a' if something very strange happens
                while len(top) < 3:
                    top.append('a')

                preds.append(''.join(top[:3]))
            except Exception:
                # on any unexpected error, return deterministic fallback
                preds.append(''.join(list('the')[:3]))
        return preds

    def save(self, work_dir):
        path = os.path.join(work_dir, 'model.checkpoint')
        with open(path, 'wb') as f:
            pickle.dump({'ngram': self.ngram_counts, 'global': self.global_counts}, f)

    @classmethod
    def load(cls, work_dir):
        path = os.path.join(work_dir, 'model.checkpoint')
        if not os.path.isfile(path):
            # no checkpoint — return empty model
            return MyModel()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return MyModel(ngram_counts=data.get('ngram', {}), global_counts=data.get('global', collections.Counter()))


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    # deterministic behavior for any randomized fallback
    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print(f'Making working directory {args.work_dir}')
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print(f'Found {len(train_data)} training lines')
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print(f'Loading test data from {args.test_data}')
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print(f'Writing predictions to {args.test_output}')
        assert len(pred) == len(test_data), f'Expected {len(test_data)} predictions but got {len(pred)}'
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError(f'Unknown mode {args.mode}')
