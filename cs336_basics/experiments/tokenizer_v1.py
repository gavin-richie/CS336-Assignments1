import pickle
import functools
import regex as re
from collections import deque
from typing import Iterable, Iterator
import time
import numpy as np
from itertools import chain
from tqdm import tqdm
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(
            self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        Not optimized at all. Trying to get it to work first.
        """
        start_time = time.time()
        logger.info("Starting Tokenizer initialization")

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = [tok for id, tok in vocab.items() if str(tok).startswith("b'<|")]
        new_special_tokens = []
        if special_tokens:
            new_special_tokens = [
                tok.encode("utf-8") for tok in tqdm(special_tokens, desc="Processing special tokens")
                if tok.encode("utf-8") not in self.special_tokens
            ]
        self.special_tokens += new_special_tokens
        self.vocab.update({id: vocab for id, vocab in enumerate(new_special_tokens, start=len(self.vocab))})
        self.vocab_to_int = {value: key for key, value in vocab.items()}
        self.merge_priority = {}
        for i, (a, b) in tqdm(enumerate(merges), desc="Building merge priority", total=len(merges)):
            pair_key = (self.vocab_to_int.get(a), self.vocab_to_int.get(b))
            if pair_key[0] is not None and pair_key[1] is not None:
                self.merge_priority[pair_key] = i

        logger.info(f"Tokenizer initialization completed in {time.time() - start_time:.2f} seconds")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Only works with .pkl files for now. Serialization of bytes with json is not totally straight forward, might need to import base64.
        """
        start_time = time.time()
        logger.info("Starting from_files")

        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)

        instance = cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
        logger.info(f"from_files completed in {time.time() - start_time:.2f} seconds")
        return instance

    @functools.lru_cache(maxsize=10000)
    def encode_ordinary(self, word: bytes) -> list[int]:
        """
        lru-cache lets us skip common words, merge_priority lets us lookup priority of the byte pairs.
        """
        start_time = time.time()
        # logger.info("Starting encode_ordinary")

        tokens = [self.vocab_to_int[bytes([byte_val])] for byte_val in word]

        while len(tokens) >= 2:
            best_priority = float("inf")
            best_pos = -1
            best_pair = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                priority = self.merge_priority.get(pair, float("inf"))

                if priority < best_priority:
                    best_priority = priority
                    best_pos = i
                    best_pair = pair

            if best_priority == float("inf"):
                break

            merge_bytes = self.merges[best_priority]
            new_token = merge_bytes[0] + merge_bytes[1]
            new_token_id = self.vocab_to_int[new_token]
            tokens[best_pos: best_pos + 2] = [new_token_id]

        # logger.info(f"encode_ordinary completed in {time.time() - start_time:.2f} seconds")
        return tokens

    def encode(self, text: str) -> list[int]:
        start_time = time.time()
        # logger.info("Starting encode")

        escaped = [re.escape(token.decode()) for token in self.special_tokens]
        escaped.sort(key=len, reverse=True)
        SPECIAL = r"|".join(escaped)
        PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        try:
            specials = [self.vocab_to_int[spec.encode()] for spec in re.findall(SPECIAL, text)]
            documents = re.split(SPECIAL, text)
        except KeyError:
            specials = []
            documents = text

        indeces = []
        for i, document in enumerate(documents):
            words = []
            for subword in PAT.findall(document):
                words.append(subword.encode())

            encoder = functools.partial(self.encode_ordinary)
            indeces.extend(list(map(encoder, words)))

            if specials and i < len(documents) - 1:
                indeces.append([specials[i]])

        # logger.info(f"encode completed in {time.time() - start_time:.2f} seconds")
        return list(chain.from_iterable(indeces))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        We are receiving a file object "iterable". We read a chunk from the file and tokenize it.
        The EncodingIterator will yield the resulting tokens one by one.
        """
        start_time = time.time()
        # logger.info("Starting encode_iterable")

        class EncodingIterator:
            def __init__(self, tokenizer, iterable):
                self.tokens = deque()
                self.iterable = iterable
                self.tokenizer = tokenizer
                self.chunk_size = 1024 * 1024
                self.buffer = ""
                self.start_time = time.time()

            def __iter__(self):
                return self

            def __next__(self):
                if not self.tokens:
                    chunk = self.buffer + self.iterable.read(self.chunk_size)
                    split_point = chunk.rfind("\n")
                    if split_point == -1:
                        # logger.warning("Did not find a proper split point, splitting at chunk end")
                        split_point = len(chunk)
                    self.buffer = chunk[split_point:]
                    chunk = chunk[:split_point]

                    if not chunk:  # if "\n" in the begin of the chunk
                        if self.buffer:
                            chunk = self.buffer
                            self.buffer = ""
                        else:
                            # logger.info(f"EncodingIterator completed in {time.time() - self.start_time:.2f} seconds")
                            raise StopIteration
                    self.tokens = deque(self.tokenizer.encode(chunk))

                return self.tokens.popleft()

        iterator = EncodingIterator(self, iterable)
        # logger.info(f"encode_iterable setup completed in {time.time() - start_time:.2f} seconds")
        return iterator

    def decode(self, ids: list[int]) -> str:
        start_time = time.time()
        logger.info("Starting decode")

        byte_string = b""
        for id in tqdm(ids, desc="Decoding IDs"):
            byte_string += self.vocab[id]
        result = str(byte_string, "utf-8", errors="replace")

        logger.info(f"decode completed in {time.time() - start_time:.2f} seconds")
        return result

    def to_numpy(self, output: str, text_file: str):
        """Tokenizes a text file into a one-dimensional numpy vector for training"""
        start_time = time.time()
        logger.info("Starting to_numpy")

        ids = []
        with open(file=text_file, encoding="utf-8") as f:
            for _id in tqdm(self.encode_iterable(f), total=os.path.getsize(text_file), desc="Tokenizing file"):
                ids.append(_id)
        ids = np.array(ids, dtype="uint16")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        np.save(output, arr=ids)

        logger.info(f"to_numpy completed in {time.time() - start_time:.2f} seconds")

    @staticmethod
    def throughput(filename: str, tokenizer) -> None:
        start_time = time.time()
        logger.info("Starting throughput")

        with open(filename, encoding="utf-8") as f:
            text = f.read()
            num_bytes = len(bytes(text, encoding="utf-8"))
            f.seek(0)
            tokenizer.encode("warmup")
            t0 = time.time()
            indices = []
            for _id in tqdm(tokenizer.encode_iterable(f), desc="Processing file"):
                indices.append(_id)
            t1 = time.time()
            throughput = num_bytes / (t1 - t0)

        compression_ratio = Tokenizer.compression_ratio(text, indices)
        logger.info(f"For the {filename} dataset we get {compression_ratio=}")
        logger.info(f"and {throughput=} bytes/s, {throughput / 1024 ** 2:.2f} MB/s")
        logger.info(f"throughput completed in {time.time() - start_time:.2f} seconds")

        return throughput

    @staticmethod
    def compression_ratio(string: str, indices: list[int]) -> float:
        start_time = time.time()
        logger.info("Starting compression_ratio")

        bytes_string = len(bytes(string, encoding="utf-8"))
        bytes_indices = len(indices)
        compression_ratio = bytes_string / bytes_indices

        logger.info(f"compression_ratio completed in {time.time() - start_time:.2f} seconds")
        return compression_ratio


if __name__ == "__main__":
    special_tokens = ["<|imstart|>", "<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        vocab_filepath="data/tokenizer_data/vocab_TinyStories-train.pkl",
        merges_filepath="data/tokenizer_data/merges_TinyStories-train.pkl",
        special_tokens=special_tokens,
    )
    print(type(tokenizer.vocab), type(tokenizer.merges))
    enc = tokenizer.encode("Lets test how lucky we can get")
    print(enc)
    dec = tokenizer.decode(enc)
    print(dec)
    # tokenizer.throughput(filename="data/owt_valid.txt", tokenizer=tokenizer)
    # tokenizer.throughput(filename="data/sample_tiny.txt", tokenizer=tokenizer)
    tokenizer.to_numpy(output=r"data/training_data/Tinystories_valid", text_file=r"data/TinyStories-valid.txt")
    import sys

    sys.exit()
    # Test throughput
    import cProfile
    import pstats

    with cProfile.Profile() as profile:
        tokenizer.throughput(filename="data/TinyStories-train.txt", tokenizer=tokenizer)
        # import sys; sys.exit()
        # tokenizer.throughput(filename="data/sample_tiny.txt", tokenizer=tokenizer)

        result = pstats.Stats(profile)
        result.sort_stats(pstats.SortKey.TIME)
        result.print_stats(10)

'''    def encode_passing_test(self, text: str) -> list[int]:

        escaped = [re.escape(token.decode()) for token in self.special_tokens]
        # regex finishes at first match. if we sort by length desc, we will get substrings of longer strings at the end. 
        # matching <|endoftext|><|endoftext|> before <|endoftext|>
        escaped.sort(key=len, reverse=True)
        SPECIAL = r"|".join(escaped)
        #text = text.encode("utf-8")
        PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        segments = []
        try:
            specials = [self.vocab_to_int[spec.encode()] for spec in re.findall(SPECIAL, text)]
            splitted = re.split(SPECIAL, text)
        except KeyError:
            specials = []
            splitted = text
            #import sys;sys.exit()
        indeces = []
        # UGLY! 
        for i, chunk in enumerate(splitted):
            segments = []
            for segment in PAT.findall(chunk): 
                segments.append(segment.encode())

            for segment in segments:
                # Convert to token IDs
                segment = [self.vocab_to_int[bytes([byte_val])] for byte_val in segment]

                # OPTIMIZED MERGE PROCESS
                while len(segment) >= 2:
                    # Find all pairs in current segment and their positions
                    pairs_in_segment = {}
                    for idx in range(len(segment) - 1):
                        pair = (segment[idx], segment[idx + 1])
                        if pair not in pairs_in_segment:
                            pairs_in_segment[pair] = []
                        pairs_in_segment[pair].append(idx)

                    # Find the highest priority merge that exists in this segment
                    best_merge_idx = None
                    best_pair = None

                    for merge_idx, merge_bytes in enumerate(self.merges):
                        a, b = merge_bytes
                        pair_ids = (self.vocab_to_int[a], self.vocab_to_int[b])
                        if pair_ids in pairs_in_segment:
                            best_merge_idx = merge_idx
                            best_pair = pair_ids
                            break  # Found highest priority merge

                    if best_pair is None:
                        break  # No more merges possible

                    # Apply the best merge
                    new_token = self.merges[best_merge_idx][0] + self.merges[best_merge_idx][1]
                    new_token_id = self.vocab_to_int[new_token]

                    # Merge all instances of this pair (from right to left to avoid index shifting)
                    positions = pairs_in_segment[best_pair]
                    for pos in reversed(positions):
                        if pos < len(segment) - 1 and segment[pos] == best_pair[0] and segment[pos + 1] == best_pair[1]:
                            segment[pos:pos + 2] = [new_token_id]

                indeces.append(segment)

            if specials and i < len(splitted) - 1:
                indeces.append([specials[i]])

        return list(chain.from_iterable(indeces))
    '''