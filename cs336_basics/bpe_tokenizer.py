# cs336_basics/tokenizer.py
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Iterable, Iterator, List, Dict, Tuple
import os
import regex as re
from array import array
import json
import pickle
import numpy as np
from tqdm import tqdm
from cs336_basics.experiments import *

GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str) -> list[bytes]:
    str_tokens = re.findall(GPT2_SPLIT_PATTERN, text)
    byte_tokens = [s.encode("utf-8") for s in str_tokens]
    return byte_tokens


GPT2_RE = re.compile(GPT2_SPLIT_PATTERN)


def iter_pretokenize(text: str) -> Iterator[bytes]:
    """按 GPT-2 正则逐个产生字节串，零内存列表。"""
    for m in GPT2_RE.finditer(text):
        yield m.group(0).encode("utf-8")


class Tokenizer:
    def __init__(self, vocab_size: int, special_tokens: list[str] | None = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        self.special_tokens_bytes = [
            token.encode("utf-8") for token in self.special_tokens
        ]

        self.merges: List[Tuple[bytes, bytes]] = []
        self.stoi: Dict[bytes, int] = {}
        self.itos: Dict[int, bytes] = {}
        self.merges_rank: Dict[Tuple[bytes, bytes], int] = {}
        self.eos_token_id = 256

        # init vocab 先初始化特殊token
        for i, token_bytes in enumerate(self.special_tokens_bytes):  # special tokens
            self.stoi[token_bytes] = i
            self.itos[i] = token_bytes

        offset = len(self.special_tokens_bytes)  # 单字节 tokens
        for i in range(256):
            self.stoi[bytes([i])] = i + offset
            self.itos[i + offset] = bytes([i])

        self.vocab = self.itos.copy()  # for serialization
        self.merges_rank = {}  # for fast lookup
        # pair2new: (p1, p2) -> new_token_id
        self.pair2new = {(p1, p2): self.stoi[p1 + p2] for (p1, p2) in self.merges}

    def _encode_ordinary_text(self, text_bytes: bytes) -> list[int]:
        """BPE encode (不含特殊 token) —— 无额外列表 / O(n) 内存"""
        if not text_bytes:
            return []

        # ➊ 只解一次字节 → str
        try:
            text = text_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = text_bytes.decode("utf-8", errors="replace")

        ids_out = array("H")  # uint16 足够 ≤ 65k vocab

        pair_rank = self.merges_rank
        pair2new = self.pair2new
        byte2id = self.stoi  # 局部 alias，加速

        # ➋ 逐个“词块”处理，避免一次性 list
        for word_b in iter_pretokenize(text):
            # a. 初始：单字节 ids
            token_ids = array("H", (byte2id[bytes([b])] for b in word_b))

            # b. 就地合并：最经典 “greedy smallest-rank merge until稳定”
            while True:
                best_rank = 1000000000
                best_pos = -1
                # ——— 找当前序列里 rank 最小的 pair ———
                for i in range(len(token_ids) - 1):
                    r = pair_rank.get(
                        (self.itos[token_ids[i]], self.itos[token_ids[i + 1]]),
                        1000000000,
                    )
                    if r < best_rank:  # 每一轮更新只能找到一个最优（也是最小）的一个pair更新
                        best_rank, best_pos = r, i
                if best_pos == -1:  # 当前词序列没有可合并的pair
                    break
                # ——— 替换 best_pos & best_pos+1 为新的 token id ———
                new_id = pair2new[
                    (self.itos[token_ids[best_pos]], self.itos[token_ids[best_pos + 1]])
                ]
                token_ids[best_pos: best_pos + 2] = array("H", [new_id])  # 列表切片右边界取不到

            ids_out.extend(token_ids)

        # ➌ array → Python list（评测期望 list）
        return ids_out.tolist()

    def encode(self, text: str) -> list[int]:
        """Encode str"""
        if not text:
            return []

        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        if not sorted_special_tokens:
            return self._encode_ordinary_text(text.encode("utf-8"))

        special_pattern = f"({'|'.join(re.escape(s) for s in sorted_special_tokens)})"
        text_parts = re.split(special_pattern, text)

        all_ids = []
        for part in text_parts:
            if part in self.special_tokens:
                all_ids.append(self.stoi[part.encode("utf-8")])
            elif part:
                all_ids.extend(self._encode_ordinary_text(part.encode("utf-8")))
        return all_ids

    def encode_iterable(
            self,
            iterable: Iterable[str],
            *,
            output_format: str = "flat",
    ) -> Iterator[int] | Iterator[list[int]]:
        flat = output_format == "flat"
        for line in iterable:
            # —— 不要 strip 换行 ——          ▼
            ids = self.encode(line)
            if flat:
                yield from ids
            else:
                yield ids

    def decode(self, ids: list[int]) -> str:
        """ID -> text"""
        all_bytes = b"".join(self.itos.get(id, b"") for id in ids)
        return all_bytes.decode("utf-8", errors="replace")

    @classmethod
    def from_serialized(
            cls,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str],
    ):
        instance = cls(vocab_size=len(vocab), special_tokens=special_tokens)
        instance.stoi = {v: k for k, v in vocab.items()}
        instance.itos = vocab
        instance.merges = merges
        instance.merges_rank = {pair: i for i, pair in enumerate(merges)}
        instance.vocab = vocab

        instance.pair2new = {(p1, p2): instance.stoi[p1 + p2] for (p1, p2) in merges}  # (bytes,bytes) -> int(token_id)

        return instance


def get_tokenizer(vocab_path,
                  merges_path,
                  special_tokens=None):
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    if vocab_path.endswith(".pkl") and merges_path.endswith(".pkl"):
        try:
            before_load_pkl = time.time()
            print("Loading tokenizer from pickle")
            print("vocab file: {}".format(vocab_path))
            print("merges file: {}".format(merges_path))

            with open(vocab_path, "rb") as f:
                vocab = pickle.load(f)
            with open(merges_path, "rb") as f:
                merges = pickle.load(f)
            tokenizer = Tokenizer.from_serialized(vocab, merges, special_tokens)
            print("Tokenizer loaded successfully")
            after_load_pkl = time.time()
            print(f"Time taken to load pickle: {after_load_pkl - before_load_pkl:.2f} seconds")
            return tokenizer
        except Exception as e:
            print("Error loading tokenizer from pickle: {}".format(e))
            raise e

    if vocab_path.endswith(".json") and merges_path.endswith(".txt"):
        try:
            print("Loading tokenizer from json")
            print("vocab file: {}".format(vocab_path))
            print("merges file: {}".format(merges_path))
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            vocab = {int(k): v.encode("utf-8") for k, v in vocab_data.items()}
            with open(merges_path, "r", encoding="utf-8") as f:
                merges_data = f.readlines()
            merges = [(line.split()[0].encode("utf-8"), line.split()[1].encode("utf-8")) for line in merges_data]
            tokenizer = Tokenizer.from_serialized(vocab, merges, special_tokens)
            print("Tokenizer loaded successfully")
            return tokenizer
        except Exception as e:
            print("Error loading tokenizer from json and txt: {}".format(e))
            raise e

    return None


def batch_tokenize(batch, tokenizer):
    # 预估每行平均 50 token，预分配数组
    estimated_tokens = len(batch) * 50
    out = np.zeros(estimated_tokens, dtype=np.int32)
    pos = 0
    for line in batch:
        tokens = tokenizer.encode(line)
        token_len = len(tokens)
        out[pos:pos + token_len] = tokens
        pos += token_len
    return out[:pos]


def read_batches(txt_path, batch_size):
    batch = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            batch.append(line)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def encode_txt_as_memarray(tokenizer, txt_path, memmap_path, batch_size=8192, n_workers=16):
    before_batch_tokenize = time.time()
    print(f"Start tokenizing {txt_path}")

    total_tokens = 0
    results = []

    before_encode = time.time()
    print(f"Start encoding {txt_path}")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []

        for batch in read_batches(txt_path, batch_size):
            futures.append(executor.submit(batch_tokenize, batch, tokenizer))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing"):
            result = future.result()
            results.append(result)
            total_tokens += result.shape[0]

    after_encode = time.time()
    print(f"Time taken to encode: {after_encode - before_encode:.2f} seconds\n total_tokens={total_tokens}")

    before_mem_map = time.time()
    print(f"Start writing mem_map {memmap_path}")
    os.makedirs(os.path.dirname(memmap_path), exist_ok=True)
    token_memmap = np.memmap(memmap_path, dtype=np.int32, mode="w+", shape=(total_tokens,))
    offset = 0
    print(f"results[0].shape[0]={results[0].shape[0]}, len(results[0])={len(results[0])}")
    for result in results:
        token_memmap[offset:offset + result.shape[0]] = result
        offset += result.shape[0]
    token_memmap.flush()

    after_mem_map = time.time()
    print(f"Time taken to write mem_map {memmap_path}: {after_mem_map - before_mem_map:.2f} seconds")

def memmap2npy(input_memmap_path: str, output_npy_path: str, chunk_size=100_000_000)->None:
    """
    Converts a memmap file(np.int32) to npy file(np.uint16).
    :param input_memmap_path: Path to the input memmap file.
    :param output_npy_path: Path to the output npy file.
    :param chunk_size: Chunk size to read from the memmap file.
    :return: None
    """
    t0 = time.time()
    logger.info(f"Converting mem_map file to npy file {input_memmap_path}")
    mem_map = np.memmap(input_memmap_path, dtype=np.int32, mode="r")
    total_tokens = mem_map.shape[0]
    logger.info(f"Total tokens={total_tokens}")

    if np.any(mem_map>np.iinfo(np.uint16).max):
        logger.error(f"mem_map contains tokens > np.iinfo(np.uint16).max={np.iinfo(np.uint16).max}")
        raise ValueError(f"Token IDs exceed uint16 max value (65,535). Cannot convert to uint16.")
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)

    chunks = []
    for start in tqdm(range(0, total_tokens, chunk_size), desc="Converting chunks"):
        end = min(start + chunk_size, total_tokens)
        chunk = mem_map[start:end].astype(np.uint16)
        chunks.append(chunk)

    # Step 5: Concatenate and save
    all_tokens = np.concatenate(chunks)
    np.save(output_npy_path, all_tokens, allow_pickle=False)

    logger.info(f"Time taken to convert mem_map file to npy file: {time.time() - t0:.2f} seconds")
