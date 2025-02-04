import os
from pathlib import Path
from typing import Callable, Dict, Union


def gen_notes():
    for o in range(3, -1, -1):
        for idx_l, l in enumerate(['C', 'D', 'E', 'F', 'G', 'A', 'B']):
            print(f"    {l * (4-o)} = ({o}, {1 + idx_l})")
        print("")

    for o in range(4, 8):
        for idx_l, l in enumerate(['c', 'd', 'e', 'f', 'g', 'a', 'b']):
            print(f"    {l * (o-3)} = ({o}, {1 + idx_l})")
        print("")


def kern_file_tokenizer(vocab: Dict[str, int], path: Union[Path, str]) -> int:
    token_count = 0
    with open(path, 'r') as file:
        for line in file:
            for kern_tokens in line.strip().split("\t"):
                for token in kern_tokens.split(" "):
                    vocab[token] = vocab.get(token, 0) + 1
                    token_count += 1
    return token_count


def tokens_file_tokenizer(vocab: Dict[str, int], path: Union[Path, str]) -> int:
    token_count = 0
    with open(path, 'r') as file:
        for line in file:
            if " " in line:
                print(path)
            for token in line.strip().split("\t"):
                if " " in token:
                    print(path)
                vocab[token] = vocab.get(token, 0) + 1
                token_count += 1
    return token_count


DATADIR = Path("/home/anselm/datasets/GrandPiano/")


def vocab(
    extension: str,
    file_tokenizer: Callable[[Dict[str, int], Union[Path, str]], int]
):
    vocab: Dict[str, int] = {}
    token_count = 0
    for root, _, filenames in os.walk(DATADIR):
        for filename in filenames:
            path = Path(root) / filename
            if path.suffix == extension and not path.name.startswith("."):
                token_count += file_tokenizer(vocab, path)
    tokens = sorted(vocab.items(), key=lambda item: item[1], reverse=True)
    for token, count in tokens:
        print(f"{token}: {count:,}")
    print(f"{token_count:,} tokens, {len(tokens):,} unique.")


def kern_vocab():
    vocab(".krn", kern_file_tokenizer)


def tokens_vocab():
    vocab(".tokens", tokens_file_tokenizer)


if __name__ == '__main__':
    # gen_notes()
    kern_vocab()
    # tokens_vocab()
