import re
import torch
from collections import namedtuple

id_start = 49406
id_end = 49407
id_pad = 49407
chunk_length = 75
comma_token = 267
comma_padding_backtrack = 20


def get_promt_embedding(pos_prompt, tokenizer, text_encoder, device=torch.device("cpu")):
    batch_chunks, token_count = process_texts([pos_prompt], tokenizer)
    used_embeddings = {}
    chunk_count = max([len(x) for x in batch_chunks])

    zs = []
    for i in range(chunk_count):
        batch_chunk = [chunks[i] if i < len(chunks) else empty_chunk() for chunks in batch_chunks]

        tokens = [x.tokens for x in batch_chunk]
        multipliers = [x.multipliers for x in batch_chunk]
        fixes = [x.fixes for x in batch_chunk]

        for fix in fixes:
            for _position, embedding in fix:
                used_embeddings[embedding.name] = embedding

        z = process_tokens(tokens, multipliers, text_encoder, device)
        zs.append(z)

    if len(used_embeddings) > 0:
        # TODO:
        comments = []
        embeddings_list = ", ".join([f'{name} [{embedding.checksum()}]' for name, embedding in used_embeddings.items()])
        comments.append(f"Used embeddings: {embeddings_list}")

    return torch.hstack(zs)


# reference: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_clip.py#L175
def process_texts(texts, tokenizer):
    """
    Accepts a list of texts and calls tokenize_line() on each, with cache. Returns the list of results and maximum
    length, in tokens, of all texts.
    """

    token_count = 0

    cache = {}
    batch_chunks = []
    for line in texts:
        if line in cache:
            chunks = cache[line]
        else:
            chunks, current_token_count = tokenize_line(line, tokenizer)
            token_count = max(current_token_count, token_count)

            cache[line] = chunks

        batch_chunks.append(chunks)

    return batch_chunks, token_count


def tokenize_line(line, tokenizer, enable_emphasis=True):
    """
    this transforms a single prompt into a list of PromptChunk objects - as many as needed to
    represent the prompt.
    Returns the list and the total number of tokens in the prompt.
    """

    if enable_emphasis:
        parsed = parse_prompt_attention(line)
    else:
        parsed = [[line, 1.0]]

    tokenized = tokenizer([text for text, _ in parsed],
                          truncation=False, add_special_tokens=False)["input_ids"]

    chunks = []
    chunk = PromptChunk()
    token_count = 0
    last_comma = -1

    def next_chunk(is_last=False):
        """puts current chunk into the list of results and produces the next one - empty;
        if is_last is true, tokens <end-of-text> tokens at the end won't add to token_count"""
        nonlocal token_count
        nonlocal last_comma
        nonlocal chunk

        if is_last:
            token_count += len(chunk.tokens)
        else:
            token_count += chunk_length

        to_add = chunk_length - len(chunk.tokens)
        if to_add > 0:
            chunk.tokens += [id_end] * to_add
            chunk.multipliers += [1.0] * to_add

        chunk.tokens = [id_start] + chunk.tokens + [id_end]
        chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

        last_comma = -1
        chunks.append(chunk)
        chunk = PromptChunk()

    for tokens, (text, weight) in zip(tokenized, parsed):
        if text == 'BREAK' and weight == -1:
            next_chunk()
            continue

        position = 0
        while position < len(tokens):
            token = tokens[position]

            if token == comma_token:
                last_comma = len(chunk.tokens)

            # this is when we are at the end of alloted 75 tokens for the current chunk, and the current token is not a comma. opts.comma_padding_backtrack
            # is a setting that specifies that if there is a comma nearby, the text after the comma should be moved out of this chunk and into the next.
            elif comma_padding_backtrack != 0 and len(
                    chunk.tokens) == chunk_length and last_comma != -1 and len(
                    chunk.tokens) - last_comma <= comma_padding_backtrack:
                break_location = last_comma + 1

                reloc_tokens = chunk.tokens[break_location:]
                reloc_mults = chunk.multipliers[break_location:]

                chunk.tokens = chunk.tokens[:break_location]
                chunk.multipliers = chunk.multipliers[:break_location]

                next_chunk()
                chunk.tokens = reloc_tokens
                chunk.multipliers = reloc_mults

            if len(chunk.tokens) == chunk_length:
                next_chunk()

            embedding, embedding_length_in_tokens = find_embedding_at_position(tokens, position)
            if embedding is None:
                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1
                continue

            emb_len = int(embedding.vec.shape[0])
            if len(chunk.tokens) + emb_len > chunk_length:
                next_chunk()

            chunk.fixes.append(PromptChunkFix(len(chunk.tokens), embedding))

            chunk.tokens += [0] * emb_len
            chunk.multipliers += [weight] * emb_len
            position += embedding_length_in_tokens

    if chunk.tokens or not chunks:
        next_chunk(is_last=True)

    return chunks, token_count


re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)
re_AND = re.compile(r"\bAND\b")
re_weight = re.compile(r"^(.*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")
re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def find_embedding_at_position(tokens, offset):
    token = tokens[offset]
    # TODO:
    ids_lookup = dict()
    possible_matches = ids_lookup.get(token, None)

    if possible_matches is None:
        return None, None

    for ids, embedding in possible_matches:
        if tokens[offset:offset + len(ids)] == ids:
            return embedding, len(ids)

    return None, None


def empty_chunk():
    """creates an empty PromptChunk and returns it"""

    chunk = PromptChunk()
    chunk.tokens = [id_start] + [id_end] * (chunk_length + 1)
    chunk.multipliers = [1.0] * (chunk_length + 2)
    return chunk


def process_tokens(remade_batch_tokens, batch_multipliers, text_encoder, device=torch.device("cpu")):
    """
    sends one single prompt chunk to be encoded by transformers neural network.
    remade_batch_tokens is a batch of tokens - a list, where every element is a list of tokens; usually
    there are exactly 77 tokens in the list. batch_multipliers is the same but for multipliers instead of tokens.
    Multipliers are used to give more or less weight to the outputs of transformers network. Each multiplier
    corresponds to one token.
    """
    tokens = torch.asarray(remade_batch_tokens).to(device)

    # this is for SD2: SD1 uses the same token for padding and end of text, while SD2 uses different ones.
    if id_end != id_pad:
        for batch_pos in range(len(remade_batch_tokens)):
            index = remade_batch_tokens[batch_pos].index(id_end)
            tokens[batch_pos, index + 1:tokens.shape[1]] = id_pad

    z = text_encoder(tokens, output_hidden_states=-1).last_hidden_state

    # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
    batch_multipliers = torch.asarray(batch_multipliers).to(device)
    original_mean = z.mean()
    z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
    new_mean = z.mean()
    z = z * (original_mean / new_mean)

    return z


PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])


class PromptChunk:
    """
    This object contains token ids, weight (multipliers:1.4) and textual inversion embedding info for a chunk of prompt.
    If a prompt is short, it is represented by one PromptChunk, otherwise, multiple are necessary.
    Each PromptChunk contains an exact amount of tokens - 77, which includes one for start and end token,
    so just 75 tokens from prompt.
    """

    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []
