"""Frozen external SID adapter using the repository's encoder/decoder mixin."""
from genrec.tokenizer import AbstractTokenizer, EncoderDecoderTokenizerMixin


class FrozenSIDTokenizer(EncoderDecoderTokenizerMixin, AbstractTokenizer):
    def __init__(self, item2tokens, capacities, max_history=50):
        config = dict(explicit_causal_examples=True, n_user_tokens=0,
                      max_item_seq_len=max_history)
        super().__init__(config, None)
        self.capacities = list(capacities)
        if not capacities or any(type(cap) is not int or cap < 1 for cap in capacities) or max_history < 1:
            raise ValueError('Invalid fixed capacities or history length')
        self.n_user_tokens = 0
        self.eos_token = sum(capacities) + 1
        self.item2tokens = {raw: tuple(tokens) for raw, tokens in item2tokens.items()}
        if not self.item2tokens:
            raise ValueError('The eligible item catalog cannot be empty')
        if len(set(self.item2tokens.values())) != len(self.item2tokens):
            raise ValueError('Full SIDs must uniquely identify items')
        self.prefix_children = {}
        for tokens in self.item2tokens.values():
            if len(tokens) != len(capacities):
                raise ValueError('Incorrect item token length')
            offset = 1
            for token, capacity in zip(tokens, capacities):
                if type(token) is not int or not offset <= token < offset + capacity:
                    raise ValueError('Token outside its codebook interval')
                offset += capacity
            seq = (*tokens, self.eos_token)
            for depth in range(len(seq)):
                self.prefix_children.setdefault(seq[:depth], set()).add(seq[depth])

    @property
    def n_digit(self):
        return len(self.capacities)

    @property
    def vocab_size(self):
        return self.eos_token + 1

    def _token_single_item(self, item):
        # Unavailable historical events occupy their original four positions,
        # but are padding with attention=0. No extra vocabulary is learned.
        return self.item2tokens.get(item, (self.padding_token,) * self.n_digit)

    def _tokenize_once(self, example):
        if not example.get('item_seq') or example['item_seq'][-1] not in self.item2tokens:
            raise ValueError('An unavailable target cannot have training/evaluation labels')
        return super()._tokenize_once(example)

    def allowed_next_tokens(self, decoder_prefix):
        """For catalog-constrained TIGER generation; prefix includes start 0."""
        prefix = tuple(int(v) for v in decoder_prefix)
        if not prefix or prefix[0] != 0:
            raise ValueError('Decoder prefix must start with padding/start token 0')
        key = prefix[1:]
        if key not in self.prefix_children:
            raise ValueError('Decoder prefix is outside the eligible catalog')
        return sorted(self.prefix_children[key])
