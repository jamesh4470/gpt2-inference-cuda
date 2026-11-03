import math, os, sys, time, urllib.request
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from safetensors.torch import load_file

kernels = load(name="gpt2_kernels", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.cu")],
               extra_cuda_cflags=["-O3", "-DGLOG_USE_GLOG_EXPORT"]) # the define is needed by Arch's PyTorch headers (which use glog)

@dataclass
class GPTConfig:
    max_sequence_length: int = 1024
    vocabulary_size: int = 50257
    num_layers: int = 12
    num_heads: int = 12 # per layer
    embedding_size: int = 768

class LayerNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, hidden_states):
        return kernels.layernorm(hidden_states, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.query_key_value_projection = nn.Linear(config.embedding_size, 3 * config.embedding_size) # q, k, v projections for all heads in one matmul
        self.output_projection = nn.Linear(config.embedding_size, config.embedding_size)

    def forward(self, hidden_states, layer_cache, current_position):
        batch_size, sequence_length, embedding_size = hidden_states.shape # (4, T, 768)
        head_size = embedding_size // self.num_heads # 768 / 12 = 64

        # split the concatenated q k v matrices
        queries, keys, values = self.query_key_value_projection(hidden_states).split(embedding_size, dim=2)

        # split into heads, (batch_size, num_heads, sequence_length, head_size)
        queries, keys, values = [tensor.view(batch_size, sequence_length, self.num_heads, head_size).transpose(1, 2) for tensor in (queries, keys, values)]

        # store this step's keys/values in the cache, then attend over every position seen so far
        end_position = current_position + sequence_length
        layer_cache[0, :, :, current_position:end_position] = keys
        layer_cache[1, :, :, current_position:end_position] = values
        keys, values = layer_cache[0, :, :, :end_position], layer_cache[1, :, :, :end_position]

        # score by multiplying query vectors with key vectors
        attention_weights = kernels.softmax(queries @ keys.transpose(-2, -1), 1.0 / math.sqrt(head_size)) # (batch_size, num_heads, sequence_length, end_position)

        # multiply each softmax score by their value vectors, and then merge heads
        output = (attention_weights @ values).transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_size) # put the head outputs back side by side

        # project 12 heads into one matrix
        return self.output_projection(output)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # computes x @ weight.T + bias
        self.expand_projection = nn.Linear(config.embedding_size, 4 * config.embedding_size)
        self.output_projection = nn.Linear(4 * config.embedding_size, config.embedding_size)

    # prompt pass: hidden_states = (batch_size, prompt_ids_size, 768)
    # decode pass: hidden_states = (batch_size, 1, 768)
    # first project to higher dimension, then GELU, then project back
    def forward(self, hidden_states):
        return self.output_projection(F.gelu(self.expand_projection(hidden_states), approximate="tanh"))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_layernorm = LayerNorm(config.embedding_size)
        self.attention = CausalSelfAttention(config)
        self.mlp_layernorm = LayerNorm(config.embedding_size)
        self.mlp = MLP(config)

    # hidden_states -> layernorm -> attention -> layernorm -> MLP
    def forward(self, hidden_states, layer_cache, current_position):
        hidden_states = hidden_states + self.attention(self.attention_layernorm(hidden_states), layer_cache, current_position)
        hidden_states = hidden_states + self.mlp(self.mlp_layernorm(hidden_states))
        return hidden_states

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.embedding_size)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.embedding_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.final_layernorm = LayerNorm(config.embedding_size)

    # token_ids is (batch_size, sequence_length) tokens starting at current_position; returns logits for the token after the last one
    def forward(self, token_ids, kv_cache, current_position):
        # look up each token's embedding vector and add a positional vector to it
        positions = torch.arange(current_position, current_position + token_ids.size(1), device=token_ids.device)
        hidden_states = self.token_embedding(token_ids) + self.position_embedding(positions)

        # pass through tower of decoder blocks 
        for block, layer_cache in zip(self.blocks, kv_cache):
            hidden_states = block(hidden_states, layer_cache, current_position)

        # final layernorm, keep only the last position (for the prompt pass), and turn its hidden vector into a logit vector
        return self.final_layernorm(hidden_states)[:, -1] @ self.token_embedding.weight.T # output layer shares weights with the token embedding

    @classmethod
    def from_pretrained(cls, path="gpt2.safetensors"):
        if not os.path.exists(path): # ~550MB of float32 weights from the HuggingFace hub
            urllib.request.urlretrieve("https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors", path)
        checkpoint = load_file(path)
        model = cls(GPTConfig())

        # maps our module names to the short names OpenAI's checkpoint uses
        checkpoint_names = {"token_embedding": "wte", "position_embedding": "wpe", "blocks": "h", "final_layernorm": "ln_f",
                            "attention_layernorm": "ln_1", "attention": "attn", "mlp_layernorm": "ln_2",
                            "query_key_value_projection": "c_attn", "expand_projection": "c_fc", "output_projection": "c_proj"}
        # OpenAI's checkpoint uses "Conv1D" layers that store weights as (in, out); nn.Linear wants (out, in)
        transposed = ("c_attn.weight", "c_proj.weight", "c_fc.weight")

        state_dict = {}
        for name in model.state_dict(): # e.g. blocks.0.attention.output_projection.weight -> h.0.attn.c_proj.weight
            checkpoint_name = ".".join(checkpoint_names.get(part, part) for part in name.split("."))
            weight = checkpoint[checkpoint_name]
            state_dict[name] = weight.t() if checkpoint_name.endswith(transposed) else weight
        model.load_state_dict(state_dict)
        return model

    # temperature sharpens or flattens the distribution.
    # T < 1, sharpens distribution. makes likely tokens even more likely
    # T > 1, flattens distribution. makes unlikely tokens more likely
    # T = 0 means choosing greedily, though you can't really divide by 0, so use argmax instead
    # 
    # only the prompt pass runs in parallel. the decode passes run sequentially
    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None):
        config = self.config

        if token_ids.dim() == 1: # a single sequence (sequence_length,): treat it as a batch of one, (1, sequence_length)
            token_ids = token_ids.unsqueeze(0)

        assert token_ids.size(1) + max_new_tokens <= config.max_sequence_length

        # per layer: keys and values for every position so far, (2, batch_size, num_heads, max_sequence_length, head_size)
        batch_size = token_ids.size(0)
        head_size = config.embedding_size // config.num_heads
        kv_cache = torch.empty(config.num_layers, 2, batch_size, config.num_heads, config.max_sequence_length, head_size, device=token_ids.device)
        current_position = 0
        for _ in range(max_new_tokens):
            # array of unprocessed tokens for all batches. result is 2d
            # only the prompt pass has unprocessed_tokens hold more than one token
            unprocessed_tokens = token_ids[:, current_position:]

            # logits vector contains scores for every single token. needs to be passed through softmax later.
            logits = self.forward(unprocessed_tokens, kv_cache, current_position) / temperature # first the whole prompt, then one token at a time

            current_position = token_ids.size(1)

            # keep only the top_k most likely tokens. discard all others
            if top_k is not None:
                kth_largest_logits = torch.topk(logits, top_k).values[:, [-1]]
                logits[logits < kth_largest_logits] = -float("inf")

            probabilities = F.softmax(logits, dim=-1)

            # next_token_ids has shape (batch_size, 1). the new tokens to be appended to the current token_ids array
            next_token_ids = torch.multinomial(probabilities, num_samples=1) # for choosing at random
            # next_token_ids = torch.argmax(probabilities, dim=-1, keepdim=True) # for choosing greedily. prone to go into loops.

            token_ids = torch.cat((token_ids, next_token_ids), dim=1)

        return token_ids

if __name__ == "__main__":
    import tiktoken
    torch.manual_seed(1337)
    tokenizer = tiktoken.get_encoding("gpt2")

    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "Hello, I'm a language model,"

    model = GPT.from_pretrained().cuda()
    prompt_token_ids = torch.tensor([tokenizer.encode(prompt)] * 4, device="cuda") # the prompt translated into token indices, repeated for a batch of 4
    start_time = time.time()
    output_token_ids = model.generate(prompt_token_ids, max_new_tokens=100, top_k=50)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    for sequence in output_token_ids:
        print(tokenizer.decode(sequence.tolist()), "\n---------------")

    print(f"generated {output_token_ids.numel() - prompt_token_ids.numel()} tokens in {elapsed_time:.2f}s")
