"""Define supported LLMs. To support an LLM, we need to allow for activation locations in LLMs to be specified in a uniform manner for the user. Internally, we need to map each model's architecture to a common format.

E.g., output of layer 1's MLP should always be uniformly translated from an activation location to a specific LLM. Each of these LLMs will have a function taking in an ActivationLocation and output'ing the NNSight formatting for this.

To do so, we will just implement the `get_mapping` method that will define a dictionary mapping from general component names ('attn', 'mlp') to model-specific names.

"""

from utils.intervention_llm import InterventionLLM


class Qwen25(InterventionLLM):
    """
    Qwen2ForCausalLM(
      (model): Qwen2Model(
        (embed_tokens): Embedding(151936, 896)
        (layers): ModuleList(
          (0-23): 24 x Qwen2DecoderLayer(
            (self_attn): Qwen2Attention(
              (q_proj): Linear(in_features=896, out_features=896, bias=True)
              (k_proj): Linear(in_features=896, out_features=128, bias=True)
              (v_proj): Linear(in_features=896, out_features=128, bias=True)
              (o_proj): Linear(in_features=896, out_features=896, bias=False)
            )
            (mlp): Qwen2MLP(
              (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
              (up_proj): Linear(in_features=896, out_features=4864, bias=False)
              (down_proj): Linear(in_features=4864, out_features=896, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
            (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
          )
        )
        (norm): Qwen2RMSNorm((896,), eps=1e-06)
        (rotary_emb): Qwen2RotaryEmbedding()
      )
      (lm_head): Linear(in_features=896, out_features=151936, bias=False)
      (generator): Generator(
        (streamer): Streamer()
      )
    )
    """
    @staticmethod
    def get_mapping():
        return {
            "attn": "self_attn",
            "mlp": "mlp",
        }

    @property
    def has_chat_template(self):
        return True

class Llama3(InterventionLLM):
    """
    LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(128256, 4096)
        (layers): ModuleList(
          (0-31): 32 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
              (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
              (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
              (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
              (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            )
            (mlp): LlamaMLP(
              (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
              (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
              (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
            (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
          )
        )
        (norm): LlamaRMSNorm((4096,), eps=1e-05)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
      (generator): Generator(
        (streamer): Streamer()
      )
    )
    """

    @staticmethod
    def get_mapping():
        return {
            "attn": "self_attn",
            "mlp": "mlp",
        }

    @property
    def has_chat_template(self):
        return True

class Gemma2(InterventionLLM):
    """
    Gemma2ForCausalLM(
      (model): Gemma2Model(
        (embed_tokens): Embedding(256000, 2304, padding_idx=0)
        (layers): ModuleList(
          (0-25): 26 x Gemma2DecoderLayer(
            (self_attn): Gemma2Attention(
              (q_proj): Linear(in_features=2304, out_features=2048, bias=False)
              (k_proj): Linear(in_features=2304, out_features=1024, bias=False)
              (v_proj): Linear(in_features=2304, out_features=1024, bias=False)
              (o_proj): Linear(in_features=2048, out_features=2304, bias=False)
            )
            (mlp): Gemma2MLP(
              (gate_proj): Linear(in_features=2304, out_features=9216, bias=False)
              (up_proj): Linear(in_features=2304, out_features=9216, bias=False)
              (down_proj): Linear(in_features=9216, out_features=2304, bias=False)
              (act_fn): PytorchGELUTanh()
            )
            (input_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
            (post_attention_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
            (pre_feedforward_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
            (post_feedforward_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
          )
        )
        (norm): Gemma2RMSNorm((2304,), eps=1e-06)
        (rotary_emb): Gemma2RotaryEmbedding()
      )
      (lm_head): Linear(in_features=2304, out_features=256000, bias=False)
    )
    """

    @staticmethod
    def get_mapping():
        return {
            "attn": "self_attn",
            "mlp": "mlp",
        }

    @property
    def has_chat_template(self):
        return True
