#!/usr/bin/env python3
#  2021, University of Stuttgart;  Pavel Denisov
#  2023, Johns Hopkins University; Cihan Xiao
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Fairseq Transformers PostEncoder."""

import copy
import logging
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError

try:
    from fairseq.models.bart import BARTModel

    is_fairseq_available = True
except ImportError:
    is_fairseq_available = False


class FairseqTransformersEncoder(AbsEncoder):
    """Fairseq Transformers PostEncoder."""

    def __init__(
        self,
        input_size: int,
        model_name_or_path: str,
        length_adaptor_n_layers: int = 0,
        lang_token_id: int = -1,
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()

        if not is_fairseq_available:
            raise ImportError(
                "`fairseq` is not available. Please install it via `pip install"
                " fairseq` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_fairseq.sh`."
            )

        model = BARTModel.from_pretrained(
            model_name_or_path,  # e.g. /home/cxiao7/research/iwslt2023/dialect/mbart
            checkpoint_file='checkpoint_best.pt',
            bpe='sentencepiece',
            sentencepiece_model=f'{model_name_or_path}/sentence.bpe.model')

        if hasattr(model, "model"):
            self.transformer = model.model.encoder
            self.decoder_class = model.model.__class__.__name__
        else:
            self.transformer = model.encoder
            self.decoder_class = model.__class__.__name__

        self.lang_token_embed = None

        embed_dim = self.transformer.embed_tokens.embedding_dim

        if hasattr(self.transformer, "embed_tokens"):
            if lang_token_id != -1:
                self.lang_token_embed = (
                    self.transformer.embed_tokens(torch.tensor(lang_token_id))
                    .detach()
                    .cpu()
                )
            del self.transformer.embed_tokens
        if hasattr(self.transformer, "wte"):
            if lang_token_id != -1:
                self.lang_token_embed = (
                    self.transformer.wte(torch.tensor(
                        lang_token_id)).detach().cpu()
                )
            del self.transformer.wte
        if hasattr(self.transformer, "word_embedding"):
            if lang_token_id != -1:
                self.lang_token_embed = (
                    self.transformer.word_embedding(
                        torch.tensor(lang_token_id))
                    .detach()
                    .cpu()
                )
            del self.transformer.word_embedding
        if hasattr(model, "embeddings") and hasattr(
            model.embeddings, "word_embeddings"
        ):
            if lang_token_id != -1:
                self.lang_token_embed = (
                    model.embeddings.word_embeddings(
                        torch.tensor(lang_token_id))
                    .detach()
                    .cpu()
                )

        if self.lang_token_embed is not None and hasattr(
            self.transformer, "embed_scale"
        ):
            self.lang_token_embed *= self.transformer.embed_scale

        self.pretrained_params = copy.deepcopy(self.transformer.state_dict())

        self.use_inputs_embeds = False

        self.linear_in = torch.nn.Linear(
            input_size, embed_dim
        )
        # TODO: Add an if to control the in-out dimension of the linear layer

        # self.alc = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, dropout=0.1)

        # Length Adaptor as in https://aclanthology.org/2021.acl-long.68.pdf

        if length_adaptor_n_layers > 0:
            length_adaptor_layers = []
            for _ in range(length_adaptor_n_layers):
                length_adaptor_layers.append(
                    torch.nn.Conv1d(input_size, input_size, 2, 2)
                )
                length_adaptor_layers.append(torch.nn.ReLU())
        else:
            length_adaptor_layers = [torch.nn.Identity()]

        self.length_adaptor = torch.nn.Sequential(*length_adaptor_layers)
        self.length_adaptor_ratio = 2**length_adaptor_n_layers

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        if input.size(1) < self.length_adaptor_ratio:
            raise TooShortUttError(
                f"has {input.size(1)} frames and is too short for subsampling "
                + f"(it needs at least {self.length_adaptor_ratio} frames), "
                + "return empty results",
                input.size(1),
                self.length_adaptor_ratio,
            )

        input = input.permute(0, 2, 1)
        input = self.length_adaptor(input)
        input = input.permute(0, 2, 1)

        breakpoint()

        input_lengths = (
            input_lengths.float().div(self.length_adaptor_ratio).floor().long()
        )

        input = self.linear_in(input)

        args = {}

        if self.lang_token_embed is not None:
            lang_token_embed = (
                self.lang_token_embed.unsqueeze(0)
                .unsqueeze(0)
                .repeat(input.size(0), 1, 1)
            )
            input = torch.cat(
                [lang_token_embed.to(input.device), input], dim=1)
            input_lengths = input_lengths + 1

        args["src_lengths"] = input_lengths
        args["token_embeddings"] = input
        # Just a dummy input for computing the positional embeddings
        args["src_tokens"] = (torch.ones(input.size(
            0), input.size(1)) * 100).long().to(input.device)
        # Mask the src_tokens with the pad token for out-of-length tokens
        args["src_tokens"][make_pad_mask(
            input_lengths)] = self.transformer.dictionary.pad()

        output = self.transformer(**args)["encoder_out"]

        return output, input_lengths, None

    def reload_pretrained_parameters(self):
        self.transformer.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Transformers model parameters reloaded!")

    def output_size(self) -> int:
        """Get the output size."""
        return self.transformer.layers[-1].final_layer_norm.normalized_shape[0]
