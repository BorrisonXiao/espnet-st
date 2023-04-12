#!/usr/bin/env python3
#  2022, University of Stuttgart;  Pavel Denisov
#  2023, Johns Hopkins University; Cihan Xiao
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers Decoder."""

import copy
import logging
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

try:
    from fairseq.models.bart import BARTModel

    is_fairseq_available = True
except ImportError:
    is_fairseq_available = False


class FairseqTransformersDecoder(AbsDecoder):
    """Fairseq Transformers Decoder.

    Args:
        encoder_output_size: dimension of encoder attention
        model_name_or_path: Fairseq Transformers model name
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        model_name_or_path: str,
    ):
        assert check_argument_types()
        super().__init__()

        if not is_fairseq_available:
            raise ImportError(
                "`fairseq` is not available. Please install it via `pip install"
                " fairseq` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_fairseq.sh`."
            )

        bart = BARTModel.from_pretrained(
            model_name_or_path,  # e.g. /home/cxiao7/research/iwslt2023/dialect/mbart
            checkpoint_file='checkpoint_best.pt',
            bpe='sentencepiece',
            sentencepiece_model=f'{model_name_or_path}/sentence.bpe.model')

        model = bart.model
        self.decoder_class = model.__class__.__name__

        if hasattr(model, "model"):
            self.decoder = model.model.decoder
        else:
            self.decoder = model.decoder

        self.cls_head = model.classification_heads
        self.model_name_or_path = model_name_or_path

        self.decoder_pretrained_params = copy.deepcopy(
            self.decoder.state_dict())
        self.cls_head_pretrained_params = copy.deepcopy(
            self.cls_head.state_dict())

        if encoder_output_size != self.decoder.embed_dim:
            self.linear_in = torch.nn.Linear(
                encoder_output_size, self.decoder.embed_dim
            )
        else:
            self.linear_in = torch.nn.Identity()

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad: input tensor (batch, maxlen_out, #mels)
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        args = {}

        if self.decoder.__class__.__name__ == "MBartDecoder" or self.decoder_class == "BARTModel":
            ys_in_pad[:, 0] = 2

        args["prev_output_tokens"] = ys_in_pad
        # mask = (~make_pad_mask(ys_in_lens)).to(ys_in_pad.device).float()
        # args["attention_mask"] = mask

        args["src_lengths"] = hlens.to(torch.int)
        hs_mask = (make_pad_mask(hlens)).to(hs_pad[0].device).float()
        args["encoder_out"] = {"encoder_out": [self.linear_in(
            hs_pad[0]).cuda()], "encoder_padding_mask": [hs_mask]}

        x = self.decoder(**args)[0]

        olens = ys_in_lens.to(torch.int)
        return x, olens

    def reload_pretrained_parameters(self):
        self.decoder.load_state_dict(self.decoder_pretrained_params)
        self.cls_head.load_state_dict(self.cls_head_pretrained_params)
        logging.info("Pretrained Transformers model parameters reloaded!")
