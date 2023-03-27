# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import RobertaModel, RobertaPreTrainedModel


from transformers.activations import ACT2FN, gelu



from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig

from transformers.utils import logging


from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from time import time
import numpy as np
import scipy



ROBERTA_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top. """, ROBERTA_START_DOCSTRING)

class RobertaForMaskedLM2(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]


    def init_alterings(self, P,P_rowspaces, Ws, alpha, direction ):
        self.P = P
        self.Ws = Ws
        self.alpha = alpha
        self.direction = direction
        self.P_rowspaces = P_rowspaces






    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead2(config)

        self.init_weights()
        self.P = None
        self.Ws = None
        self.P_rowspaces = None
        self.alpha = None
        self.direction = None
        self.used_device = None


    def get_rowspace_projection(self, current_W: torch.Tensor) -> torch.Tensor:
        """
        :param W: the matrix over its nullspace to project
        :return: the projection matrix over the rowspace
        """

        #current_W = W.cpu()
        if np.allclose(current_W, 0):
            w_basis = torch.zeros_like(current_W.T)
        else:
            w_basis = scipy.linalg.orth(current_W.T)  # orthogonal basis

        P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace

        return P_W

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def get_projection_to_intersection_of_nullspaces(self, rowspace_projection_matrices, input_dim: int):
        print(len(rowspace_projection_matrices))
        I = np.eye(input_dim)
        Q = np.sum(rowspace_projection_matrices, axis=0)
        P = I - self.get_rowspace_projection(Q)

        return P


    def get_all_steps(self, representations_to_alter, n_P):
        list_rpz_neutralizing = []
        list_h_t = []

        for i, representation in enumerate(representations_to_alter):

            current_list_h_t = [representation]
            #list_h_t_bar.append(representations_to_alter)
            rpz_neutralizing = torch.clone(representation)
            rowspace_projections = []
            for i in range(n_P):
                W = self.Ws[n_P][i]
                P_rowspace_wi = self.get_rowspace_projection(W)  # projection to W's rowspace
                rowspace_projections.append(P_rowspace_wi)
                P = self.get_projection_to_intersection_of_nullspaces(rowspace_projections, 1024)
                rpz_neutralizing = (P.dot(rpz_neutralizing.T)).T
                current_list_h_t.append(rpz_neutralizing)
            list_rpz_neutralizing.append(rpz_neutralizing)
            list_h_t.append(current_list_h_t)

        return list_rpz_neutralizing, list_h_t



    def compare_proj_2(self, representations_to_alter, n_P):
        neutral, list_rpz = self.get_all_steps(representations_to_alter, n_P)
        counter_repz = torch.zeros(representations_to_alter.shape).to(self.device)

        for i, representation in enumerate(representations_to_alter):
            list_h_t_bar = []
            list_h_t = [representation]
            # list_h_t_bar.append(representations_to_alter)
            rpz_neutralizing = torch.clone(representation)

            for k in range(n_P):
                current_rowspace = self.get_rowspace_projection(self.Ws[n_P][k])
                # currrent_rpz = rpz_neutralizing @ self.P_rowspaces[n_P][k]
                currrent_rpz_1 = rpz_neutralizing @ current_rowspace

                #P = self.get_projection_to_intersection_of_nullspaces(self.P_rowspaces[n_P][k:], 1024)
                current_rpz_2 = list_rpz[i][k]
                print("\n\n##########\n")
                print(currrent_rpz_1)
                print(current_rpz_2)
                print(currrent_rpz_1 == current_rpz_2)

    def compare_proj(self, representations_to_alter, n_P):


        for i, representation in enumerate(representations_to_alter):
            rowspace_projections = []
            original_rpz = torch.clone(representation)

            print(f"original_rpz :\n {original_rpz}")


            W = self.Ws[n_P][0]

            P_rowspace_wi = self.get_rowspace_projection(W)

            rowspace_projections.append(P_rowspace_wi)

            P = self.get_projection_to_intersection_of_nullspaces(rowspace_projections, 1024)

            rpz_1 = (P.dot(original_rpz.T)).T
            print(f"rpz_1 :\n {rpz_1}")

            rpz_2 = original_rpz @ (torch.eye(1024).to(self.device) - self.P_rowspaces[n_P][0])
            print(f"rpz_2 :\n {rpz_2}")


            print("\n\n##########################\nSTEP 2 \n##########################\n\n")

            W = self.Ws[n_P][1]

            P_rowspace_wi = self.get_rowspace_projection(W)

            rowspace_projections.append(P_rowspace_wi)

            P = self.get_projection_to_intersection_of_nullspaces(rowspace_projections, 1024)

            rpz_0 = (P.dot(original_rpz.T)).T
            print(f"rpz_1 :\n {rpz_1}")

            rpz_1 = (P.dot(original_rpz.T)).T
            print(f"rpz_1 :\n {rpz_1}")

            rpz_2 = rpz_2 @ (torch.eye(1024).to(self.device) - self.P_rowspaces[n_P][1])
            print(f"rpz_2 :\n {rpz_2}")


            quit()





    def alter_represention(self, representations_to_alter, n_P):

        #print(f"np : {n_P}")
        #print(f"size W : {len(self.Ws[n_P])}")
        #print(f"size row P : {len(self.P_rowspaces[n_P])}")

        counter_repz = torch.zeros(representations_to_alter.shape).to(self.device)

        for i, representation in enumerate(representations_to_alter):

            #print(f'\n\n RPZ : {i} \n\n')

            list_h_t_bar = []
            list_h_t = [representation]
            #list_h_t_bar.append(representations_to_alter)
            rpz_neutralizing = torch.clone(representation)

            for k in range(n_P):
                current_rowspace = (torch.eye(1024).to(self.device) - self.P_rowspaces[n_P][k])

                currrent_rpz = rpz_neutralizing @ current_rowspace
                #print(currrent_rpz)

                list_h_t.append(currrent_rpz)
                list_h_t_bar.append(rpz_neutralizing - currrent_rpz)
                rpz_neutralizing = currrent_rpz

            rpz_deneutralized = torch.clone(rpz_neutralizing)


            for k in range(n_P):

                currrent_rpz = list_h_t_bar[-k-1]
                rpz_neutral = list_h_t[-k-2]
                scal = rpz_neutral @  self.Ws[n_P][-k-1].T
                coeff = -torch.ones(scal.shape).to(self.device)
                coeff[torch.where(torch.sign(scal) == torch.sign(torch.tensor(self.direction)))] = 1


                rpz_deneutralized += self.alpha*coeff*currrent_rpz

            counter_repz[i] = rpz_deneutralized

        return counter_repz




    def alter_represention_2(self, representations_to_alter, n_P):

        time_0 = time()

        current_P = self.P[n_P]
        current_Ws = self.Ws[n_P]

        counter_repz = torch.zeros(representations_to_alter.shape).to(self.device)
        print('lmlm')
        print(representations_to_alter.shape)

        for i, sentence_encoded in enumerate(representations_to_alter):

            print(f"im: {sentence_encoded.shape}")


            neutral_rpz = sentence_encoded @ current_P #projection sur l'espace nul

            feature_rpz = torch.zeros(neutral_rpz.shape).to(self.device) #Projection sur l'espace ortogonal


            for w_vect in current_Ws:
                scal = sentence_encoded @ w_vect
                coeff = -torch.ones(scal.shape).to(self.device)
                coeff[torch.where(torch.sign(scal) == torch.sign(torch.tensor(self.direction)))] = 1
                w_proj_mat = torch.tensor(self.get_rowspace_projection(w_vect)).to(self.device)
                prof_feature = sentence_encoded @ w_proj_mat
                feature_rpz += coeff * prof_feature

            time_3 = time()

            counter_repz[i] = neutral_rpz + self.alpha * feature_rpz #multiplicaiton par le facteur alpha

        return counter_repz

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings


    #@add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        #tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        n_P = None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        #print("outputs", outputs)
        sequence_output = outputs[0]
        #print("sequence_output", sequence_output)
        #print("sequence_output.shape", sequence_output.shape)

        #self.compare_proj(sequence_output, n_P)


        sequence_output = self.alter_represention(sequence_output, n_P)
        #quit()


        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




class RobertaLMHead2(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):


        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


