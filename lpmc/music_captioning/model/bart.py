### code reference: https://github.com/XinhaoMei/WavCaps/blob/master/captioning/models/bart_captioning.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lpmc.music_captioning.model.modules import AudioEncoder
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig

class BartCaptionModel(nn.Module):
    def __init__(self, n_mels=128, num_of_conv=6, sr=16000, duration=10, max_length=128, label_smoothing=0.1, bart_type="facebook/bart-base", audio_dim=768):
        super(BartCaptionModel, self).__init__()
        # non-finetunning case
        bart_config = BartConfig.from_pretrained(bart_type)
        self.tokenizer = BartTokenizer.from_pretrained(bart_type)
        self.bart = BartForConditionalGeneration(bart_config)
        
        self.n_sample = sr * duration
        self.hop_length = int(0.01 * sr) # hard coding hop_size
        self.n_frames = int(self.n_sample // self.hop_length)
        self.num_of_stride_conv = num_of_conv - 1
        self.n_ctx = int(self.n_frames // 2**self.num_of_stride_conv) + 1
        self.audio_encoder = AudioEncoder(
            n_mels = n_mels, # hard coding n_mel
            n_ctx = self.n_ctx, 
            audio_dim = audio_dim, 
            text_dim = self.bart.config.hidden_size,
            num_of_stride_conv = self.num_of_stride_conv
        )

        self.max_length = max_length
        self.loss_fct = nn.CrossEntropyLoss(label_smoothing= label_smoothing, ignore_index=-100)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.ls
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def forward_encoder(self, audio):
        audio_embs = self.audio_encoder(audio)
        encoder_outputs = self.bart.model.encoder(
            input_ids=None,
            inputs_embeds=audio_embs,
            return_dict=True
        )["last_hidden_state"]
        return encoder_outputs, audio_embs

    def forward_decoder(self, text, encoder_outputs):
        text = self.tokenizer(text,
                              padding='longest',
                              truncation=True,
                              max_length=self.max_length,
                              return_tensors="pt")
        input_ids = text["input_ids"].to(self.device)
        attention_mask = text["attention_mask"].to(self.device)

        decoder_targets = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100
        )

        decoder_input_ids = self.shift_tokens_right(
            decoder_targets, self.bart.config.pad_token_id, self.bart.config.decoder_start_token_id
        )

        decoder_outputs = self.bart(
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=attention_mask,
            inputs_embeds=None,
            labels=None,
            encoder_outputs=(encoder_outputs,),
            return_dict=True
        )
        lm_logits = decoder_outputs["logits"]
        loss = self.loss_fct(lm_logits.view(-1, self.tokenizer.vocab_size), decoder_targets.view(-1))
        return loss

    def forward(self, audio, text):
        encoder_outputs, _ = self.forward_encoder(audio)
        loss = self.forward_decoder(text, encoder_outputs)
        return loss

    def generate(self,
                 samples,
                 use_nucleus_sampling=False,
                 num_beams=5,
                 max_length=128,
                 min_length=2,
                 top_p=0.9,
                 repetition_penalty=1.0,
                 ):

        # self.bart.force_bos_token_to_be_generated = True
        audio_embs = self.audio_encoder(samples)
        encoder_outputs = self.bart.model.encoder(
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=audio_embs,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True)

        input_ids = torch.zeros((encoder_outputs['last_hidden_state'].size(0), 1)).long().to(self.device)
        input_ids[:, 0] = self.bart.config.decoder_start_token_id
        decoder_attention_mask = torch.ones((encoder_outputs['last_hidden_state'].size(0), 1)).long().to(self.device)
        if use_nucleus_sampling:
            outputs = self.bart.generate(
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                repetition_penalty=1.1)
        else:
            outputs = self.bart.generate(input_ids=None,
                                            attention_mask=None,
                                            decoder_input_ids=input_ids,
                                            decoder_attention_mask=decoder_attention_mask,
                                            encoder_outputs=encoder_outputs,
                                            head_mask=None,
                                            decoder_head_mask=None,
                                            inputs_embeds=None,
                                            decoder_inputs_embeds=None,
                                            use_cache=None,
                                            output_attentions=None,
                                            output_hidden_states=None,
                                            max_length=max_length,
                                            min_length=min_length,
                                            num_beams=num_beams,
                                            repetition_penalty=repetition_penalty)

        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions
