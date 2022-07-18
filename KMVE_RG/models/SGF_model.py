import torch
import torch.nn as nn
import numpy as np

import sys

sys.path.append('../')
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from torch.autograd import Variable
from modules.new_model_utils import SemanticEmbedding, classfication


class SGF(nn.Module):
    def __init__(self, args, tokenizer):
        super(SGF, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.classfication_layers = classfication()
        print('vocabulary size:', self.tokenizer.get_vocab_size())
        self.forward = self._forward_inference

    def __str__(self):
        """
        该方法中必须有一个return
        调用该方法的时机是print对象时
        return的内容就是print打印的内容
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def _forward_inference(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0, _, kmve_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, _, kmve_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)  # 两个feature concate
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        kmve = torch.cat((kmve_0, kmve_1), dim=1)
        if mode == 'train':
            # output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            output, _ = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            kmve_output = self.classfication_layers(kmve)
            return output, kmve_output
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            kmve_output = self.classfication_layers(kmve)
        elif mode == 'evaluate':
            output, first_sentence, first_attmap, first_sentence_probs = \
                self.encoder_decoder(fc_feats, att_feats, mode='evaluate')
            kmve_output = self.classfication_layers(kmve)
            return output, kmve_output, first_sentence, first_attmap, first_sentence_probs
        else:
            raise ValueError
        return output, kmve_output

