import torch
import os
import sys

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
repo_dir = os.path.join(project_dir, "OpenVoice")
sys.path.append(repo_dir)
from openvoice.api import BaseSpeakerTTS, ToneColorConverter, OpenVoiceBaseClass

class OVOpenVoiceBase(torch.nn.Module):
    """
    Base class for both TTS and voice tone conversion model: constructor is same for both of them.
    """

    def __init__(self, voice_model: OpenVoiceBaseClass):
        super().__init__()
        self.voice_model = voice_model
        for par in voice_model.model.parameters():
            par.requires_grad = False


class OVOpenVoiceTTS(OVOpenVoiceBase):
    """
    Constructor of this class accepts BaseSpeakerTTS object for speech generation and wraps it's 'infer' method with forward.
    """

    def get_example_input(self):
        stn_tst = self.voice_model.get_text("this is original text", self.voice_model.hps, False)
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        speaker_id = torch.LongTensor([1])
        noise_scale = torch.tensor(0.667)
        length_scale = torch.tensor(1.0)
        noise_scale_w = torch.tensor(0.6)
        return (
            x_tst,
            x_tst_lengths,
            speaker_id,
            noise_scale,
            length_scale,
            noise_scale_w,
        )

    def forward(self, x, x_lengths, sid, noise_scale, length_scale, noise_scale_w):
        return self.voice_model.model.infer(x, x_lengths, sid, noise_scale, length_scale, noise_scale_w)


class OVOpenVoiceConverter(OVOpenVoiceBase):
    """
    Constructor of this class accepts ToneColorConverter object for voice tone conversion and wraps it's 'voice_conversion' method with forward.
    """

    def get_example_input(self):
        y = torch.randn([1, 513, 238], dtype=torch.float32)
        y_lengths = torch.LongTensor([y.size(-1)])
        target_se = torch.randn(*(1, 256, 1))
        source_se = torch.randn(*(1, 256, 1))
        tau = torch.tensor(0.3)
        return (y, y_lengths, source_se, target_se, tau)

    def forward(self, y, y_lengths, sid_src, sid_tgt, tau):
        return self.voice_model.model.voice_conversion(y, y_lengths, sid_src, sid_tgt, tau)