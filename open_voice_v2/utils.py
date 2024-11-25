import torch
import os
import re
import sys
import time
from os import PathLike
import numpy as np
import urllib.parse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn as nn
import openvino as ov
from tqdm import tqdm
import soundfile
from melo.download_utils import load_or_download_config # load_or_download_model
# from melo.models import SynthesizerTrn
from melo.split_utils import split_sentence

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
download_dir = os.path.join(project_dir, "download")
repo_dir = os.path.join(project_dir, "OpenVoice")
sys.path.append(repo_dir)
from openvoice.api import BaseSpeakerTTS, ToneColorConverter, OpenVoiceBaseClass
# from melo.utils import get_text_for_tts_infer
# import unidic.download
from melo.text import cleaned_text_to_sequence, get_bert
from melo.text.cleaner import clean_text
from melo import commons

zh_mix_en_bert_dir = os.path.join(download_dir, "bert-base-multilingual-uncased")
en_bert_dir = os.path.join(download_dir, "bert-base-uncased")
zh_mix_en_bert_model = AutoModelForMaskedLM.from_pretrained(zh_mix_en_bert_dir)
zh_mix_en_tokenizer = AutoTokenizer.from_pretrained(zh_mix_en_bert_dir)
en_bert_model = AutoModelForMaskedLM.from_pretrained(en_bert_dir)
en_bert_tokenizer = AutoTokenizer.from_pretrained(en_bert_dir)
core = ov.Core()


def get_bert_feature(model, tokenizer, text, word2ph, device="cpu"):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # import pdb; pdb.set_trace()
    # assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


def get_text_for_tts_infer(text, language_str, hps, device, symbol_to_id=None):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, symbol_to_id)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    if getattr(hps.data, "disable_bert", False):
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))
    else:
        st = time.time()
        if language_str == "ZH_MIX_EN":
            model = zh_mix_en_bert_model
            tokenizer = zh_mix_en_tokenizer
        elif language_str == "EN":
            model = en_bert_model
            tokenizer = en_bert_tokenizer
        else:
            raise  Exception(language_str + "not supported")
        # bert = get_bert(norm_text, word2ph, language_str, device)
        bert = get_bert_feature(model, tokenizer, text, word2ph, device)
        et = time.time()
        print("[INFO] tts bert duration: ", et - st)
        del word2ph
        assert bert.shape[-1] == len(phone), phone

        if language_str == "ZH":
            bert = bert
            ja_bert = torch.zeros(768, len(phone))
        elif language_str in ["JP", "EN", "ZH_MIX_EN", 'KR', 'SP', 'ES', 'FR', 'DE', 'RU']:
            ja_bert = bert
            bert = torch.zeros(1024, len(phone))
        else:
            raise NotImplementedError()

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language

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
    def __init__(self, voice_model: OpenVoiceBaseClass):
        super().__init__(voice_model)
        self.language = self.voice_model.language
        self.hps = self.voice_model.hps
        self.device = self.voice_model.device
        self.symbol_to_id = self.voice_model.symbol_to_id

    def get_example_input(self):
        if self.language == "ZH" or self.language == "ZH_MIX_EN":
            example_text = "今天天气真好，我们一起出去吃饭吧。"
            speaker_id = 1
        else:
            example_text = "this is original text"
            speaker_id = 0
        bert, ja_bert, phones, tones, lang_ids = get_text_for_tts_infer(
            example_text,
            self.language,
            self.hps,
            self.device,
            self.symbol_to_id
        )
        x_tst = phones.unsqueeze(0)
        tones = tones.unsqueeze(0)
        lang_ids = lang_ids.unsqueeze(0)
        bert = bert.unsqueeze(0)
        ja_bert = ja_bert.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)])
        del phones
        speakers = torch.LongTensor([speaker_id])
        noise_scale = torch.tensor(0.6)
        length_scale = torch.tensor(1.0)
        noise_scale_w = torch.tensor(0.8)
        # max_len = torch.tensor(256) # 静态图，最大输入长度256
        sdp_ratio = torch.tensor(0.2)
        return (
            x_tst,
            x_tst_lengths,
            speakers,
            tones,
            lang_ids,
            bert,
            ja_bert,
            noise_scale,
            length_scale,
            noise_scale_w,
            # None,
            sdp_ratio
        )

    def forward(
       self,
       x,
       x_lengths,
       speakers,
       tones,
       lang_ids,
       bert,
       ja_bert,
       noise_scale,
       length_scale,
       noise_scale_w,
       # max_length = None,
       sdp_ratio=torch.tensor(0.2)
    ):
        return self.voice_model.model.infer(
            x,
            x_lengths,
            speakers,
            tones,
            lang_ids,
            bert,
            ja_bert,
            noise_scale,
            length_scale,
            noise_scale_w,
            # max_length,
            sdp_ratio=sdp_ratio
        )


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


def download_file(
    url: PathLike,
    filename: PathLike = None,
    directory: PathLike = None,
    silent: bool = False,
) -> PathLike:
    """
    Download a file from a url and save it to the local filesystem. The file is saved to the
    current directory by default, or to `directory` if specified. If a filename is not given,
    the filename of the URL will be used.

    :param url: URL that points to the file to download
    :param filename: Name of the local file to save. Should point to the name of the file only,
                     not the full path. If None the filename from the url will be used
    :param directory: Directory to save the file to. Will be created if it doesn't exist
                      If None the file will be saved to the current working directory
    :param show_progress: If True, show an TQDM ProgressBar
    :param silent: If True, do not print a message if the file already exists
    :param timeout: Number of seconds before cancelling the connection attempt
    :return: path to downloaded file
    """
    from tqdm.notebook import tqdm_notebook
    import requests

    filename = filename or Path(urllib.parse.urlparse(url).path).name
    chunk_size = 16384  # make chunks bigger so that not too many updates are triggered for Jupyter front-end

    filename = Path(filename)
    if len(filename.parts) > 1:
        raise ValueError(
            "`filename` should refer to the name of the file, excluding the directory. "
            "Use the `directory` parameter to specify a target directory for the downloaded file."
        )

    # create the directory if it does not exist, and add the directory to the filename
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / Path(filename)

    try:
        response = requests.get(url=url, headers={"User-agent": "Mozilla/5.0"}, stream=True)
        response.raise_for_status()
    except (
        requests.exceptions.HTTPError
    ) as error:  # For error associated with not-200 codes. Will output something like: "404 Client Error: Not Found for url: {url}"
        raise Exception(error) from None
    except requests.exceptions.Timeout:
        raise Exception(
            "Connection timed out. If you access the internet through a proxy server, please "
            "make sure the proxy is set in the shell from where you launched Jupyter."
        ) from None
    except requests.exceptions.RequestException as error:
        raise Exception(f"File downloading failed with error: {error}") from None

    # download the file if it does not exist, or if it exists with an incorrect file size
    filesize = int(response.headers.get("Content-length", 0))
    if not filename.exists() or (os.stat(filename).st_size != filesize):
        with open(filename, "wb") as file_object:
            for chunk in response.iter_content(chunk_size):
                file_object.write(chunk)
    else:
        if not silent:
            print(f"'{filename}' already exists.")

    response.close()

    return filename.resolve()


class OpenVinoTTS(nn.Module):
    def __init__(self,
                 language,
                 ov_xml_path,
                 pt_device='cpu',
                 ov_device="AUTO",
                 use_hf=True,
                 config_path=None,
                 ckpt_path=None):
        super().__init__()
        if pt_device == 'auto':
            pt_device = 'cpu'
            if torch.cuda.is_available(): pt_device = 'cuda'
            if torch.backends.mps.is_available(): pt_device = 'mps'
        if 'cuda' in pt_device:
            assert torch.cuda.is_available()
        ov_model = core.read_model(ov_xml_path)
        self.model = ov.compile_model(ov_model, device_name=ov_device)

        hps = load_or_download_config(language, use_hf=use_hf,
                                      config_path=config_path)
        symbols = hps.symbols

        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = pt_device
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language  # we support a ZH_MIX_EN model

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2,
                    noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None,
                    format=None, position=None, quiet=False, ):
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            # device = self.device
            bert, ja_bert, phones, tones, lang_ids = get_text_for_tts_infer(
                t, language, self.hps, self.device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.unsqueeze(0)
                tones = tones.unsqueeze(0)
                lang_ids = lang_ids.unsqueeze(0)
                bert = bert.unsqueeze(0)
                ja_bert = ja_bert.unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)])
                del phones
                speakers = torch.LongTensor([speaker_id])
                # outputs = self.model.infer(
                #     x_tst,
                #     x_tst_lengths,
                #     speakers,
                #     tones,
                #     lang_ids,
                #     bert,
                #     ja_bert,
                #     sdp_ratio=sdp_ratio,
                #     noise_scale=noise_scale,
                #     noise_scale_w=noise_scale_w,
                #     length_scale=1. / speed,
                # )
                length_scale = torch.tensor(1.0 / speed)
                outputs = self.model((
                    x_tst,
                    x_tst_lengths,
                    speakers,
                    tones,
                    lang_ids,
                    bert.contiguous(),
                    ja_bert.contiguous(),
                    noise_scale,
                    length_scale,
                    noise_scale_w,
                    sdp_ratio
                ))
                # audio = outputs[0][0, 0].data.cpu().float().numpy()
                audio = outputs[0][0, 0]
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                #
            audio_list.append(audio)
        # torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list,
                                        sr=self.hps.data.sampling_rate,
                                        speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate,
                                format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)