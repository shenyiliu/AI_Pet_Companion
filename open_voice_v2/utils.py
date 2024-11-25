import torch
import os
import sys
from os import PathLike
import urllib.parse
from pathlib import Path

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
repo_dir = os.path.join(project_dir, "OpenVoice")
sys.path.append(repo_dir)
from openvoice.api import BaseSpeakerTTS, ToneColorConverter, OpenVoiceBaseClass
from melo.utils import get_text_for_tts_infer
import unidic.download

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
        # max_len = torch.tensor(1024)  # 默认是None,换成1024代替一下？
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