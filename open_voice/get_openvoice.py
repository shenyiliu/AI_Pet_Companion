import sys
import os
from pathlib import Path

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
repo_dir = Path(os.path.join(project_dir, "OpenVoice"))

if not repo_dir.exists():
    os.system("git clone --depth 1 https://github.com/myshell-ai/OpenVoice")
    orig_english_path = Path(os.path.join(repo_dir, "openvoice/text/_orig_english.py"))
    english_path = Path(os.path.join(repo_dir, "openvoice/text/english.py"))
    english_path.rename(orig_english_path)

    with orig_english_path.open("r") as f:
        data = f.read()
        data = data.replace("unidecode", "anyascii")
        with english_path.open("w") as out_f:
            out_f.write(data)
# append to sys.path so that modules from the repo could be imported
sys.path.append(str(repo_dir))


# fix a problem with silero downloading and installing
with Path(os.path.join(repo_dir, "openvoice/se_extractor.py")).open("r") as orig_file:
    data = orig_file.read()
    data = data.replace("method=\"silero\"", "method=\"silero:3.0\"")
    with Path(os.path.join(repo_dir, "openvoice/se_extractor.py")).open("w") as out_f:
            out_f.write(data)

