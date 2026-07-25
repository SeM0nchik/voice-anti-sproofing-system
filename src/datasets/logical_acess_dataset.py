import torchaudio
from tqdm.auto import tqdm
from pathlib import Path

from src.datasets.base_dataset import BaseDataset

class LogicalAcessDataset(BaseDataset):

    def __init__(self, protocol_path, audio_path, name,  *args, **kwargs):
        index = self._create_index(protocol_path, audio_path, name)

        super().__init__(index, *args, **kwargs)


    def _create_index(self, protocol_path, audio_path, name):
        index = []
        data_path = Path(audio_path) / f"ASVspoof2019_LA_{name}"  / "flac"

        print(f"Creating new {name} Dataset")

        with open(protocol_path) as f:
            for line in tqdm(f):
                line = line.strip()
                parts = line.split()
                obj_path = data_path / f"{parts[1]}.flac"
                obj_label = 1 if parts[-1] == "bonafide" else 0
                index.append({"path": str(obj_path), "label": obj_label})

        return index


    def load_object(self, path):
        audio, _ = torchaudio.load(path)
        return audio