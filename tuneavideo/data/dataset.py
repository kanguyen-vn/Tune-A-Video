import decord

decord.bridge.set_bridge("torch")

from torch.utils.data import Dataset
from einops import rearrange

from pathlib import Path
from typing import Union, List
import collections

import pandas as pd


class TuneAVideoDataset(Dataset):
    def __init__(
        self,
        video_path: str,
        prompt: str,
        width: int = 512,
        height: int = 512,
        n_sample_frames: int = 8,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(
            range(self.sample_start_idx, len(vr), self.sample_frame_rate)
        )[: self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {"pixel_values": (video / 127.5 - 1.0), "prompt_ids": self.prompt_ids}

        return example


class TuneAVideoKineticsPretrainDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        mode: str,
        tokenizer=None,
        labels: List[Union[int, str]] = None,
        width: int = 512,
        height: int = 512,
        n_sample_frames: int = 8,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 1,
        n_per_class: int = -1,
    ):
        assert mode in [
            "train",
            "val",
            "test",
        ], f"mode must be train, val, or test; got {mode}"
        self.data_dir = Path(data_dir)
        if not isinstance(labels, collections.Sequence):
            labels = [labels]
        self.labels = labels
        self.data_csv = self.load_labels(mode)

        self.tokenizer = tokenizer
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.n_per_class = n_per_class

    def load_labels(self, mode: str):
        labels_csv = pd.read_csv(self.data_dir / f"{mode}.csv", header=None)
        labels_csv.columns = ["videopath", "id"]

        def change_path(entry):
            path = Path(entry)
            return self.data_dir / path.relative_to(path.parents[2])

        labels_csv["videopath"] = labels_csv["videopath"].apply(change_path)

        name2id_csv = pd.read_csv(self.data_dir / "name-to-id.csv", header=None)
        name2id_csv.columns = ["id", "label"]
        id2name = {row["id"]: row["label"] for _, row in name2id_csv.iterrows()}
        labels_csv["label"] = labels_csv["id"].apply(lambda x: id2name[x])

        if self.labels is not None:
            ids = []
            labels = []
            for label in self.labels:
                (ids if isinstance(label, int) else labels).append(label)

            labels_from_ids = [id2name[id] for id in ids]
            final_labels = list(set(labels + labels_from_ids))
            labels_csv = labels_csv[labels_csv["label"].isin(final_labels)]

        if self.n_per_class != -1:
            labels_csv = labels_csv.groupby("id").sample(
                n=self.n_per_class, random_state=42
            )

        return labels_csv

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, index):
        row = self.data_csv.iloc[[index]]
        videopath = str(row["videopath"].values[0])
        vr = decord.VideoReader(videopath, width=self.width, height=self.height)
        sample_index = list(
            range(self.sample_start_idx, len(vr), self.sample_frame_rate)
        )[: self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        prompt = row["label"].values[0]
        prompt_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        example = {"pixel_values": (video / 127.5 - 1.0), "prompt_ids": prompt_ids}

        return example
