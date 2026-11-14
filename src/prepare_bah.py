from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "BAH"
DEFAULT_OUTPUT = PROJECT_ROOT / "bah_data.json"


def to_seconds(time_string):
    timestamp = datetime.strptime(time_string, "%H:%M:%S.%f")
    return (
        timestamp.hour * 3600
        + timestamp.minute * 60
        + timestamp.second
        + timestamp.microsecond / 1_000_000
    )


def get_video_ids(video_path):
    path_parts = video_path.split("/")
    sample_id = int(path_parts[1])
    question_id = int(path_parts[3].split("_")[2])
    return sample_id, question_id


def build_split_lookup(split_dir):
    split_lookup = {}
    for split_name in ("train", "val", "test"):
        split_file = split_dir / f"{split_name}.txt"
        with split_file.open(encoding="utf-8") as file:
            for line in file:
                parts = line.rstrip("\n").split(",", 2)
                if not parts or not parts[0]:
                    continue
                sample_id, question_id = get_video_ids(parts[0])
                split_lookup[(sample_id, question_id)] = split_name
    return split_lookup


def prepare_bah():
    data_dir = DEFAULT_DATA_DIR
    output_path = DEFAULT_OUTPUT
    annotation_path = data_dir / "video_annotation_transcript.yaml"
    split_lookup = build_split_lookup(data_dir / "split")

    with annotation_path.open("r", encoding="utf-8") as file:
        raw_data = yaml.load(file, Loader=yaml.FullLoader)

    rows = []
    missing_split_count = 0
    for video_path, video in raw_data.items():
        sample_id, question_id = get_video_ids(video_path)
        split_type = split_lookup.get((sample_id, question_id), "unknown")
        if split_type == "unknown":
            missing_split_count += 1

        detailed_ah = video.get("time_detailed_ah", [])
        ah_times = [to_seconds(value) for value in detailed_ah[0]] if detailed_ah else []

        for chunk_num, chunk in enumerate(video["transcript"]["chunks"]):
            chunk_start, chunk_end = chunk["timestamp"]
            ah_annotated_count = sum(
                chunk_start < ah_time < chunk_end for ah_time in ah_times
            )
            rows.append(
                {
                    "id": sample_id,
                    "question_id": question_id,
                    "chunk_num": int(chunk_num),
                    "text": chunk["text"],
                    "has_ah": int(ah_annotated_count > 0),
                    "ah_annotated_count": ah_annotated_count,
                    "split_type": split_type,
                }
            )

    video_df = pd.DataFrame(rows)
    video_df = video_df.sort_values(["id", "question_id", "chunk_num"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_df.to_json(output_path, orient="records", force_ascii=False)

    print(f"Saved {len(video_df)} transcript chunks to {output_path}")
    if missing_split_count:
        print(f"Warning: {missing_split_count} videos were not found in the train/val/test splits")


if __name__ == "__main__":
    prepare_bah()
