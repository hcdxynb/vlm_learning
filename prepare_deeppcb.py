"""将 DeepPCB 转换为 YOLO 检测数据。依赖 Pillow（pip install pillow）。"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


NAMES = ("open", "short", "mousebite", "spur", "copper", "pin-hole")
BASE = Path(__file__).resolve().parent


def read_samples(source, filename):
    samples = []
    for line in (source / filename).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        image_name, label_name = line.split()
        image_path = source / image_name
        image_path = image_path.with_name(image_path.stem + "_test.jpg")
        label_path = source / label_name
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"缺少图片或标注：{image_path}, {label_path}")
        samples.append((image_path, label_path))
    return samples


def convert_label(image_path, label_path):
    with Image.open(image_path) as image:
        width, height = image.size
        image.verify()
    rows = []
    counts = Counter()
    for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            xmin, ymin, xmax, ymax, category = map(int, line.split())
        except ValueError as error:
            raise ValueError(f"{label_path}:{line_number} 标注应为五个整数") from error
        if not (1 <= category <= 6 and 0 <= xmin < xmax <= width
                and 0 <= ymin < ymax <= height):
            raise ValueError(f"{label_path}:{line_number} 类别或边界框无效：{line}")
        values = ((xmin + xmax) / (2 * width), (ymin + ymax) / (2 * height),
                  (xmax - xmin) / width, (ymax - ymin) / height)
        rows.append(f"{category - 1} " + " ".join(f"{value:.8f}" for value in values))
        counts[category - 1] += 1
    return "\n".join(rows) + ("\n" if rows else ""), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE / "DeepPCB-master/PCBData")
    parser.add_argument("--output", type=Path, default=BASE / "datasets/deeppcb")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="从官方 trainval 中随机划入验证集的比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true", help="仅检查数据，不生成文件")
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio 必须在 0 与 1 之间")
    source = args.source.resolve()
    output = args.output.resolve()
    trainval = read_samples(source, "trainval.txt")
    test = read_samples(source, "test.txt")
    all_images = [image.resolve() for image, _ in trainval + test]
    if len(set(all_images)) != len(all_images):
        raise ValueError("官方清单中出现重复图片或训练测试交集")
    all_names = [image.stem for image, _ in trainval + test]
    if len(set(all_names)) != len(all_names):
        raise ValueError("存在同名图片，无法安全写入平铺目录")
    random.Random(args.seed).shuffle(trainval)
    val_count = round(len(trainval) * args.val_ratio)
    if not 0 < val_count < len(trainval):
        parser.error("划分后训练集或验证集为空，请调整 --val-ratio")
    splits = {"train": trainval[val_count:], "val": trainval[:val_count], "test": test}

    # 先校验全部数据，避免错误标注造成部分转换。
    prepared = {}
    for split, samples in splits.items():
        prepared[split] = []
        counts = Counter()
        for image, label in samples:
            text, sample_counts = convert_label(image, label)
            prepared[split].append((image, text))
            counts.update(sample_counts)
        print(f"{split}: {len(samples)} images, {sum(counts.values())} boxes")
        print("  " + ", ".join(f"{name}={counts[index]}" for index, name in enumerate(NAMES)))
    if args.check_only:
        print("检查通过，未写入文件。")
        return

    # 使用新目录，重复运行时可指定另一个 --output。
    output.mkdir(parents=True, exist_ok=False)
    for split, samples in prepared.items():
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for image, text in samples:
            shutil.copy2(image, image_dir / image.name)
            (label_dir / f"{image.stem}.txt").write_text(text, encoding="utf-8")
    yaml = (f"path: '{output.as_posix().replace(chr(39), chr(39) * 2)}'\n"
            "train: images/train\nval: images/val\ntest: images/test\nnames:\n")
    yaml += "".join(f"  {index}: {name}\n" for index, name in enumerate(NAMES))
    (output / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"转换完成：{output / 'data.yaml'}")


if __name__ == "__main__":
    main()
