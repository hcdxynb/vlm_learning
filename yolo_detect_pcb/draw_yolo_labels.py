"""在图片上绘制 YOLO 标注框。"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


NAMES = ("open", "short", "mousebite", "spur", "copper", "pin-hole")
COLORS = ("red", "blue", "green", "orange", "purple", "cyan")
BASE = Path(__file__).resolve().parent

image_path = BASE / "datasets" / "deeppcb" / "images" / "test" / "00041200_test.jpg"
label_path = BASE / "datasets" / "deeppcb" / "labels" / "test" / "00041200_test.txt"
output_path = BASE / "outputs" / "00041200_test_label.jpg"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=image_path, help="原始图片")
    parser.add_argument("--label", type=Path, default=label_path, help="YOLO 标签文件")
    parser.add_argument("--output", type=Path, default=output_path, help="输出图片，默认保存为 *_boxed")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for line_number, line in enumerate(args.label.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{args.label}:{line_number} 应包含 5 个字段")
        class_id, x, y, box_width, box_height = map(float, values)
        class_id = int(class_id)
        if not 0 <= class_id < len(NAMES):
            raise ValueError(f"{args.label}:{line_number} 类别编号无效")
        left = (x - box_width / 2) * width
        top = (y - box_height / 2) * height
        right = (x + box_width / 2) * width
        bottom = (y + box_height / 2) * height
        color = COLORS[class_id]
        draw.rectangle((left, top, right, bottom), outline=color, width=3)
        draw.text((left, top), NAMES[class_id], fill=color)

    output = args.output or args.image.with_name(args.image.stem + "_boxed" + args.image.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"已保存：{output}")


if __name__ == "__main__":
    main()
