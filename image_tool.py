# -*- coding: utf-8 -*-
import argparse
import os
import sys
import cv2
import numpy as np


def read_image(path):
       # 检查文件是否存在
    if not os.path.exists(path):
        print(f"错误：文件不存在 -> {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        print(f"错误：无法读取文件 {path}: {e}", file=sys.stderr)
        sys.exit(1)
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        print(f"错误：{path} 不是有效的图片文件", file=sys.stderr)
        sys.exit(1)
    return arr


def save_image(img, path):
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in (".jpg", ".jpeg") and len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        print(f"错误：保存图片失败 -> {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    print(f"已保存到: {os.path.abspath(path)}")


def file_size_str(size_bytes):
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def cmd_info(args):
    img = read_image(args.input)
    file_size = os.path.getsize(args.input)
    _, ext = os.path.splitext(args.input)
    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    if channels == 1:
        mode = "灰度(GRAY)"
    elif channels == 3:
        mode = "RGB"
    elif channels == 4:
        mode = "RGBA"
    else:
        mode = f"未知({channels}通道)"
    fmt = ext.upper().lstrip(".")
    if fmt == "JPG":
        fmt = "JPEG"
    print("===== 图片信息 =====")
    print(f"文件名    : {os.path.basename(args.input)}")
    print(f"图片格式  : {fmt}")
    print(f"宽度      : {w} 像素")
    print(f"高度      : {h} 像素")
    print(f"颜色模式  : {mode}")
    print(f"文件大小  : {file_size_str(file_size)}")


def cmd_resize(args):
    img = read_image(args.input)
    h, w = img.shape[:2]
    if args.width is not None and args.width <= 0:
        print("错误：宽度必须大于 0", file=sys.stderr)
        sys.exit(1)
    if args.height is not None and args.height <= 0:
        print("错误：高度必须大于 0", file=sys.stderr)
        sys.exit(1)
    if args.width is None and args.height is None:
        print("错误：必须至少指定 --width 或 --height", file=sys.stderr)
        sys.exit(1)
    elif args.width is None:
        new_h = args.height
        new_w = int(w * (new_h / h))
    elif args.height is None:
        new_w = args.width
        new_h = int(h * (new_w / w))
    else:
        new_w, new_h = args.width, args.height
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    save_image(resized, args.output)


def cmd_crop(args):
    img = read_image(args.input)
    h, w = img.shape[:2]
    l, t, r, b = args.left, args.top, args.right, args.bottom
    if not (0 <= l < r <= w and 0 <= t < b <= h):
        print(f"错误：裁剪区域越界或不合法。图片大小 {w}x{h}，给定 left={l}, top={t}, right={r}, bottom={b}", file=sys.stderr)
        sys.exit(1)
    cropped = img[t:b, l:r]
    save_image(cropped, args.output)


def cmd_rotate(args):
    img = read_image(args.input)
    if args.angle not in (90, 180, 270):
        print("错误：只支持 90、180、270 度", file=sys.stderr)
        sys.exit(1)
    if args.angle == 90:
        code = cv2.ROTATE_90_CLOCKWISE
    elif args.angle == 180:
        code = cv2.ROTATE_180
    else:
        code = cv2.ROTATE_90_COUNTERCLOCKWISE
    rotated = cv2.rotate(img, code)
    save_image(rotated, args.output)


def cmd_flip(args):
    img = read_image(args.input)
    if args.direction == "horizontal":
        code = 1
    else:
        code = 0
    flipped = cv2.flip(img, code)
    save_image(flipped, args.output)


def cmd_gray(args):
    img = read_image(args.input)
    channels = img.shape[2] if len(img.shape) == 3 else 1
    if channels == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_image(img, args.output)


def cmd_convert(args):
    img = read_image(args.input)
    _, out_ext = os.path.splitext(args.output)
    out_ext = out_ext.lower()
    if out_ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        print(f"错误：不支持的目标格式 {out_ext}", file=sys.stderr)
        sys.exit(1)
    save_image(img, args.output)


def build_parser():
    parser = argparse.ArgumentParser(description="基于 OpenCV 的图片处理命令行工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="读取并打印图片信息")
    p.add_argument("input", help="输入图片路径")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("resize", help="缩放图片")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.set_defaults(func=cmd_resize)

    p = sub.add_parser("crop", help="裁剪图片")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--left", type=int, required=True)
    p.add_argument("--top", type=int, required=True)
    p.add_argument("--right", type=int, required=True)
    p.add_argument("--bottom", type=int, required=True)
    p.set_defaults(func=cmd_crop)

    p = sub.add_parser("rotate", help="旋转图片 (90/180/270)")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--angle", type=int, required=True)
    p.set_defaults(func=cmd_rotate)

    p = sub.add_parser("flip", help="翻转图片")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--direction", required=True, choices=["horizontal", "vertical"])
    p.set_defaults(func=cmd_flip)

    p = sub.add_parser("gray", help="转灰度图")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_gray)

    p = sub.add_parser("convert", help="格式转换 (按输出后缀)")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_convert)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
