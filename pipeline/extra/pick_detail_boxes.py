"""
pick_detail_boxes.py

Công cụ hỗ trợ chọn toạ độ hộp bao (bounding box) cho các vùng chi tiết cao
(ăng-ten, RRU, cáp...) trên trụ BTS, để điền vào file JSON dùng với
--detail_regions_json của train_compact.py.

Cách dùng:
    python pick_detail_boxes.py --sparse /path/to/scene/sparse/0 --out preview.html

Sau đó mở preview.html (tải về máy nếu chạy trên Kaggle rồi mở bằng trình
duyệt) — đây là biểu đồ 3D tương tác (xoay/zoom/pan bằng chuột), rê chuột
vào từng điểm sẽ hiện toạ độ (x, y, z). Xoay tới góc nhìn thấy rõ cụm
ăng-ten/RRU/cáp trên đỉnh trụ, đọc khoảng toạ độ min/max bao quanh cụm đó
rồi điền vào file JSON (xem detail_regions_example.json).

Yêu cầu: pip install plotly (đã có sẵn trong hầu hết môi trường Kaggle;
nếu chưa có: !pip install -q plotly)

Đọc points3D bằng scene/colmap_loader.py có sẵn trong repo
graphdeco-inria/gaussian-splatting (script này cần chạy từ trong thư mục
gốc của repo đó, hoặc thêm đường dẫn repo vào PYTHONPATH).
"""

import argparse
import os
import sys

import numpy as np


def load_points3D(sparse_dir: str):
    bin_path = os.path.join(sparse_dir, "points3D.bin")
    txt_path = os.path.join(sparse_dir, "points3D.txt")
    from scene.colmap_loader import read_points3D_binary, read_points3D_text
    # Lưu ý: trong repo graphdeco-inria/gaussian-splatting, 2 hàm này trả thẳng
    # về tuple (xyz, rgb, errors) dạng numpy array — KHÔNG phải dict Point3D.
    if os.path.exists(bin_path):
        xyz, rgb, _errors = read_points3D_binary(bin_path)
    elif os.path.exists(txt_path):
        xyz, rgb, _errors = read_points3D_text(txt_path)
    else:
        raise FileNotFoundError(f"Không tìm thấy points3D.bin/.txt trong {sparse_dir}")

    return xyz, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", required=True, help="Thư mục chứa points3D.bin/.txt (thường là <scene>/sparse/0)")
    ap.add_argument("--out", default="preview.html", help="File HTML xuất ra để xem tương tác")
    ap.add_argument("--max_points", type=int, default=200_000, help="Giới hạn số điểm hiển thị (subsample) để nhẹ trình duyệt")
    args = ap.parse_args()

    xyz, rgb = load_points3D(args.sparse)
    print(f"Tổng số điểm sparse: {xyz.shape[0]}")

    if xyz.shape[0] > args.max_points:
        idx = np.random.choice(xyz.shape[0], args.max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    print("Bounding box toàn cảnh (x, y, z):")
    print("  min:", xyz.min(axis=0))
    print("  max:", xyz.max(axis=0))

    import plotly.graph_objects as go

    colors = ["rgb({},{},{})".format(*c) for c in rgb.astype(int)]
    fig = go.Figure(data=[go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=colors),
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
    )])
    fig.update_layout(
        title="Sparse point cloud — xoay/zoom để tìm cụm ăng-ten/RRU/cáp, "
              "rê chuột để đọc toạ độ điểm",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(args.out)
    print(f"Đã ghi {args.out} — mở file này bằng trình duyệt để xem 3D tương tác.")


if __name__ == "__main__":
    main()
