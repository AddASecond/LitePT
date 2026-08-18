# Robotruck Occ Viewer — Scene Package Schema (`robotruck_occ_scene/v1`)

本格式是前端 `tools/occ_viewer` 的唯一输入约定。路径全部相对 **scene 根目录**（或将来 Mongo/GridFS/S3 上的逻辑前缀），**不绑定本机绝对路径**；落库时只需把 `uri` 改写成对象存储 key / GridFS id，前端用同一字段解析。

---

## 1. 包目录（文件系统落地时）

```text
<scene_root>/
  index.json                 # scene 清单（可作 Mongo scene 文档）
  SCHEMA.md                  # 可选：本说明副本
  frames/
    <timestamp>/
      meta.json              # frame 文档（可作 Mongo frame 文档）
      occ_ijk.i32.bin
      occ_labels.u8.bin
      occ_centers.f32.bin
      occ_counts.i32.bin
      points_xyz.f32.bin     # 可选
      points_labels.u8.bin
      points_lidar_id.u8.bin
      cameras/
        camera1.jpg
        ...
```

Mongo 建议：

| 集合 | 文档 | blob |
|------|------|------|
| `occ_scenes` | = `index.json` 内容 | — |
| `occ_frames` | = 每帧 `meta.json` 内容，带 `scene_id` | — |
| GridFS / S3 | `assets.*.uri` 指向的二进制与图片 | 按 uri 取 |

---

## 2. `index.json`（Scene）

```json
{
  "schema_version": "robotruck_occ_scene/v1",
  "scene_id": "<clip_id>",
  "clip_id": "<clip_id>",
  "created_at": "2026-08-18T02:00:00Z",
  "defaults": {
    "occ_voxel": 0.2,
    "roi": { "x": [-24, 24], "y": [-25, 150], "z": [-5, 3] },
    "vehicle_frame": {
      "x": "lateral (+ numeric right in data)",
      "y": "forward",
      "z": "up"
    }
  },
  "taxonomy": {
    "fine": {
      "n": 22,
      "names": ["Car", "...", "Sidewalk"],
      "colors_rgb": [[229,25,25], "..."]
    },
    "coarse": {
      "names": ["dynamic", "static", "freespace", "noise"],
      "colors_rgb": [[230,64,64], [64,160,255], [72,200,96], [160,160,160]],
      "fine_to_coarse": [0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1,2,2,2,2,2]
    },
    "lidar_ids": {
      "ids": [1, 2, 14],
      "colors_rgb": [[40,180,255], [40,220,40], [255,40,40]]
    }
  },
  "frames": [
    {
      "frame_id": "1784423017401143040",
      "timestamp": "1784423017401143040",
      "meta_uri": "frames/1784423017401143040/meta.json",
      "n_occ": 351793,
      "n_points": 200000
    }
  ]
}
```

`fine_to_coarse`：长度 22，取值 `0=dynamic, 1=static, 2=freespace, 3=noise`。

粗分类映射（与前端一致）：

| coarse | fine ids | names |
|--------|----------|-------|
| dynamic | 0–6, 11, 12 | Car…Pedestrian, Bicycle, Motorcycle |
| static | 7–10, 13–16 | Sign…Cone, Building…Curb |
| freespace | 17–21 | Road…Sidewalk |
| noise | label∉[0,21] | 非法/未标注 |

---

## 3. `frames/<ts>/meta.json`（Frame）

所有资产用 **相对 scene 根的 `uri`**（不是只写文件名）。前端：`assetUrl = sceneBase + "/" + uri`；Mongo：把 `uri` 换成 `s3://…` / `gridfs:…` 即可。

```json
{
  "schema_version": "robotruck_occ_frame/v1",
  "scene_id": "<clip_id>",
  "frame_id": "<timestamp>",
  "timestamp": "<timestamp>",
  "coordinate": {
    "frame": "vehicle",
    "x": "lateral",
    "y": "forward",
    "z": "up"
  },
  "grid": {
    "voxel": 0.2,
    "x_range": [-30.0, 30.0],
    "y_range": [-200.0, 400.0],
    "z_range": [-5.0, 20.0],
    "shape": [300, 3000, 125],
    "origin": [-30.0, -200.0, -5.0],
    "index_rule": "ijk = floor((p - origin) / voxel); center = origin + (ijk + 0.5) * voxel"
  },
  "stats": {
    "n_occ": 351793,
    "n_static_roi": 0,
    "n_vis_points": 0,
    "n_points_exported": 200000
  },
  "assets": {
    "occupancy": {
      "n": 351793,
      "ijk": { "uri": "frames/<ts>/occ_ijk.i32.bin", "dtype": "int32", "shape": [351793, 3], "byte_order": "little" },
      "labels": { "uri": "frames/<ts>/occ_labels.u8.bin", "dtype": "uint8", "shape": [351793] },
      "centers": { "uri": "frames/<ts>/occ_centers.f32.bin", "dtype": "float32", "shape": [351793, 3] },
      "counts": { "uri": "frames/<ts>/occ_counts.i32.bin", "dtype": "int32", "shape": [351793] }
    },
    "points": {
      "n": 200000,
      "xyz": { "uri": "frames/<ts>/points_xyz.f32.bin", "dtype": "float32", "shape": [200000, 3] },
      "labels": { "uri": "frames/<ts>/points_labels.u8.bin", "dtype": "uint8", "shape": [200000] },
      "lidar_id": { "uri": "frames/<ts>/points_lidar_id.u8.bin", "dtype": "uint8", "shape": [200000] }
    },
    "cameras": [
      {
        "name": "camera1",
        "image": {
          "uri": "frames/<ts>/cameras/camera1.jpg",
          "mime": "image/jpeg",
          "width": 1920,
          "height": 1080
        },
        "K": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
        "dist5": [k1, k2, p1, p2, k3],
        "T_c_v": [16 floats, row-major 4x4],
        "T_v_c": [16 floats, row-major 4x4]
      }
    ]
  },
  "ego_pose": { },
  "taxonomy_ref": "see index.json taxonomy"
}
```

### 二进制布局

| 文件 | dtype | 布局 |
|------|-------|------|
| `occ_ijk.i32.bin` | int32 LE | `N×3` 展平：`ix,iy,iz,...` |
| `occ_labels.u8.bin` | uint8 | `N`，Waymo fine id `0..21` |
| `occ_centers.f32.bin` | float32 LE | `N×3`：`x,y,z` 车体坐标（格心） |
| `occ_counts.i32.bin` | int32 LE | `N` |
| `points_xyz.f32.bin` | float32 LE | `N×3` 车体 `x,y,z` |
| `points_labels.u8.bin` | uint8 | `N` fine id |
| `points_lidar_id.u8.bin` | uint8 | `N`，常见 `{1,2,14}` |

`points` 可为 `null`（未导出点云时）。

---

## 4. 前端解析规则

1. `GET {sceneBase}/index.json`
2. 选帧 → `GET {sceneBase}/{meta_uri}`
3. 任意资产：`url = absolute(uri) ? uri : sceneBase + "/" + uri`
4. Occupancy 显示：用 `assets.occupancy.ijk` + `grid`；点云用 `assets.points.*`
5. 兼容旧包：若无 `assets`，回退 `occupancy.ijk` 等相对 **frame 目录** 的旧字段

本地预览：

```bash
.venv_smoke/bin/python tools/occ_viewer/serve.py \
  --scene <scene_root> --host 0.0.0.0 --port 8765
# http://127.0.0.1:8765/?scene=/scene
```

---

## 5. 生产命令示例

```bash
export PYTHONPATH=./
.venv_smoke/bin/python tools/export_robotruck_occ_scene.py \
  --clip stop_1784423032302844849_vehicle-V002-20260719_090818 \
  --stride 2 --max-frames 2 --reuse-pred --occ-voxel 0.2 \
  --export-points --max-export-points 200000 --aggregate-static
```
