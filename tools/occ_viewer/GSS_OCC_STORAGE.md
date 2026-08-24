# LitePT Occ → GSS 落库方案（v2）

> 原则：与 GSS 一致——**元数据进 Mongo，大文件进数据盘**；  
> Occ 使用**独立命名空间**，且 **Mongo 表名 / 磁盘目录与原始 `raw_data_*` 层级一一对应**。

相关前端约定见 [SCHEMA.md](./SCHEMA.md)（`assets.*.uri` 仍用相对或绝对 URI，ingest 时改写）。

---

## 1. 原始 GSS 现状（对齐基准）

| 层 | 原始约定 |
|----|----------|
| DB | `perception_experiment` |
| Clip 表 | `raw_data_clips_<suffix>`，例：`raw_data_clips_lidar14_0731` |
| Frame 表 | `raw_data_frames_<suffix>`，例：`raw_data_frames_lidar14_0731` |
| 点云盘 | `/data/rawdata/lidar/{md5[0:2]}/{md5[2:4]}/{md5[4:]}.bin` |
| 图像盘 | `/data/rawdata/camera/{md5[0:2]}/{md5[2:4]}/{md5[4:]}.jpg` |
| 关联键 | Frame：`md5`（常与 `dependency.sensors.lidar_merge.md5` 相同）；Clip：`md5` / `bag_name` / `bag_path` |

`<suffix>` 表示数据切片名（`lidar14_0731`、`lidar14_0813`、`stop_riku`、空后缀的 `raw_data_frames` 等），**Occ 侧必须复用同一 suffix**，才能和原始表成对出现。

---

## 2. Occ 专用命名空间（Mongo）

仍落在 **`perception_experiment`**，但表前缀从 `raw_data_` 换成 **`occ_data_`**（专门的 OCC 数据，不写回 raw 表）。

| 原始 | Occ 对应（一对一改前缀） |
|------|--------------------------|
| `raw_data_clips_<suffix>` | **`occ_data_clips_<suffix>`** |
| `raw_data_frames_<suffix>` | **`occ_data_frames_<suffix>`** |

示例：

```text
raw_data_clips_lidar14_0731   →  occ_data_clips_lidar14_0731
raw_data_frames_lidar14_0731  →  occ_data_frames_lidar14_0731
raw_data_clips_lidar14_0813   →  occ_data_clips_lidar14_0813
raw_data_frames                →  occ_data_frames
```

可选（clip 集合级清单，类似部分 `*_groundtruths_*`）：

```text
occ_data_sets_<suffix>    # 记录本次 LitePT 跑批、ckpt、voxel、tag 列表
```

**禁止**：把 occ bin / pred 塞进 `raw_data_frames_*` 文档；raw 只保留传感器索引。

---

## 3. 大文件落盘：原始盘在哪、Occ 写哪

### 3.0 Mongo 没有「文件仓库」

- **MongoDB**：只存 Document（元数据 + **uri 字符串**）。  
- **大文件**：写在 RTGPU **已 attach 的数据卷**上；别人要读，必须挂**同一卷**。

### 3.1 原始盘到底在哪（RTGPU / sakura）

原始传感器 **不是**「Mongo 里的路径」，而是平台上的命名卷，用 `rtgpu volume attach` 挂进 notebook：

| 挂载点 | 卷名（RTGPU） | 容量（同事环境参考） | 典型内容 |
|--------|---------------|----------------------|----------|
| **`/data/rawdata`** | `h200-data-krk030-rawdata` | 6T，**基本已满** | **主原始库**：`lidar/`、`camera/`（md5 分片） |
| `/data/rawdata-2` | `h200-data-krk030-rawdata-2` | ~6T，约数百 G 空闲 | 扩展 raw / 其它数据 |
| `/data/rawdata-3` | `h200-data-krk030-rawdata-3` | ~4T，约 **2T 空闲** | 扩展（**适合新写 Occ**） |
| `/data/rawdata-4` | `h200-data-krk030-rawdata-4` | ~4T，约 1T+ 空闲 | 扩展 |
| `/data/zexi` 等 | 个人卷 | 视人而定 | 个人实验，**不要当团队 GSS 正式路径** |

挂载示例（把 workspace 换成自己的 notebook 名）：

```bash
RTGPU_CLUSTER=sakura ./rtgpu volume attach \
  <your-workspace-name> \
  h200-data-krk030-rawdata \
  --read-write
# → 出现 /data/rawdata

RTGPU_CLUSTER=sakura ./rtgpu volume attach \
  <your-workspace-name> \
  h200-data-krk030-rawdata-3 \
  --read-write
# → 出现 /data/rawdata-3
```

完成后 `ls /data` 应能看到 `rawdata`、`rawdata-2`…（未 attach 的目录本机不存在）。

**本仓库当前 notebook 实测**：通常只挂了 `/data/rawdata`（lidar + camera），**未挂** rawdata-2/3/4；主卷已接近 100% 满，**不宜再往 `/data/rawdata` 里堆 Occ**。

原始读路径（与现有脚本一致）：

```text
/data/rawdata/lidar/{md5[0:2]}/{md5[2:4]}/{md5[4:]}.bin
/data/rawdata/camera/{md5[0:2]}/{md5[2:4]}/{md5[4:]}.jpg
```

### 3.2 Occ 大文件写哪（修订）

| 方案 | 路径 | 何时用 |
|------|------|--------|
| **A′（推荐生产）** | **`/data/rawdata-3/occ/...`**（或 rawdata-2/4，选有空闲且团队约定的一卷） | 需先 `attach` 该卷；Mongo uri 写绝对路径；消费端同样 attach |
| ~~A 旧~~ | `/data/rawdata/occ/...` | **不推荐**：与 lidar 同卷但主库已满 |
| **B** | 平台新卷如 `h200-data-krk030-occ` → `/data/occdata` | 正式隔离时向平台申请 |
| **C** | `exp/robotruck/occ_scenes/...` | 仅开发；ingest 前再拷到 A′/B |

**推荐默认（有空闲扩展卷时）：**

```text
/data/rawdata-3/occ/lidar/{a}/{b}/{c}/     # 帧级，md5=frame/lidar_merge
  occ_ijk.i32.bin
  occ_labels.u8.bin
  occ_centers.f32.bin
  occ_counts.i32.bin
  points_*.bin                             # 可选

/data/rawdata-3/occ/clips/{a}/{b}/{c}/     # clip 级，md5=clip.md5
  static_agg/...
```

Mongo 示例 uri：

```text
/data/rawdata-3/occ/lidar/d1/fd/e97cd1204583098314b1cf899c4b/occ_ijk.i32.bin
```

相机仍读主库：

```text
/data/rawdata/camera/...
```

### 3.3 给 Matrix / 同事的「能用」条件

1. Document 在 Mongo（`occ_data_*`）。  
2. uri 指向 **真实挂载点**（如 `/data/rawdata-3/occ/...`），不要写未 attach 的路径。  
3. Matrix 后端或评测机执行同样的 `rtgpu volume attach ... h200-data-krk030-rawdata-3`（以及读相机所需的 `...-rawdata`）。  
4. 团队书面约定：**Occ 固定落在哪一卷**（建议 rawdata-3），避免每人写到自己的 `/data/zexi`。

### 3.4 总览

```text
RTGPU 卷 h200-data-krk030-rawdata     → /data/rawdata      → lidar/, camera/   (原始，满)
RTGPU 卷 h200-data-krk030-rawdata-3   → /data/rawdata-3    → occ/lidar/, occ/clips/  (建议 Occ)
Mongo perception_experiment           → occ_data_* 文档里的 uri 指向上述路径
```

---

## 4. Document 字段（保持可溯源到 raw）

### 4.1 `occ_data_frames_<suffix>`

与对应 `raw_data_frames_<suffix>` **同键对齐**：

| 字段 | 说明 |
|------|------|
| `md5` | **= raw frame.md5**（主键对齐） |
| `timestamp` | 同 raw |
| `clip_md5` | = raw `clip_md5` / 所属 clip.md5 |
| `bag_name` / `bag_md5` | 同 raw，便于按 bag 检索 |
| `tag` | 继承 raw + 追加 `litept` / `occ` / `semseg` 等 |
| `source` | `{ db, collection: "raw_data_frames_<suffix>", md5 }` |
| `model` | `{ name, ckpt, grid_size, occ_voxel }` |
| `grid` / `ego_pose` / `stats` | 同现有 scene meta |
| `assets.occupancy.*.uri` | `/data/rawdata/occ/lidar/{a}/{b}/{c}/occ_*.bin`（或方案 B 前缀） |
| `assets.points.*.uri` | 同上目录 |
| `assets.cameras[]` | 标定可冗余存；`image.uri` → `/data/rawdata/camera/...` |

### 4.2 `occ_data_clips_<suffix>`

| 字段 | 说明 |
|------|------|
| `md5` | **= raw clip.md5** |
| `bag_name` / `bag_path` | 同 raw |
| `frame_count` / 时间戳范围 | 同 raw 或本趟实际导出帧数 |
| `tag` | 继承 + `litept`/`occ` |
| `source` | 指向 `raw_data_clips_<suffix>` |
| `static_agg` | uris → `/data/rawdata/occ/clips/{a}/{b}/{c}/static_agg/` |
| `taxonomy` / `defaults` / `model` | clip 级共享 |

---

## 5. 训练 / 评测 Query 示例

```text
# 与某批原始 lidar14_0731 对齐的 Occ 帧
"perception_experiment:occ_data_frames_lidar14_0731",
{ "$and": [
    { "tag": "litept" },
    { "tag": "occ" },
    { "model.name": "litept-small-waymo" },
    { "grid.voxel": 0.2 }
] }
```

按 bag：

```text
{ "bag_name": "vehicle-V002-20260719_090818", "tag": "occ" }
```

用 **同一 `md5`** join raw：

```text
raw_data_frames_lidar14_0731.md5  ==  occ_data_frames_lidar14_0731.md5
```

---

## 6. 数据流

```text
raw_data_clips_<suffix> / raw_data_frames_<suffix>
        │ 读 lidar md5 → /data/rawdata/lidar/a/b/c.bin
        │ 读 camera md5 → /data/rawdata/camera/...
        ▼
LitePT infer + export_robotruck_occ_scene
        │
        ▼
/data/rawdata/occ/lidar/a/b/c/*     (frame blobs；需已挂载 rawdata)
/data/rawdata/occ/clips/a/b/c/static_agg
        │
        ▼ upsert
occ_data_clips_<suffix>
occ_data_frames_<suffix>
        │
        ├── Occ Viewer（uri 读 occdata / rawdata）
        └── 训练配置 DB:Collection + query
```

---

## 7. URI 书写约定

推荐绝对路径（集群内稳定）：

```text
/data/rawdata/occ/lidar/d1/fd/e97cd1204583098314b1cf899c4b/occ_ijk.i32.bin
```

或逻辑 scheme（ingest / viewer 解析）：

```text
rawdata://occ/lidar/d1/fd/e97cd1204583098314b1cf899c4b/occ_ijk.i32.bin
rawdata://camera/ee/16/71a3057a68ebc8169c591ec70b03.jpg
```

分片函数与 raw 完全一致：

```python
def md5_shards(md5: str) -> tuple[str, str, str]:
    return md5[:2], md5[2:4], md5[4:]
```

---

## 8. 落地步骤（建议）

1. 与平台确认：用方案 A（在已有 rawdata 挂载下建 `occ/`）还是方案 B（新 PVC 挂 `/data/occdata`）；并确认容量。  
2. 在**已挂载**路径创建 `occ/lidar`、`occ/clips`（勿写未挂载的目录）。  
3. 导出/ingest：按 frame `md5` 写 blob，uri 写入 Mongo。  
4. Upsert `occ_data_frames_<suffix>` / `occ_data_clips_<suffix>`。  
5. 本地 `exp/robotruck/occ_scenes/` 仅开发缓存；GSS 主键仍是 **md5 + suffix**。

---

## 9. 与旧提案差异（v1 → v2）

| 项 | v1（已废弃倾向） | **v2（本方案）** |
|----|------------------|------------------|
| 表名 | `litept_occ_scenes_v1` 等自造名 | **`occ_data_{clips,frames}_<suffix>`** 对齐 raw |
| 磁盘 | `/data/gss/litept_occ/v1/<clip_id>/` 或空想的 `/data/occdata` | **默认 `/data/rawdata/occ/`**（同挂载）；或平台新挂 `/data/occdata` |
| 主键 | clip 目录名 / scene_id 字符串 | **raw 同款 `md5` + 表 suffix** |
| 相机 | 常拷进 scene | **默认指回 rawdata/camera** |

---

## 10. 与 Matrix API 对接（本侧只做「可被调用的数据源」）

> **分工**：LitePT / GSS Occ 侧负责 **Mongo 文档契约稳定、可查询**（文档内用 `uri` 指向大文件）；  
> Matrix 侧（专人）负责 **HTTP API、鉴权（JWT/PAT/RBAC）、前端可视化页**。  
> 本节不要求本仓库实现 Matrix Controller；只规定「Matrix 应读什么、如何关联 Claymore」。

### 10.0 先澄清：「交给 Matrix 的」是 Mongo，不是让对方去扫盘

和 GSS / 原始数据完全同一套分工：

| 放哪 | 放什么 | Matrix 怎么用 |
|------|--------|----------------|
| **MongoDB**（`occ_data_*`） | Clip/Frame **元数据**：bag、时间、md5、标定、taxonomy、grid、**资产 uri 列表** | **只查这个**——对接面、query、列表、详情都来自 Mongo |
| **已挂载数据盘**（默认 `/data/rawdata/occ/...`） | Occ bin 等大文件；uri 写在 Mongo 里 | Matrix **只查 Mongo**；读文件由其后端按 uri 访问同一挂载 |

因此：

- 「交给 Matrix 的数据源」= **`perception_experiment.occ_data_clips_*` / `occ_data_frames_*` 的 Document Schema + 样例 query**。  
- 「盘」不是第二套交付物，而是 **Document 里 `assets.*.uri` 的物理落点**（和 raw 帧里 `lidar_merge.md5` → `/data/rawdata/lidar/...` 一样）。  
- **不要**把整份 Occ 体素数组塞进 Mongo Document（体积太大）；GSS 明确是元数据与大数据分离。若 Matrix 暂时只能读 Mongo、不能挂载数据盘，由 **Matrix 后端**在内网按 uri 读盘后通过 `/api/occ/.../assets` 转给前端——对 Matrix **调用方**仍只见 HTTP，不见盘路径。

```text
Matrix 开发 / CLI 用户
    │  只认：Mongo 查询结果（含 uri 字段）
    ▼
occ_data_frames_<suffix>  Document
    │  assets.occupancy.ijk.uri = "/data/rawdata/occ/lidar/a/b/c/occ_ijk.i32.bin"
    ▼
Matrix 服务端（专人）内网读文件或签 URL
    ▼
浏览器可视化
```

参考 Matrix 用户手册：Base `https://matrix-api.internal.robotruck.jp/api`，Claymore 管 Bag/Clip/Clipset/Tag，RoadSense / Scene Hunter / Mementos 等为其它域。

### 10.1 定位：Occ 不是新 Claymore 资产类型，而是挂在 Clip/Bag 上的「感知产物数据源」

| 层 | 谁拥有 | 作用 |
|----|--------|------|
| Claymore Bag / Clip | Matrix | 时间轴、车辆、下载、预览、Clipset/Tag 组织 |
| `raw_data_*` + `/data/rawdata` | GSS | 原始传感器索引与 blob |
| **`occ_data_*` + 挂载上的 occ 目录** | **本数据源** | LitePT Occ / 语义点云产物 |
| Matrix Occ 可视化页 / API | Matrix 专人 | 鉴权后读本数据源并渲染 |

用户在 Matrix 上的心智：

```text
在 Claymore 打开某个 Clip
  → Matrix 用 bagName + 纳秒起止时间（或 frame md5 列表）
  → 查询 occ_data_frames_<suffix>
  → 拉 occ bin + 原始 camera jpg
  → 三维 Occ / 多相机投影（可复用 tools/occ_viewer 契约）
```

### 10.2 推荐关联键（Claymore ↔ Mongo Occ）

Matrix Clip 已有：`bagName`、`startTimestamp` / `endTimestamp`（**纳秒字符串**）、`vehicleName`。

Occ frame 文档必须带齐下列字段，便于 Matrix **无需猜路径**：

| 字段 | 用途 |
|------|------|
| `bag_name` | = Claymore `bagName` / raw `bag_name` |
| `timestamp` | 帧时间（与 raw 一致；Matrix 用纳秒窗过滤） |
| `md5` | = raw frame / lidar_merge md5；读 `/data/rawdata/occ/lidar/a/b/c/` |
| `clip_md5` | 所属 raw clip.md5；读 static_agg |
| `source.collection` | 如 `raw_data_frames_lidar14_0731` → 推得 suffix |
| `matrix`（可选冗余） | `{ bagName, claymoreClipId?, clipsetIds? }` 回写后填充 |

**查询范式（给 Matrix 后端）：**

```text
suffix  ← 配置或由 clip/tag 映射（如 lidar14_0731）
frames  ← occ_data_frames_<suffix>.find({
            bag_name: <Claymore bagName>,
            timestamp: { $gte: startNs, $lte: endNs }
          })
blob    ← /data/rawdata/occ/lidar/{md5[0:2]}/{md5[2:4]}/{md5[4:]}/...
camera  ← /data/rawdata/camera/...（文档内 uri）
```

若 Claymore Clip 创建时已带上 **frame md5 列表**（推荐 Scene Hunter / 导出流水线写入 description 或扩展字段），则 Matrix 可直接：

```text
occ_data_frames_<suffix>.find({ md5: { $in: [...] } })
```

更稳、免时间戳边界歧义。

### 10.3 Matrix 侧建议暴露的 API 形态（由专人开发，本侧不实现）

放在 Matrix 业务域下即可（命名示意，以他们 Swagger 为准），例如：

| Method | Path（示意） | 行为 |
|--------|----------------|------|
| GET | `/api/occ/sets` 或挂在 Claymore | 列出可用 `suffix` / model / tag |
| GET | `/api/occ/clips?bagName=&suffix=` | 代理查 `occ_data_clips_*` |
| GET | `/api/occ/frames?bagName=&start=&end=` 或 `?md5=` | 代理查 `occ_data_frames_*` |
| GET | `/api/occ/frames/{md5}/assets/{name}` | **鉴权后**读盘或返回短时签名 URL |
| GET | `/api/occ/frames/{md5}/scene-index` | 返回 viewer 可用的 index/meta JSON（已把 uri 换成可拉取 URL） |

认证：与其它域相同，`Authorization: Bearer <MATRIX_PAT|JWT>` + Claymore RBAC（能读该 Bag/Clip 才能读对应 Occ）。

**本侧交付物**：Mongo 文档 + 磁盘文件 + 本文契约；**不**实现上述 HTTP。

### 10.4 可视化怎么在 Matrix 上出现（产品路径）

GSS 已描述「用户自定义脚本的真值可视化」。Occ 建议走同一模式：

```text
┌─────────────┐     JWT/PAT      ┌──────────────────┐
│ Matrix Web  │ ───────────────► │ Matrix API       │
│ Occ 面板    │                  │ /api/occ/*       │
└─────────────┘                  └────────┬─────────┘
                                          │ PyMongo / 内网读盘
                                          ▼
                                 occ_data_* + /data/rawdata/occ
                                          │
                                          ▼
                                 渲染：复用 SCHEMA.md
                                 （Occ 体素 + 相机投影）
```

可选落地方式（Matrix 选一，本侧都兼容）：

1. **嵌入式 Viewer**：Matrix 前端 iframe / 微前端加载与 `tools/occ_viewer` 同契约的静态页，`sceneBase` 指到 Matrix 签发的 asset URL。  
2. **服务端出图**：Matrix 调内部 worker，按帧渲 BEV/投影图，走类似 RoadSense `/media` 的 preview。  
3. **自定义 GT 脚本**：平台注册「LitePT Occ」可视化脚本，入参为 `md5`/`bagName`+时间窗，读本数据源。

无论哪种，**唯一真相来源仍是 `occ_data_*` + 文档 uri 指向的挂载路径**，与 Claymore 只做键关联。

### 10.5 与 Matrix 其它域的关系（避免用错入口）

| 需求 | 用 Matrix 哪块 | 是否读 Occ Mongo |
|------|----------------|------------------|
| 找 Bag、切 Clip、Clipset、Tag | **Claymore** `/api/bags|clips|clipsets|tags` | 否（先组织资产） |
| 看 Occ / 语义点云 / 投影 | **新建 Occ API + 可视化**（或 GT 脚本） | **是** |
| 触发事件 / 热力 | RoadSense | 否 |
| 文本搜场景再沉淀 Clip | Scene Hunter → 再链 Occ | 搜完后可用 bag+时间查 Occ |
| 闭环仿真 | Mementos | 否（除非评测指标引用 Occ） |

自动化脚本典型顺序：

1. `GET /auth/me`、`GET /rbac/me`  
2. Claymore 查 Bag / 创建或选取 Clip  
3. （Matrix 上线后）`GET /api/occ/frames?...` 拉 Occ meta + 签名资源  
4. 本地或 Matrix UI 可视化  

### 10.6 本侧「数据源就绪」验收清单（给 Matrix 开发的对接包）

**交给 Matrix 的只有 Mongo 侧契约**（他们用 PyMongo / 或自己的 Occ API 读这些 Document）：

- [ ] DB：`perception_experiment`  
- [ ] Collections：`occ_data_frames_<suffix>` / `occ_data_clips_<suffix>` 可查  
- [ ] Document 含：`bag_name`、`timestamp`、`md5`、`grid`、`taxonomy`、`assets.*.uri`（uri 字符串即可）  
- [ ] 样例：对某个 `bag_name` 的 `find` 结果 JSON（可脱敏）  
- [ ] 说明：各 batch 的 `suffix` / `model` / `tag`  

**盘上的文件**：由你们（或平台存储）按 Document 里的 uri 写好并保证 Matrix **服务端内网**可读；**不作为「另一份目录说明书」交给 Matrix 业务开发去对接**。若他们只能调 HTTP，由 Matrix 后端按 uri 读盘并封装。

**不在本侧范围**：Matrix Controller、PAT/RBAC、前端 Occ 页、对公网暴露盘路径。

### 10.7 一句话协议

> **Claymore 管「这段车在哪个 Bag/Clip」；Occ Mongo 管「这段车上 LitePT Occ 文件在哪」；Matrix API 用 Bag/时间或 md5 把两者接起来做可视化。**
