import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const qs = new URLSearchParams(location.search);
/** Active scene HTTP prefix, e.g. /scenes/<clip_id> or /scene */
let sceneRoot = (qs.get("scene") || "").replace(/\/$/, "");
let clipsCatalog = [];
let frameIndex = 0;
let currentOccVariant = "litept";

/** Resolve asset URI relative to scene root (Mongo/S3 can swap to absolute later). */
function assetUrl(uri, frameRelFallback = null) {
  if (!uri) return null;
  if (/^(https?:|data:|blob:|gridfs:|s3:)/i.test(uri)) return uri;
  if (uri.startsWith("/")) return uri;
  // scene-root relative (v1)
  if (uri.startsWith("frames/")) return `${sceneRoot}/${uri}`;
  // legacy: relative to frame dir
  if (frameRelFallback) return `${sceneRoot}/${frameRelFallback}/${uri}`;
  return `${sceneRoot}/${uri}`;
}

function occAsset(meta, key) {
  const variants = meta.assets && meta.assets.occupancy_variants;
  const group = (variants && variants[currentOccVariant]) || (meta.assets && meta.assets.occupancy);
  const a = group && group[key];
  if (a && a.uri) return assetUrl(a.uri);
  if (meta.occupancy && meta.occupancy[key]) return assetUrl(meta.occupancy[key], frameDir);
  return null;
}

function pointsAsset(meta, key) {
  const a = meta.assets && meta.assets.points && meta.assets.points[key];
  if (a && a.uri) return assetUrl(a.uri);
  if (meta.points && meta.points[key]) return assetUrl(meta.points[key], frameDir);
  return null;
}

function frameSensorPointsAsset(meta, key) {
  const a = meta.assets && meta.assets.frame_sensor_points && meta.assets.frame_sensor_points[key];
  return a && a.uri ? assetUrl(a.uri, frameDir) : null;
}

function camImageUrl(cam) {
  if (cam.image && cam.image.uri) return assetUrl(cam.image.uri);
  if (cam.file) return assetUrl(cam.file, frameDir);
  return null;
}

const el = {
  frameSelect: document.getElementById("frameSelect"),
  clipSelect: document.getElementById("clipSelect"),
  occVariant: document.getElementById("occVariant"),
  btnPrevFrame: document.getElementById("btnPrevFrame"),
  btnNextFrame: document.getElementById("btnNextFrame"),
  framePos: document.getElementById("framePos"),
  titleMeta: document.getElementById("titleMeta"),
  sceneInfo: document.getElementById("sceneInfo"),
  status: document.getElementById("status"),
  cams: document.getElementById("cams"),
  togOcc: document.getElementById("togOcc"),
  togPts: document.getElementById("togPts"),
  togOdBoxes: document.getElementById("togOdBoxes"),
  togGrid: document.getElementById("togGrid"),
  togAxes: document.getElementById("togAxes"),
  occOpacity: document.getElementById("occOpacity"),
  occGap: document.getElementById("occGap"),
  occGrow: document.getElementById("occGrow"),
  ptSize: document.getElementById("ptSize"),
  voxelSize: document.getElementById("voxelSize"),
  btnRebuildOcc: document.getElementById("btnRebuildOcc"),
  btnResetOcc: document.getElementById("btnResetOcc"),
  occRebuildHint: document.getElementById("occRebuildHint"),
  projMode: document.getElementById("projMode"),
  projRadius: document.getElementById("projRadius"),
  projRadiusValue: document.getElementById("projRadiusValue"),
  projAlpha: document.getElementById("projAlpha"),
  projAlphaValue: document.getElementById("projAlphaValue"),
  projColorMode: document.getElementById("projColorMode"),
  projClassFilter: document.getElementById("projClassFilter"),
  projHeightMin: document.getElementById("projHeightMin"),
  projHeightMax: document.getElementById("projHeightMax"),
  projPointSource: document.getElementById("projPointSource"),
  projDistortion: document.getElementById("projDistortion"),
  btnRefreshProj: document.getElementById("btnRefreshProj"),
  btnFit: document.getElementById("btnFit"),
  classLegend: document.getElementById("classLegend"),
  colorMode: document.getElementById("colorMode"),
  coarseToggles: document.getElementById("coarseToggles"),
  lidarToggles: document.getElementById("lidarToggles"),
  togDyn: document.getElementById("togDyn"),
  togSta: document.getElementById("togSta"),
  togFree: document.getElementById("togFree"),
  togNoise: document.getElementById("togNoise"),
  togLid1: document.getElementById("togLid1"),
  togLid2: document.getElementById("togLid2"),
  togLid14: document.getElementById("togLid14"),
  roiX0: document.getElementById("roiX0"),
  roiX1: document.getElementById("roiX1"),
  roiY0: document.getElementById("roiY0"),
  roiY1: document.getElementById("roiY1"),
  roiZ0: document.getElementById("roiZ0"),
  roiZ1: document.getElementById("roiZ1"),
  btnApplyRoi: document.getElementById("btnApplyRoi"),
  btnResetRoi: document.getElementById("btnResetRoi"),
  togRoiBox: document.getElementById("togRoiBox"),
  togRoiClip: document.getElementById("togRoiClip"),
  vidMode: document.getElementById("vidMode"),
  vidFps: document.getElementById("vidFps"),
  vidMaxFrames: document.getElementById("vidMaxFrames"),
  btnExportVid: document.getElementById("btnExportVid"),
  btnRefreshVid: document.getElementById("btnRefreshVid"),
  vidStatus: document.getElementById("vidStatus"),
  vidList: document.getElementById("vidList"),
  wrap: document.getElementById("canvas-wrap"),
  lightbox: document.getElementById("lightbox"),
  lbStage: document.getElementById("lb-stage"),
  lbCanvas: document.getElementById("lb-canvas"),
  lbTitle: document.getElementById("lb-title"),
  lbZoomIn: document.getElementById("lb-zoom-in"),
  lbZoomOut: document.getElementById("lb-zoom-out"),
  lbReset: document.getElementById("lb-reset"),
  lbClose: document.getElementById("lb-close"),
};

let index = null;
let currentMeta = null;
let frameDir = null;
let occMesh = null;
let pointsObj = null;
let odBoxesGroup = null;
let gridHelper = null;
let axesGroup = null;
let classColors = null;
let classNames = null;

let occIjk = null; // Int32 ix,iy,iz
let occLabels = null;
let occCenters = null; // for projection
let occProjLabels = null; // filtered labels parallel to occCenters
let occProjIjk = null; // filtered ijk parallel to occCenters (for neat quads)
let activeVoxel = null;
let exportedOcc = null; // { voxel, ijk: Int32Array, labels: Uint8Array }
let ptXYZ = null;
let ptLabels = null;
let ptLidar = null; // Uint8Array lidar_id per point
let framePtXYZ = null; // current-frame deskew points, before static aggregation
let framePtLabels = null;
let framePtLidar = null;
/** Clip-level static in map frame (same points every frame; only pose changes). */
let staticAgg = null; // { xyz: Float32Array, labels: Uint8Array, lidar: Uint8Array, n, voxel }
let roiHelper = null;
let roi = { x0: -24, x1: 24, y0: -25, y1: 150, z0: -5, z1: 3 };

/** Coarse 4-class mapping (Waymo 22 → dynamic/static/freespace/noise) */
const COARSE_DYNAMIC = new Set([0, 1, 2, 3, 4, 5, 6, 11, 12]);
const FINE_DYNAMIC = COARSE_DYNAMIC;
const COARSE_FREESPACE = new Set([17, 18, 19, 20, 21]);
const COARSE_STATIC = new Set([7, 8, 9, 10, 13, 14, 15, 16]);
const COARSE_COLORS = {
  dynamic: [230, 64, 64],
  static: [64, 160, 255],
  freespace: [72, 200, 96],
  noise: [160, 160, 160],
};
const COARSE_ORDER = ["dynamic", "static", "freespace", "noise"];
/** RGB matching render_robotruck_clip_video LIDAR_ID_BGR */
const LIDAR_COLORS = {
  1: [40, 180, 255],
  2: [40, 220, 40],
  14: [255, 40, 40],
};

/** Lightbox state */
let lb = { scale: 1, tx: 0, ty: 0, dragging: false, lx: 0, ly: 0, source: null };

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x0b0e13, 1);
el.wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 3000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const sun = new THREE.DirectionalLight(0xffffff, 0.85);
sun.position.set(40, 120, -30);
scene.add(sun);

/**
 * Vehicle → Three.js display frame.
 * Vehicle: +x right, +y forward, +z up.
 * Three: Y-up, ground on XZ. Remap (x,y,z)_veh → (-x, z, y)_three so that
 * when the default camera sits behind ego looking along +Z (= forward),
 * vehicle +x lands on screen-right — matching camera images.
 * (A plain (x,z,y) map flips L/R: Three lookAt(+Z) makes camera-right = −X.)
 * Cam projection still uses raw vehicle xyz (unchanged).
 */
function vehToThree(x, y, z, out = new THREE.Vector3()) {
  return out.set(-x, z, y);
}

gridHelper = new THREE.GridHelper(400, 80, 0x445066, 0x243041);
scene.add(gridHelper);

function makeAxisArrow(dir, color, length) {
  return new THREE.ArrowHelper(
    dir.clone().normalize(),
    new THREE.Vector3(0, 0.02, 0),
    length,
    color,
    2.0,
    1.2
  );
}

function makeSprite(text, pos, color) {
  const canvas = document.createElement("canvas");
  canvas.width = 320;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = color;
  ctx.font = "bold 30px sans-serif";
  ctx.fillText(text, 6, 42);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const spr = new THREE.Sprite(mat);
  spr.position.copy(pos);
  spr.scale.set(10, 2, 1);
  return spr;
}

function clearOdBoxes() {
  if (!odBoxesGroup) return;
  scene.remove(odBoxesGroup);
  odBoxesGroup.traverse((obj) => {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      if (obj.material.map) obj.material.map.dispose();
      obj.material.dispose();
    }
  });
  odBoxesGroup = null;
}

function buildOdBoxes(meta) {
  clearOdBoxes();
  odBoxesGroup = new THREE.Group();
  const objects = (meta.od_boxes && meta.od_boxes.objects) || [];
  for (const obj of objects) {
    const c = obj.center_imu || [];
    const s = obj.size || [];
    if (c.length < 3 || s.length < 3) continue;
    const [length, width, height] = s.map(Number);
    const yaw = Number(obj.orientation_imu || 0);
    const ux = [Math.cos(yaw) * length / 2, Math.sin(yaw) * length / 2];
    const uy = [-Math.sin(yaw) * width / 2, Math.cos(yaw) * width / 2];
    const corners = [];
    for (const zSign of [-1, 1]) {
      for (const xSign of [-1, 1]) {
        for (const ySign of [-1, 1]) {
          corners.push(vehToThree(
            Number(c[0]) + xSign * ux[0] + ySign * uy[0],
            Number(c[1]) + xSign * ux[1] + ySign * uy[1],
            Number(c[2]) + zSign * height / 2
          ));
        }
      }
    }
    const edges = [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]];
    const positions = [];
    for (const [a, b] of edges) positions.push(...corners[a].toArray(), ...corners[b].toArray());
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const matched = obj.display_id !== null && obj.display_id !== undefined;
    const color = matched ? 0xffcc33 : 0xff5c7a;
    odBoxesGroup.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color })));
    const id = matched ? obj.display_id : "unmatched";
    const label = makeSprite(`OD#${obj.od_index} → ${id}`, vehToThree(Number(c[0]), Number(c[1]), Number(c[2]) + height / 2 + 0.8), matched ? "#ffdd55" : "#ff6b86");
    label.scale.set(12, 2.4, 1);
    odBoxesGroup.add(label);
  }
  odBoxesGroup.visible = el.togOdBoxes.checked;
  scene.add(odBoxesGroup);
  return objects.length;
}

function buildAxes() {
  if (axesGroup) scene.remove(axesGroup);
  axesGroup = new THREE.Group();
  // Directions in Three space after vehToThree (vehicle +x → Three −X).
  axesGroup.add(makeAxisArrow(new THREE.Vector3(-1, 0, 0), 0xff4d4d, 20));
  axesGroup.add(makeAxisArrow(new THREE.Vector3(0, 0, 1), 0x3dde6a, 20));
  axesGroup.add(makeAxisArrow(new THREE.Vector3(0, 1, 0), 0x4da3ff, 20));
  axesGroup.add(makeSprite("+X", vehToThree(22, 0, 0.8), "#ff4d4d"));
  axesGroup.add(makeSprite("+Y", vehToThree(0, 22, 0.8), "#3dde6a"));
  axesGroup.add(makeSprite("+Z", vehToThree(0, 0, 22), "#4da3ff"));
  axesGroup.visible = el.togAxes.checked;
  scene.add(axesGroup);
}
buildAxes();

function resize() {
  const w = el.wrap.clientWidth;
  const h = el.wrap.clientHeight;
  camera.aspect = w / Math.max(1, h);
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}
window.addEventListener("resize", resize);
resize();

function setStatus(msg) {
  el.status.textContent = msg || "";
}

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
  return r.json();
}

async function fetchBin(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
  return r.arrayBuffer();
}

function quatToRotMat(q) {
  let x = +q.x,
    y = +q.y,
    z = +q.z,
    w = +q.w;
  const n = Math.hypot(x, y, z, w) || 1;
  x /= n;
  y /= n;
  z /= n;
  w /= n;
  return [
    1 - 2 * (y * y + z * z),
    2 * (x * y - z * w),
    2 * (x * z + y * w),
    2 * (x * y + z * w),
    1 - 2 * (x * x + z * z),
    2 * (y * z - x * w),
    2 * (x * z - y * w),
    2 * (y * z + x * w),
    1 - 2 * (x * x + y * y),
  ];
}

/** Row-major 4x4 map←vehicle from ego_pose.pose */
function egoPoseToTMapVehicle(pose) {
  const R = quatToRotMat(pose.orientation);
  const p = pose.position;
  return new Float64Array([
    R[0],
    R[1],
    R[2],
    +p.x,
    R[3],
    R[4],
    R[5],
    +p.y,
    R[6],
    R[7],
    R[8],
    +p.z,
    0,
    0,
    0,
    1,
  ]);
}

function invertRigid4(T) {
  // T = [R|t]; inv = [R^T | -R^T t]
  const R = [
    T[0],
    T[1],
    T[2],
    T[4],
    T[5],
    T[6],
    T[8],
    T[9],
    T[10],
  ];
  const tx = T[3],
    ty = T[7],
    tz = T[11];
  const ix = -(R[0] * tx + R[3] * ty + R[6] * tz);
  const iy = -(R[1] * tx + R[4] * ty + R[7] * tz);
  const iz = -(R[2] * tx + R[5] * ty + R[8] * tz);
  return new Float64Array([
    R[0],
    R[3],
    R[6],
    ix,
    R[1],
    R[4],
    R[7],
    iy,
    R[2],
    R[5],
    R[8],
    iz,
    0,
    0,
    0,
    1,
  ]);
}

function transformPoints(xyz, T) {
  const n = xyz.length / 3;
  const out = new Float32Array(xyz.length);
  for (let i = 0; i < n; i++) {
    const o = i * 3;
    const x = xyz[o],
      y = xyz[o + 1],
      z = xyz[o + 2];
    out[o] = T[0] * x + T[1] * y + T[2] * z + T[3];
    out[o + 1] = T[4] * x + T[5] * y + T[6] * z + T[7];
    out[o + 2] = T[8] * x + T[9] * y + T[10] * z + T[11];
  }
  return out;
}

/**
 * Map-frame static_agg → current vehicle frame (optional crop).
 * Same points every frame; only the rigid transform changes.
 */
function staticAggInVehicle(pose, xRange, yRange, zRange) {
  if (!staticAgg || !pose) return null;
  const T_map_v = egoPoseToTMapVehicle(pose);
  const T_v_map = invertRigid4(T_map_v);
  const xyz = transformPoints(staticAgg.xyz, T_v_map);
  const n = staticAgg.n;
  const keep = [];
  for (let i = 0; i < n; i++) {
    const o = i * 3;
    const x = xyz[o],
      y = xyz[o + 1],
      z = xyz[o + 2];
    if (xRange && (x < xRange[0] || x > xRange[1])) continue;
    if (yRange && (y < yRange[0] || y > yRange[1])) continue;
    if (zRange && (z < zRange[0] || z > zRange[1])) continue;
    keep.push(i);
  }
  const m = keep.length;
  const outX = new Float32Array(m * 3);
  const outL = new Uint8Array(m);
  const outI = new Uint8Array(m);
  for (let k = 0; k < m; k++) {
    const i = keep[k];
    outX[k * 3] = xyz[i * 3];
    outX[k * 3 + 1] = xyz[i * 3 + 1];
    outX[k * 3 + 2] = xyz[i * 3 + 2];
    outL[k] = staticAgg.labels[i];
    outI[k] = staticAgg.lidar ? staticAgg.lidar[i] : 0;
  }
  return { xyz: outX, labels: outL, lidar: outI, n: m };
}

function mergeStaticAndDynamicPoints(staticPart, dynXYZ, dynLab, dynLid) {
  const ns = staticPart ? staticPart.n : 0;
  const nd = dynLab ? dynLab.length : 0;
  const n = ns + nd;
  const xyz = new Float32Array(n * 3);
  const lab = new Uint8Array(n);
  const lid = new Uint8Array(n);
  if (ns) {
    xyz.set(staticPart.xyz);
    lab.set(staticPart.labels);
    lid.set(staticPart.lidar);
  }
  for (let i = 0; i < nd; i++) {
    const o = (ns + i) * 3;
    const s = i * 3;
    xyz[o] = dynXYZ[s];
    xyz[o + 1] = dynXYZ[s + 1];
    xyz[o + 2] = dynXYZ[s + 2];
    lab[ns + i] = dynLab[i];
    lid[ns + i] = dynLid ? dynLid[i] : 0;
  }
  return { xyz, labels: lab, lidar: lid };
}

async function loadStaticAggFromIndex() {
  staticAgg = null;
  const pointAgg = index && index.point_aggregate;
  const staticVariants = index && index.static_agg_variants;
  const sa = pointAgg || (staticVariants && staticVariants[currentOccVariant]) ||
    (index && index.static_agg);
  if (!sa || !sa.xyz_map || !sa.labels) return;
  const xyzBuf = await fetchBin(assetUrl(sa.xyz_map.uri));
  const labBuf = await fetchBin(assetUrl(sa.labels.uri));
  let lid = null;
  if (sa.lidar_id && sa.lidar_id.uri) {
    lid = new Uint8Array(await fetchBin(assetUrl(sa.lidar_id.uri)));
  }
  staticAgg = {
    xyz: new Float32Array(xyzBuf),
    labels: new Uint8Array(labBuf),
    lidar: lid,
    n: sa.n || new Uint8Array(labBuf).length,
    voxel: sa.voxel || 0.25,
    full: Boolean(pointAgg),
  };
  setStatus(`${staticAgg.full ? "full point aggregate" : "static_agg"} loaded · n=${staticAgg.n.toLocaleString()}`);
}

function fineToCoarse(lab) {
  const i = lab | 0;
  if (i < 0 || i >= 22) return "noise";
  if (COARSE_DYNAMIC.has(i)) return "dynamic";
  if (COARSE_FREESPACE.has(i)) return "freespace";
  if (COARSE_STATIC.has(i)) return "static";
  return "noise";
}

function colorModeValue() {
  return el.colorMode ? el.colorMode.value : "fine";
}

function readRoiFromInputs() {
  const a = (lo, hi) => {
    let x0 = Number(lo.value);
    let x1 = Number(hi.value);
    if (!(x0 < x1)) {
      const t = x0;
      x0 = x1;
      x1 = t;
    }
    return [x0, x1];
  };
  const [x0, x1] = a(el.roiX0, el.roiX1);
  const [y0, y1] = a(el.roiY0, el.roiY1);
  const [z0, z1] = a(el.roiZ0, el.roiZ1);
  roi = { x0, x1, y0, y1, z0, z1 };
  return roi;
}

function inRoi(x, y, z) {
  if (!el.togRoiClip || !el.togRoiClip.checked) return true;
  return x >= roi.x0 && x <= roi.x1 && y >= roi.y0 && y <= roi.y1 && z >= roi.z0 && z <= roi.z1;
}

function coarseVisible(name) {
  if (name === "dynamic") return !el.togDyn || el.togDyn.checked;
  if (name === "static") return !el.togSta || el.togSta.checked;
  if (name === "freespace") return !el.togFree || el.togFree.checked;
  if (name === "noise") return !el.togNoise || el.togNoise.checked;
  return true;
}

function lidarVisible(lid) {
  if (lid === 1) return !el.togLid1 || el.togLid1.checked;
  if (lid === 2) return !el.togLid2 || el.togLid2.checked;
  if (lid === 14) return !el.togLid14 || el.togLid14.checked;
  return true;
}

function normalizeClassColors(raw) {
  if (!raw || !raw.length) return null;
  const out = [];
  let maxv = 0;
  for (let i = 0; i < raw.length; i++) {
    const c = raw[i];
    if (!c || c.length < 3) {
      out.push([180, 180, 180]);
      continue;
    }
    const r = Number(c[0]),
      g = Number(c[1]),
      b = Number(c[2]);
    out.push([r, g, b]);
    maxv = Math.max(maxv, r, g, b);
  }
  // Some exports store 0..1 floats; CSS/Three need 0..255.
  if (maxv > 0 && maxv <= 1.5) {
    for (let i = 0; i < out.length; i++) {
      out[i] = [
        Math.round(out[i][0] * 255),
        Math.round(out[i][1] * 255),
        Math.round(out[i][2] * 255),
      ];
    }
  } else {
    for (let i = 0; i < out.length; i++) {
      out[i] = [
        Math.round(out[i][0]),
        Math.round(out[i][1]),
        Math.round(out[i][2]),
      ];
    }
  }
  return out;
}

function resolveClassColors(meta) {
  // Prefer scene taxonomy (stable), then per-frame meta.
  const fromTax =
    index && index.taxonomy && index.taxonomy.fine && index.taxonomy.fine.colors_rgb;
  const fromMeta = meta && meta.class_colors_rgb;
  return normalizeClassColors(fromTax || fromMeta);
}

function colorFromLabel(label) {
  const mode = colorModeValue();
  if (mode === "coarse") {
    const c = COARSE_COLORS[fineToCoarse(label)] || COARSE_COLORS.noise;
    return new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255);
  }
  const i = label | 0;
  if (!classColors || i < 0 || i >= classColors.length || !classColors[i]) {
    return new THREE.Color(0.7, 0.7, 0.7);
  }
  const c = classColors[i];
  return new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255);
}

function colorFromLidar(lid) {
  const c = LIDAR_COLORS[lid] || [160, 160, 160];
  return new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255);
}

function rgbCss(label) {
  const mode = colorModeValue();
  if (mode === "coarse") {
    const c = COARSE_COLORS[fineToCoarse(label)] || COARSE_COLORS.noise;
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }
  const i = label | 0;
  if (!classColors || i < 0 || i >= classColors.length || !classColors[i]) {
    return "rgb(180,180,180)";
  }
  const c = classColors[i];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function rgbCssLidar(lid) {
  const c = LIDAR_COLORS[lid] || [160, 160, 160];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function syncColorModeUi() {
  const mode = colorModeValue();
  if (el.coarseToggles) el.coarseToggles.style.display = mode === "coarse" ? "block" : "none";
  if (el.lidarToggles) el.lidarToggles.style.display = mode === "lidar" ? "block" : "none";
  renderClassLegend();
}

function renderClassLegend() {
  if (!el.classLegend) return;
  const mode = colorModeValue();
  if (mode === "coarse") {
    el.classLegend.innerHTML = COARSE_ORDER.map((name) => {
      const c = COARSE_COLORS[name];
      return `<div class="legend-item" title="${name}">
        <i class="legend-swatch" style="background:rgb(${c[0]},${c[1]},${c[2]})"></i>
        <span>${name}</span>
      </div>`;
    }).join("");
    return;
  }
  if (mode === "lidar") {
    el.classLegend.innerHTML = [1, 2, 14]
      .map((lid) => {
        const c = LIDAR_COLORS[lid];
        return `<div class="legend-item" title="lidar_${lid}">
          <i class="legend-swatch" style="background:rgb(${c[0]},${c[1]},${c[2]})"></i>
          <span>lidar_${lid}</span>
        </div>`;
      })
      .join("");
    return;
  }
  if (!classColors || !classColors.length) {
    el.classLegend.innerHTML = `<span class="hint">No class colors in meta</span>`;
    return;
  }
  const names = classNames && classNames.length ? classNames : classColors.map((_, i) => `class ${i}`);
  el.classLegend.innerHTML = names
    .map((name, i) => {
      const c = classColors[i] || [180, 180, 180];
      return `<div class="legend-item" title="${i}: ${name}">
        <i class="legend-swatch" style="background:rgb(${c[0]},${c[1]},${c[2]})"></i>
        <span>${i} ${name}</span>
      </div>`;
    })
    .join("");
}

function updateRoiHelper() {
  if (roiHelper) {
    scene.remove(roiHelper);
    roiHelper.geometry.dispose();
    roiHelper.material.dispose();
    roiHelper = null;
  }
  if (!el.togRoiBox || !el.togRoiBox.checked) return;
  // Box in vehicle frame → Three via vehToThree (center); size along remapped axes.
  const cx = 0.5 * (roi.x0 + roi.x1);
  const cy = 0.5 * (roi.y0 + roi.y1);
  const cz = 0.5 * (roi.z0 + roi.z1);
  const sx = Math.max(1e-3, roi.x1 - roi.x0);
  const sy = Math.max(1e-3, roi.y1 - roi.y0);
  const sz = Math.max(1e-3, roi.z1 - roi.z0);
  const geo = new THREE.BoxGeometry(sx, sz, sy);
  const edges = new THREE.EdgesGeometry(geo);
  geo.dispose();
  const mat = new THREE.LineBasicMaterial({ color: 0xffcc33 });
  roiHelper = new THREE.LineSegments(edges, mat);
  vehToThree(cx, cy, cz, roiHelper.position);
  scene.add(roiHelper);
}

function clearOcc() {
  if (!occMesh) return;
  scene.remove(occMesh);
  occMesh.geometry.dispose();
  occMesh.material.dispose();
  occMesh = null;
}

function clearPoints() {
  if (!pointsObj) return;
  scene.remove(pointsObj);
  pointsObj.geometry.dispose();
  pointsObj.material.dispose();
  pointsObj = null;
}

/**
 * Place cubes on the occupancy lattice:
 *   center = origin + (ijk + 0.5) * voxel
 * not on raw point positions.
 */
function buildOccMesh() {
  clearOcc();
  if (!occIjk || !occLabels || !currentMeta || !activeVoxel) return;
  const nAll = occLabels.length;
  const v = activeVoxel;
  const x0 = currentMeta.x_range[0];
  const y0 = currentMeta.y_range[0];
  const z0 = currentMeta.z_range[0];
  const gap = Math.max(0, Number(el.occGap.value));
  const grow = Number(el.occGrow.value);
  const size = Math.max(1e-4, v * (1.0 - gap) * grow);
  const mode = colorModeValue();

  const keepIdx = [];
  const centersAll = new Float32Array(nAll * 3);
  for (let i = 0; i < nAll; i++) {
    const ix = occIjk[i * 3];
    const iy = occIjk[i * 3 + 1];
    const iz = occIjk[i * 3 + 2];
    const vx = x0 + (ix + 0.5) * v;
    const vy = y0 + (iy + 0.5) * v;
    const vz = z0 + (iz + 0.5) * v;
    centersAll[i * 3] = vx;
    centersAll[i * 3 + 1] = vy;
    centersAll[i * 3 + 2] = vz;
    if (!inRoi(vx, vy, vz)) continue;
    if (mode === "coarse" && !coarseVisible(fineToCoarse(occLabels[i]))) continue;
    keepIdx.push(i);
  }

  const n = keepIdx.length;
  if (n === 0) {
    occCenters = new Float32Array(0);
    occProjLabels = null;
    occProjIjk = null;
    return;
  }

  const geo = new THREE.BoxGeometry(size, size, size);
  let op = Number(el.occOpacity.value);
  if (pointsObj && el.togPts.checked && el.togOcc.checked) {
    op = Math.min(op, 0.35);
  }
  const mat = new THREE.MeshLambertMaterial({
    transparent: op < 0.999,
    opacity: op,
    depthWrite: true,
  });
  const mesh = new THREE.InstancedMesh(geo, mat, n);
  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  const centers = new Float32Array(n * 3);
  const labs = new Uint8Array(n);
  const ijks = new Int32Array(n * 3);

  for (let k = 0; k < n; k++) {
    const i = keepIdx[k];
    const vx = centersAll[i * 3];
    const vy = centersAll[i * 3 + 1];
    const vz = centersAll[i * 3 + 2];
    centers[k * 3] = vx;
    centers[k * 3 + 1] = vy;
    centers[k * 3 + 2] = vz;
    labs[k] = occLabels[i];
    ijks[k * 3] = occIjk[i * 3];
    ijks[k * 3 + 1] = occIjk[i * 3 + 1];
    ijks[k * 3 + 2] = occIjk[i * 3 + 2];
    vehToThree(vx, vy, vz, dummy.position);
    dummy.updateMatrix();
    mesh.setMatrixAt(k, dummy.matrix);
    // lidar mode: occ still uses fine semantic (or coarse if switched)
    if (mode === "lidar") {
      // force fine coloring for occ under lidar mode
      const iLab = occLabels[i] | 0;
      if (classColors && iLab >= 0 && iLab < classColors.length && classColors[iLab]) {
        const c = classColors[iLab];
        color.setRGB(c[0] / 255, c[1] / 255, c[2] / 255);
      } else {
        color.setRGB(0.7, 0.7, 0.7);
      }
    } else {
      color.copy(colorFromLabel(occLabels[i]));
    }
    mesh.setColorAt(k, color);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.frustumCulled = false;
  occMesh = mesh;
  occMesh.visible = el.togOcc.checked;
  scene.add(mesh);
  occCenters = centers;
  occProjLabels = labs;
  occProjIjk = ijks;
}

function voxelizeFromPoints(voxel) {
  if (!ptXYZ || !ptLabels || !currentMeta) {
    throw new Error("Need exported points to rebuild occupancy");
  }
  const v = Math.max(0.05, Number(voxel));
  const x0 = currentMeta.x_range[0];
  const x1 = currentMeta.x_range[1];
  const y0 = currentMeta.y_range[0];
  const y1 = currentMeta.y_range[1];
  const z0 = currentMeta.z_range[0];
  const z1 = currentMeta.z_range[1];
  const nx = Math.max(1, Math.ceil((x1 - x0) / v));
  const ny = Math.max(1, Math.ceil((y1 - y0) / v));
  const nz = Math.max(1, Math.ceil((z1 - z0) / v));

  // key -> { bestLab, bestCnt, counts: Map }
  const cells = new Map();
  const n = ptLabels.length;
  for (let i = 0; i < n; i++) {
    const o = i * 3;
    const x = ptXYZ[o];
    const y = ptXYZ[o + 1];
    const z = ptXYZ[o + 2];
    if (x < x0 || x >= x1 || y < y0 || y >= y1 || z < z0 || z >= z1) continue;
    let ix = Math.floor((x - x0) / v);
    let iy = Math.floor((y - y0) / v);
    let iz = Math.floor((z - z0) / v);
    if (ix < 0) ix = 0;
    if (iy < 0) iy = 0;
    if (iz < 0) iz = 0;
    if (ix >= nx) ix = nx - 1;
    if (iy >= ny) iy = ny - 1;
    if (iz >= nz) iz = nz - 1;
    const key = ix + nx * (iy + ny * iz);
    const lab = ptLabels[i];
    let cell = cells.get(key);
    if (!cell) {
      cell = { ix, iy, iz, votes: new Map(), bestLab: lab, bestCnt: 0 };
      cells.set(key, cell);
    }
    const cnt = (cell.votes.get(lab) || 0) + 1;
    cell.votes.set(lab, cnt);
    if (cnt > cell.bestCnt) {
      cell.bestCnt = cnt;
      cell.bestLab = lab;
    }
  }

  const nOcc = cells.size;
  const ijk = new Int32Array(nOcc * 3);
  const labels = new Uint8Array(nOcc);
  let k = 0;
  for (const cell of cells.values()) {
    ijk[k * 3] = cell.ix;
    ijk[k * 3 + 1] = cell.iy;
    ijk[k * 3 + 2] = cell.iz;
    labels[k] = cell.bestLab;
    k += 1;
  }
  return { ijk, labels, voxel: v, nOcc };
}

function applyOccupancy(ijk, labels, voxel, sourceNote) {
  occIjk = ijk;
  occLabels = labels;
  activeVoxel = voxel;
  if (el.voxelSize) el.voxelSize.value = String(voxel);
  buildOccMesh();
  if (el.occRebuildHint) {
    el.occRebuildHint.textContent = `${sourceNote} · voxel=${voxel}m · n=${labels.length.toLocaleString()}`;
  }
  if (currentMeta) {
    const t = `${index.clip}  ·  ts=${currentMeta.timestamp}  ·  voxel=${voxel}m  ·  occ=${labels.length}`;
    el.titleMeta.textContent = t;
    el.titleMeta.title = t;
  }
}

function buildPoints() {
  clearPoints();
  if (!ptXYZ || !ptLabels) return;
  const nAll = ptLabels.length;
  const mode = colorModeValue();
  const keepIdx = [];
  for (let i = 0; i < nAll; i++) {
    const o = i * 3;
    const x = ptXYZ[o];
    const y = ptXYZ[o + 1];
    const z = ptXYZ[o + 2];
    if (!inRoi(x, y, z)) continue;
    if (mode === "coarse" && !coarseVisible(fineToCoarse(ptLabels[i]))) continue;
    if (mode === "lidar") {
      const lid = ptLidar ? ptLidar[i] : 0;
      if (!lidarVisible(lid)) continue;
    }
    keepIdx.push(i);
  }
  const n = keepIdx.length;
  if (n === 0) return;

  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  const projXYZ = new Float32Array(n * 3);
  const projLab = new Uint8Array(n);
  const projLid = new Uint8Array(n);
  for (let k = 0; k < n; k++) {
    const i = keepIdx[k];
    const o = i * 3;
    const pk = k * 3;
    // Display uses vehToThree; proj* stay in vehicle frame for cam projection.
    positions[pk] = -ptXYZ[o];
    positions[pk + 1] = ptXYZ[o + 2];
    positions[pk + 2] = ptXYZ[o + 1];
    projXYZ[pk] = ptXYZ[o];
    projXYZ[pk + 1] = ptXYZ[o + 1];
    projXYZ[pk + 2] = ptXYZ[o + 2];
    projLab[k] = ptLabels[i];
    projLid[k] = ptLidar ? ptLidar[i] : 0;
    let c;
    if (mode === "lidar") c = colorFromLidar(projLid[k]);
    else c = colorFromLabel(ptLabels[i]);
    colors[pk] = c.r;
    colors[pk + 1] = c.g;
    colors[pk + 2] = c.b;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.PointsMaterial({
    size: Number(el.ptSize.value),
    vertexColors: true,
    sizeAttenuation: true,
    depthWrite: false,
  });
  pointsObj = new THREE.Points(geo, mat);
  pointsObj.renderOrder = 2;
  pointsObj.visible = el.togPts.checked;
  pointsObj.userData.projXYZ = projXYZ;
  pointsObj.userData.projLab = projLab;
  pointsObj.userData.projLid = projLid;
  scene.add(pointsObj);
}

function rebuildColoredViews() {
  buildOccMesh();
  buildPoints();
  updateRoiHelper();
  refreshCamProjections();
}

function fitCamera(meta) {
  // Behind ego, elevated, look along +Y (Three +Z). Target ~40m ahead.
  const ahead = Math.min(80, Math.max(30, meta.y_range[1] * 0.15));
  controls.target.set(0, 1.2, ahead);
  camera.position.set(0, 55, -25);
  camera.up.set(0, 1, 0);
  controls.minDistance = 5;
  controls.maxDistance = 800;
  controls.update();
}

function projectOneVeh(x, y, z, cam, useDistortion = true) {
  const K = cam.K;
  const T = cam.T_c_v_lidar_ref || cam.T_c_v;
  const fx = K[0],
    fy = K[4],
    cx = K[2],
    cy = K[5];
  const xc = T[0] * x + T[1] * y + T[2] * z + T[3];
  const yc = T[4] * x + T[5] * y + T[6] * z + T[7];
  const zc = T[8] * x + T[9] * y + T[10] * z + T[11];
  if (zc <= 0.3) return null;
  let xn = xc / zc;
  let yn = yc / zc;
  if (useDistortion) {
    const dist = cam.dist5 || [0, 0, 0, 0, 0];
    const k1 = dist[0] || 0,
      k2 = dist[1] || 0,
      p1 = dist[2] || 0,
      p2 = dist[3] || 0,
      k3 = dist[4] || 0;
    const r2 = xn * xn + yn * yn;
    // Guard extreme radtan (e.g. camera17 k3≈-3.8) outside reliable FOV.
    if (r2 > 1.2) return null;
    const r4 = r2 * r2;
    const r6 = r4 * r2;
    const radial = 1 + k1 * r2 + k2 * r4 + k3 * r6;
    if (!Number.isFinite(radial) || Math.abs(radial) > 20) return null;
    const xpp = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn);
    const ypp = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn;
    if (!Number.isFinite(xpp) || !Number.isFinite(ypp)) return null;
    xn = xpp;
    yn = ypp;
  }
  const u = fx * xn + cx;
  const v = fy * yn + cy;
  if (!Number.isFinite(u) || !Number.isFinite(v)) return null;
  return { u, v, z: zc };
}

function projectionUsesDistortion() {
  return Boolean(el.projDistortion && el.projDistortion.checked);
}

/**
 * Project each occupied voxel cube onto the image:
 * 8 lattice corners → distorted UV → convex hull silhouette.
 * Skip any cube that is behind the camera, straddles the near plane,
 * or has a corner that flies outside a generous image FOV (avoids the
 * huge warped polygons on side/rear cams).
 */
function projectOccCubes(cam, maxCells = 60000) {
  const labs = occProjLabels || occLabels;
  const ijks = occProjIjk;
  if (!labs || !labs.length || !activeVoxel || !currentMeta || !ijks) return [];

  const v = activeVoxel;
  const x0 = currentMeta.x_range[0];
  const y0 = currentMeta.y_range[0];
  const z0 = currentMeta.z_range[0];
  const n = labs.length;
  let step = 1;
  if (n > maxCells) step = Math.ceil(n / maxCells);
  const w = cam.width || 1920;
  const h = cam.height || 1080;
  // Rectangular image bounds only — do NOT gate on r² (that draws a circle
  // on wide pinhole cams and looks like a wrong fisheye model).
  const uLo = -0.02 * w;
  const uHi = 1.02 * w;
  const vLo = -0.02 * h;
  const vHi = 1.02 * h;
  const nearZ = 1.0;
  const maxSpan = 0.12 * Math.max(w, h);

  const quads = [];
  for (let i = 0; i < n; i += step) {
    const ix = ijks[i * 3];
    const iy = ijks[i * 3 + 1];
    const iz = ijks[i * 3 + 2];
    const xa = x0 + ix * v;
    const xb = x0 + (ix + 1) * v;
    const ya = y0 + iy * v;
    const yb = y0 + (iy + 1) * v;
    const za = z0 + iz * v;
    const zb = z0 + (iz + 1) * v;
    const cx = 0.5 * (xa + xb);
    const cy = 0.5 * (ya + yb);
    const cz = 0.5 * (za + zb);
    const cProj = projectOneVeh(cx, cy, cz, cam, projectionUsesDistortion());
    if (!cProj || cProj.z < nearZ) continue;
    // Center must land inside the image — otherwise corner hulls become
    // huge radial wings (esp. camera17 / strong radtan).
    if (cProj.u < 0 || cProj.u >= w || cProj.v < 0 || cProj.v >= h) continue;

    const corners3 = [
      [xa, ya, za],
      [xb, ya, za],
      [xb, yb, za],
      [xa, yb, za],
      [xa, ya, zb],
      [xb, ya, zb],
      [xb, yb, zb],
      [xa, yb, zb],
    ];
    const pts = [];
    let zSum = 0;
    let ok = true;
    let uMin = Infinity,
      uMax = -Infinity,
      vMin = Infinity,
      vMax = -Infinity;
    for (let c = 0; c < 8; c++) {
      const p = projectOneVeh(
        corners3[c][0],
        corners3[c][1],
        corners3[c][2],
        cam,
        projectionUsesDistortion()
      );
      // All 8 corners must be valid — partial cubes → huge warped shards.
      if (!p || p.z < nearZ) {
        ok = false;
        break;
      }
      if (p.u < uLo || p.u > uHi || p.v < vLo || p.v > vHi) {
        ok = false;
        break;
      }
      pts.push(p);
      zSum += p.z;
      if (p.u < uMin) uMin = p.u;
      if (p.u > uMax) uMax = p.u;
      if (p.v < vMin) vMin = p.v;
      if (p.v > vMax) vMax = p.v;
    }
    if (!ok || pts.length !== 8) continue;
    if (uMax - uMin > maxSpan || vMax - vMin > maxSpan) continue;
    // Drop cubes whose silhouette is far larger than a voxel at that depth
    // (camera17 / strong radtan otherwise paints huge wing shards).
    const zMean = zSum / 8;
    const fx = cam.K[0] || 1000;
    const expect = (2.5 * v * fx) / Math.max(zMean, nearZ);
    if (uMax - uMin > Math.max(24, 3 * expect) || vMax - vMin > Math.max(24, 3 * expect)) {
      continue;
    }
    const hull = convexHull2D(pts);
    if (hull.length < 3) continue;
    quads.push({
      pts: hull,
      z: zSum / 8,
      lab: labs[i],
    });
  }
  quads.sort((a, b) => b.z - a.z);
  return quads;
}

/** Monotone chain convex hull on {u,v} points (returns CCW). */
function convexHull2D(points) {
  const pts = points.slice().sort((a, b) => (a.u === b.u ? a.v - b.v : a.u - b.u));
  if (pts.length <= 2) return pts;
  const cross = (o, a, b) => (a.u - o.u) * (b.v - o.v) - (a.v - o.v) * (b.u - o.u);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function fillStyleForOccLab(lab) {
  const cmode = colorModeValue();
  if (cmode === "lidar") {
    const i = lab | 0;
    const c =
      classColors && i >= 0 && i < classColors.length && classColors[i]
        ? classColors[i]
        : [180, 180, 180];
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }
  return rgbCss(lab);
}

function projectVehToImage(xyz, labels, cam, maxN = 120000, lidarIds = null) {
  const K = cam.K;
  const T = cam.T_c_v_lidar_ref || cam.T_c_v;
  const dist = projectionUsesDistortion()
    ? (cam.dist5 || [0, 0, 0, 0, 0])
    : [0, 0, 0, 0, 0];
  const w = cam.width;
  const h = cam.height;
  const fx = K[0], fy = K[4], cx = K[2], cy = K[5];
  const k1 = dist[0] || 0, k2 = dist[1] || 0, p1 = dist[2] || 0, p2 = dist[3] || 0, k3 = dist[4] || 0;
  const nAll = labels.length;
  const step = nAll > maxN ? Math.ceil(nAll / maxN) : 1;
  const out = [];
  for (let i = 0; i < nAll; i += step) {
    if (lidarIds && !lidarVisible(lidarIds[i])) continue;
    const classFilter = el.projClassFilter ? el.projClassFilter.value : "all";
    if (classFilter !== "all" && labels[i] !== Number(classFilter)) continue;
    const o = i * 3;
    const x = xyz[o], y = xyz[o + 1], z = xyz[o + 2];
    const xc = T[0] * x + T[1] * y + T[2] * z + T[3];
    const yc = T[4] * x + T[5] * y + T[6] * z + T[7];
    const zc = T[8] * x + T[9] * y + T[10] * z + T[11];
    if (zc <= 0.3) continue;
    let xn = xc / zc;
    let yn = yc / zc;
    const r2 = xn * xn + yn * yn;
    const r4 = r2 * r2;
    const r6 = r4 * r2;
    const radial = 1 + k1 * r2 + k2 * r4 + k3 * r6;
    const xpp = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn);
    const ypp = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn;
    const u = fx * xpp + cx;
    const v = fy * ypp + cy;
    if (u < -40 || v < -40 || u >= w + 40 || v >= h + 40) continue;
    out.push({
      u,
      v,
      z: zc,
      lab: labels[i],
      lid: lidarIds ? lidarIds[i] : 0,
      zVeh: z,
      fx,
      fy,
    });
  }
  out.sort((a, b) => b.z - a.z);
  return out;
}

function heightRgbCss(z, zMin, zMax) {
  const span = Math.max(1e-6, zMax - zMin);
  const t = Math.max(0, Math.min(1, (z - zMin) / span));
  // Deliberately discrete and non-adjacent in RGB space. Smooth ramps hide
  // sub-metre objects; these bands make a cone cross several obvious colors.
  const bands = [
    [25, 25, 210],
    [0, 120, 255],
    [0, 235, 255],
    [0, 220, 80],
    [245, 245, 0],
    [255, 145, 0],
    [255, 35, 20],
    [255, 0, 190],
  ];
  const rgb = bands[Math.min(bands.length - 1, Math.floor(t * bands.length))];
  return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
}

/**
 * Occupancy: each voxel cube projected as filled faces (8 corners → 6 faces).
 * Points: single image pixels (optional min px from slider)
 * RGB photo is always drawn first; overlays stay semi-transparent.
 */
function drawProjectionOnCanvas(canvas, img, cam, mode, ptMinPx = 1) {
  const ctx = canvas.getContext("2d");
  const cw = canvas.width;
  const ch = canvas.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, cw, ch);
  const scale = Math.min(cw / cam.width, ch / cam.height);
  const dw = cam.width * scale;
  const dh = cam.height * scale;
  const ox = (cw - dw) / 2;
  const oy = (ch - dh) / 2;
  const hasImg = img && img.complete && img.naturalWidth > 0;
  if (hasImg) {
    ctx.drawImage(img, ox, oy, dw, dh);
  } else {
    ctx.fillStyle = "#222";
    ctx.fillRect(ox, oy, dw, dh);
    ctx.fillStyle = "#888";
    ctx.font = "14px sans-serif";
    ctx.fillText("loading image…", ox + 12, oy + 28);
  }

  if (mode === "none") return 0;

  const alpha = el.projAlpha ? Number(el.projAlpha.value) : 0.4;
  const maxN = canvas.width >= cam.width * 0.8 ? 160000 : 70000;

  let n = 0;
  if (mode === "occ" || mode === "both") {
    if (occProjIjk && occProjLabels && occProjLabels.length && activeVoxel) {
      const faces = projectOccCubes(cam, maxN);
      ctx.globalAlpha = alpha;
      for (const q of faces) {
        ctx.fillStyle = fillStyleForOccLab(q.lab);
        ctx.beginPath();
        ctx.moveTo(ox + q.pts[0].u * scale, oy + q.pts[0].v * scale);
        for (let k = 1; k < q.pts.length; k++) {
          ctx.lineTo(ox + q.pts[k].u * scale, oy + q.pts[k].v * scale);
        }
        ctx.closePath();
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      n += faces.length;
    }
  }

  if (mode === "points" || mode === "both") {
    const wantFrame = !el.projPointSource || el.projPointSource.value === "frame";
    const useFrame = wantFrame && framePtXYZ && framePtLabels;
    const useFilt = !useFrame && pointsObj && pointsObj.userData && pointsObj.userData.projXYZ;
    const xyz = useFrame ? framePtXYZ : (useFilt ? pointsObj.userData.projXYZ : ptXYZ);
    const labs = useFrame ? framePtLabels : (useFilt ? pointsObj.userData.projLab : ptLabels);
    const lids = useFrame ? framePtLidar : (useFilt ? pointsObj.userData.projLid : ptLidar);
    if (xyz && labs && labs.length) {
      const pts = projectVehToImage(xyz, labs, cam, maxN, lids);
      const sourcePx = Math.max(1, ptMinPx | 0);
      const px = Math.max(1, sourcePx * scale);
      const cmode = colorModeValue();
      const heightMode = el.projColorMode && el.projColorMode.value === "height";
      const coneOnly = el.projClassFilter && el.projClassFilter.value === "10";
      let zMin = el.projHeightMin ? Number(el.projHeightMin.value) : -2.2;
      let zMax = el.projHeightMax ? Number(el.projHeightMax.value) : 0.5;
      if (!Number.isFinite(zMin)) zMin = -2.2;
      if (!Number.isFinite(zMax)) zMax = 0.5;
      if (zMax <= zMin) zMax = zMin + 0.1;
      for (const p of pts) {
        // Do not add hidden opacity: dense full-frame deskew points otherwise
        // cover the RGB image and make a successfully loaded frame look blank.
        ctx.globalAlpha = alpha;
        ctx.fillStyle = coneOnly
          ? "rgb(255,80,0)"
          : heightMode
          ? heightRgbCss(p.zVeh, zMin, zMax)
          : (cmode === "lidar" ? rgbCssLidar(p.lid || 0) : rgbCss(p.lab));
        ctx.fillRect(
          ox + p.u * scale - px * 0.5,
          oy + p.v * scale - px * 0.5,
          px,
          px
        );
      }
      ctx.globalAlpha = 1;
      n += pts.length;
    }
  }
  return n;
}

function refreshCamProjections() {
  if (!currentMeta) return;
  const mode = el.projMode.value;
  const radius = Number(el.projRadius.value);
  el.cams.querySelectorAll(".cam-card").forEach((card) => {
    const canvas = card.querySelector("canvas.thumb");
    const img = card.querySelector("img.base");
    const cam = card._cam;
    if (!canvas || !cam) return;
    drawProjectionOnCanvas(canvas, img, cam, mode, radius);
  });
  if (el.lightbox.classList.contains("open") && lb.cam && lb.img) {
    drawProjectionOnCanvas(el.lbCanvas, lb.img, lb.cam, mode, radius);
  }
}

function applyLbTransform() {
  const c = el.lbCanvas;
  c.style.transform = `translate(calc(-50% + ${lb.tx}px), calc(-50% + ${lb.ty}px)) scale(${lb.scale})`;
}

function openLightbox(cam, img) {
  lb.img = img;
  lb.cam = cam;
  lb.tx = 0;
  lb.ty = 0;
  el.lbTitle.textContent = cam.name;
  const c = el.lbCanvas;
  // Native-ish resolution so zoom stays sharp
  c.width = cam.width;
  c.height = cam.height;
  drawProjectionOnCanvas(c, img, cam, el.projMode.value, Number(el.projRadius.value));
  // Fit into stage on open (then Zoom+/wheel to enlarge)
  const sw = Math.max(320, el.lbStage.clientWidth - 40);
  const sh = Math.max(240, el.lbStage.clientHeight - 40);
  lb.scale = Math.min(1, sw / cam.width, sh / cam.height);
  applyLbTransform();
  el.lightbox.classList.add("open");
}

function closeLightbox() {
  el.lightbox.classList.remove("open");
}

el.lbClose.addEventListener("click", closeLightbox);
el.lbZoomIn.addEventListener("click", () => {
  lb.scale = Math.min(8, lb.scale * 1.25);
  applyLbTransform();
});
el.lbZoomOut.addEventListener("click", () => {
  lb.scale = Math.max(0.2, lb.scale / 1.25);
  applyLbTransform();
});
el.lbReset.addEventListener("click", () => {
  lb.tx = 0;
  lb.ty = 0;
  if (lb.cam) {
    const sw = Math.max(320, el.lbStage.clientWidth - 40);
    const sh = Math.max(240, el.lbStage.clientHeight - 40);
    lb.scale = Math.min(1, sw / lb.cam.width, sh / lb.cam.height);
  } else {
    lb.scale = 1;
  }
  applyLbTransform();
});
el.lbStage.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const f = e.deltaY > 0 ? 0.9 : 1.1;
    lb.scale = Math.min(8, Math.max(0.2, lb.scale * f));
    applyLbTransform();
  },
  { passive: false }
);
el.lbStage.addEventListener("pointerdown", (e) => {
  lb.dragging = true;
  lb.lx = e.clientX;
  lb.ly = e.clientY;
  el.lbStage.classList.add("dragging");
  el.lbStage.setPointerCapture(e.pointerId);
});
el.lbStage.addEventListener("pointermove", (e) => {
  if (!lb.dragging) return;
  lb.tx += e.clientX - lb.lx;
  lb.ty += e.clientY - lb.ly;
  lb.lx = e.clientX;
  lb.ly = e.clientY;
  applyLbTransform();
});
el.lbStage.addEventListener("pointerup", () => {
  lb.dragging = false;
  el.lbStage.classList.remove("dragging");
});
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeLightbox();
});

function renderCams(meta) {
  el.cams.innerHTML = "";
  for (const cam of meta.cameras || []) {
    const card = document.createElement("div");
    card.className = "cam-card";
    card._cam = cam;
    const name = document.createElement("div");
    name.className = "name";
    name.innerHTML = `<span>${cam.name}</span><span>${cam.width}×${cam.height} · click to zoom</span>`;
    const stage = document.createElement("div");
    stage.className = "stage";
    const img = document.createElement("img");
    img.className = "base";
    img.crossOrigin = "anonymous";
    img.style.display = "none";
    img.src = camImageUrl(cam);
    const canvas = document.createElement("canvas");
    canvas.className = "thumb";
    canvas.width = 1280;
    canvas.height = 720;
    stage.appendChild(img);
    stage.appendChild(canvas);
    img.onload = () => {
      drawProjectionOnCanvas(canvas, img, cam, el.projMode.value, Number(el.projRadius.value));
    };
    stage.addEventListener("click", () => openLightbox(cam, img));
    card.appendChild(name);
    card.appendChild(stage);
    card.addEventListener("click", (e) => {
      if (e.target.closest(".stage")) return;
      [...el.cams.children].forEach((c) => c.classList.remove("active"));
      card.classList.add("active");
    });
    el.cams.appendChild(card);
  }
}

async function loadFrame(frameEntry) {
  setStatus("Loading frame…");
  frameDir =
    frameEntry.dir ||
    (frameEntry.meta_uri ? frameEntry.meta_uri.replace(/\/meta\.json$/, "") : null);
  const metaUrl = frameEntry.meta_uri
    ? assetUrl(frameEntry.meta_uri)
    : `${sceneRoot}/${frameDir}/meta.json`;
  const meta = await fetchJson(metaUrl);
  currentMeta = meta;
  classColors = resolveClassColors(meta);
  classNames =
    (index && index.taxonomy && index.taxonomy.fine && index.taxonomy.fine.names) ||
    meta.class_names ||
    null;
  renderClassLegend();
  const title = `${index.clip_id || index.clip}  ·  ts=${meta.timestamp}  ·  voxel=${(meta.grid && meta.grid.voxel) || meta.voxel}m  ·  occ=${(meta.stats && meta.stats.n_occ) || meta.n_occ}`;
  el.titleMeta.textContent = title;
  el.titleMeta.title = title;

  // normalize ranges for ROI/grid helpers
  if (meta.grid) {
    meta.voxel = meta.grid.voxel;
    meta.x_range = meta.grid.x_range;
    meta.y_range = meta.grid.y_range;
    meta.z_range = meta.grid.z_range;
    meta.occ_shape = meta.grid.shape;
    meta.n_occ = (meta.stats && meta.stats.n_occ) || meta.n_occ;
  }

  const ijkUrl = occAsset(meta, "ijk");
  const labUrl = occAsset(meta, "labels");
  const ijkBuf = await fetchBin(ijkUrl);
  const labBuf = await fetchBin(labUrl);
  const ijkArr = new Int32Array(ijkBuf);
  const labArr = new Uint8Array(labBuf);
  exportedOcc = {
    voxel: meta.voxel,
    ijk: ijkArr,
    labels: labArr,
  };

  clearPoints();
  ptXYZ = null;
  ptLabels = null;
  ptLidar = null;
  framePtXYZ = null;
  framePtLabels = null;
  framePtLidar = null;
  let ptsNote = "not exported";
  let frameDynXYZ = null;
  let frameDynLab = null;
  let frameDynLid = null;

  if (meta.points || (meta.assets && meta.assets.points)) {
    try {
      const xyzUrl = pointsAsset(meta, "xyz");
      const labUrlP = pointsAsset(meta, "labels");
      const lidUrl = pointsAsset(meta, "lidar_id");
      const xyzBuf = await fetchBin(xyzUrl);
      const plabBuf = await fetchBin(labUrlP);
      const rawXYZ = new Float32Array(xyzBuf);
      const rawLab = new Uint8Array(plabBuf);
      let rawLid = null;
      if (lidUrl) {
        rawLid = new Uint8Array(await fetchBin(lidUrl));
      }
      // Keep frame dynamic only — static comes from clip static_agg + ego_pose.
      const dynIdx = [];
      for (let i = 0; i < rawLab.length; i++) {
        if (FINE_DYNAMIC.has(rawLab[i] | 0)) dynIdx.push(i);
      }
      frameDynXYZ = new Float32Array(dynIdx.length * 3);
      frameDynLab = new Uint8Array(dynIdx.length);
      frameDynLid = new Uint8Array(dynIdx.length);
      for (let k = 0; k < dynIdx.length; k++) {
        const i = dynIdx[k];
        frameDynXYZ[k * 3] = rawXYZ[i * 3];
        frameDynXYZ[k * 3 + 1] = rawXYZ[i * 3 + 1];
        frameDynXYZ[k * 3 + 2] = rawXYZ[i * 3 + 2];
        frameDynLab[k] = rawLab[i];
        frameDynLid[k] = rawLid ? rawLid[i] : 0;
      }
      ptXYZ = rawXYZ;
      ptLabels = rawLab;
      ptLidar = rawLid;
      framePtXYZ = rawXYZ;
      framePtLabels = rawLab;
      framePtLidar = rawLid;
      ptsNote = rawLab.length.toLocaleString();
    } catch (e) {
      ptsNote = `export broken: ${e}`;
    }
  }

  const frameSensorXyzUrl = frameSensorPointsAsset(meta, "xyz");
  const frameSensorLabelsUrl = frameSensorPointsAsset(meta, "labels");
  const frameSensorLidarUrl = frameSensorPointsAsset(meta, "lidar_id");
  if (frameSensorXyzUrl && frameSensorLabelsUrl) {
    framePtXYZ = new Float32Array(await fetchBin(frameSensorXyzUrl));
    framePtLabels = new Uint8Array(await fetchBin(frameSensorLabelsUrl));
    framePtLidar = frameSensorLidarUrl
      ? new Uint8Array(await fetchBin(frameSensorLidarUrl))
      : null;
  }

  // Prefer clip static_agg (map→vehicle) + frame dynamic. Same static points
  // every frame; only ego_pose changes coordinates.
  if (staticAgg && meta.ego_pose) {
    const xr = meta.x_range;
    const yr = meta.y_range;
    const zr = meta.z_range;
    const st = staticAggInVehicle(
      meta.ego_pose,
      staticAgg.full ? null : [xr[0] * 1.5, xr[1] * 1.5],
      staticAgg.full ? null : yr,
      staticAgg.full ? null : [zr[0] - 2, zr[1] + 5]
    );
    const merged = staticAgg.full
      ? st
      : mergeStaticAndDynamicPoints(st, frameDynXYZ, frameDynLab, frameDynLid);
    ptXYZ = merged.xyz;
    ptLabels = merged.labels;
    ptLidar = merged.lidar;
    ptsNote = staticAgg.full
      ? `${merged.labels.length.toLocaleString()} (all deskew frames, pose-only)`
      : `${merged.labels.length.toLocaleString()} (static_agg⊕dyn)`;
    // Occupancy bins were already built from the same aggregate at export;
    // keep them (fast). Points now match: same static set, pose-only change.
    applyOccupancy(ijkArr, labArr, meta.voxel, "exported grid (static_agg⊕dyn)");
  } else {
    applyOccupancy(ijkArr, labArr, meta.voxel, "exported grid");
  }

  if (ptXYZ && ptLabels) {
    try {
      buildPoints();
      let xmin = Infinity,
        xmax = -Infinity,
        ymin = Infinity,
        ymax = -Infinity,
        zmin = Infinity,
        zmax = -Infinity;
      for (let i = 0; i < ptXYZ.length; i += 3) {
        xmin = Math.min(xmin, ptXYZ[i]);
        xmax = Math.max(xmax, ptXYZ[i]);
        ymin = Math.min(ymin, ptXYZ[i + 1]);
        ymax = Math.max(ymax, ptXYZ[i + 1]);
        zmin = Math.min(zmin, ptXYZ[i + 2]);
        zmax = Math.max(zmax, ptXYZ[i + 2]);
      }
      document.getElementById("axisHelp").innerHTML = `
        Arrows follow stored numeric signs.<br/>
        <span style="color:var(--x)">+X</span> x∈[${xmin.toFixed(1)}, ${xmax.toFixed(1)}]<br/>
        <span style="color:var(--y)">+Y</span> y∈[${ymin.toFixed(1)}, ${ymax.toFixed(1)}]<br/>
        <span style="color:var(--z)">+Z</span> z∈[${zmin.toFixed(1)}, ${zmax.toFixed(1)}]<br/>
        Three map: (x,z,y) so XY ground stays horizontal.
      `;
    } catch (e) {
      ptsNote = `build points: ${e}`;
    }
  }
  syncColorModeUi();
  updateRoiHelper();
  // re-apply color/ROI filters now that points+occ are both loaded
  rebuildColoredViews();
  const odBoxCount = buildOdBoxes(meta);

  el.sceneInfo.innerHTML = `
    <div>occ voxels: <b>${occLabels ? occLabels.length.toLocaleString() : meta.n_occ.toLocaleString()}</b></div>
    <div>active voxel: <b>${activeVoxel}m</b> (exported ${meta.voxel}m)</div>
    <div>grid snap: floor((p-origin)/v)</div>
    <div>x: [${meta.x_range.join(", ")}]</div>
    <div>y: [${meta.y_range.join(", ")}]</div>
    <div>z: [${meta.z_range.join(", ")}]</div>
    <div>points: <b>${ptsNote}</b></div>
    <div>OD boxes: <b>${odBoxCount}</b> (yellow=oracle ID matched, red=unmatched)</div>
  `;

  renderCams(meta);
  fitCamera(meta);
  if (pointsObj && el.togPts.checked && occMesh && el.togOcc.checked) {
    occMesh.material.transparent = true;
    occMesh.material.opacity = Math.min(Number(el.occOpacity.value), 0.35);
    occMesh.material.needsUpdate = true;
  }
  setStatus(`ready · occ=${occLabels.length} · voxel=${activeVoxel}m · points=${ptsNote}`);
}

function updateFramePos() {
  if (!el.framePos || !index || !index.frames) {
    if (el.framePos) el.framePos.textContent = "—";
    return;
  }
  const n = index.frames.length;
  const i = Math.min(Math.max(0, frameIndex), Math.max(0, n - 1));
  el.framePos.textContent = n ? `${i + 1} / ${n}` : "—";
  if (el.btnPrevFrame) el.btnPrevFrame.disabled = i <= 0;
  if (el.btnNextFrame) el.btnNextFrame.disabled = i >= n - 1;
}

async function loadFrameByIndex(i) {
  if (!index || !index.frames || !index.frames.length) return;
  const n = index.frames.length;
  frameIndex = ((i % n) + n) % n;
  const fr = index.frames[frameIndex];
  if (fr.meta_uri && !fr.dir) {
    fr.dir = fr.meta_uri.replace(/\/meta\.json$/, "");
  }
  if (el.frameSelect) {
    el.frameSelect.value = String(fr.timestamp || fr.frame_id);
  }
  updateFramePos();
  await loadFrame(fr);
}

async function loadClip(clipId) {
  const entry = clipsCatalog.find((c) => c.id === clipId || c.clip_id === clipId);
  sceneRoot = (entry && entry.url ? entry.url : `/scenes/${clipId}`).replace(/\/$/, "");
  // keep URL shareable
  const u = new URL(location.href);
  u.searchParams.set("scene", sceneRoot);
  history.replaceState(null, "", u);
  setStatus(`Loading clip ${clipId}…`);
  index = await fetchJson(`${sceneRoot}/index.json`);
  if (!index.clip && index.clip_id) index.clip = index.clip_id;
  await loadStaticAggFromIndex();
  if (el.occVariant) {
    const config = index.occupancy_variants;
    el.occVariant.innerHTML = "";
    const variants = (config && config.variants) || [{ id: "litept", name: "LitePT dynamic" }];
    for (const variant of variants) {
      const opt = document.createElement("option");
      opt.value = variant.id;
      opt.textContent = variant.name || variant.id;
      el.occVariant.appendChild(opt);
    }
    currentOccVariant = (config && config.default) || variants[0].id;
    el.occVariant.value = currentOccVariant;
  }
  el.frameSelect.innerHTML = "";
  for (const fr of index.frames) {
    const opt = document.createElement("option");
    opt.value = String(fr.timestamp || fr.frame_id);
    opt.textContent = `${fr.timestamp || fr.frame_id}  (occ=${fr.n_occ})`;
    el.frameSelect.appendChild(opt);
  }
  if (!index.frames.length) {
    setStatus("No frames in index.json");
    updateFramePos();
    return;
  }
  await loadFrameByIndex(0);
  if (typeof refreshVideoList === "function") refreshVideoList();
}

async function boot() {
  // discover clips
  try {
    const cat = await fetchJson("/api/clips");
    clipsCatalog = cat.clips || [];
    const preferred =
      (sceneRoot && sceneRoot.replace(/^\/scenes\//, "").replace(/^\/scene\/?/, "")) ||
      cat.default_clip ||
      (clipsCatalog[0] && clipsCatalog[0].id);
    if (el.clipSelect) {
      el.clipSelect.innerHTML = "";
      for (const c of clipsCatalog) {
        const opt = document.createElement("option");
        opt.value = c.id;
        opt.textContent = `${c.id}  (${c.n_frames} frames)`;
        el.clipSelect.appendChild(opt);
      }
      if (preferred && clipsCatalog.some((c) => c.id === preferred)) {
        el.clipSelect.value = preferred;
      } else if (clipsCatalog.length) {
        el.clipSelect.value = clipsCatalog[0].id;
      }
    }
    if (clipsCatalog.length) {
      await loadClip(el.clipSelect ? el.clipSelect.value : preferred);
      return;
    }
  } catch (e) {
    console.warn(" /api/clips unavailable, fallback single scene", e);
  }
  // fallback: legacy ?scene=/scene
  if (!sceneRoot) sceneRoot = "/scene";
  index = await fetchJson(`${sceneRoot}/index.json`);
  if (!index.clip && index.clip_id) index.clip = index.clip_id;
  await loadStaticAggFromIndex();
  el.frameSelect.innerHTML = "";
  for (const fr of index.frames) {
    const opt = document.createElement("option");
    opt.value = String(fr.timestamp || fr.frame_id);
    opt.textContent = `${fr.timestamp || fr.frame_id}  (occ=${fr.n_occ})`;
    el.frameSelect.appendChild(opt);
  }
  if (!index.frames.length) {
    setStatus("No frames in index.json");
    return;
  }
  await loadFrameByIndex(0);
}

el.frameSelect.addEventListener("change", async () => {
  const i = index.frames.findIndex(
    (f) => String(f.timestamp || f.frame_id) === String(el.frameSelect.value)
  );
  if (i >= 0) await loadFrameByIndex(i);
});
if (el.occVariant) {
    await loadStaticAggFromIndex();
  el.occVariant.addEventListener("change", async () => {
    currentOccVariant = el.occVariant.value;
    await loadFrameByIndex(frameIndex);
  });
}
if (el.clipSelect) {
  el.clipSelect.addEventListener("change", async () => {
    await loadClip(el.clipSelect.value);
  });
}
if (el.btnPrevFrame) {
  el.btnPrevFrame.addEventListener("click", () => loadFrameByIndex(frameIndex - 1));
}
if (el.btnNextFrame) {
  el.btnNextFrame.addEventListener("click", () => loadFrameByIndex(frameIndex + 1));
}
window.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA")) {
    return;
  }
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    loadFrameByIndex(frameIndex - 1);
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    loadFrameByIndex(frameIndex + 1);
  }
});
el.togOcc.addEventListener("change", () => {
  if (occMesh) occMesh.visible = el.togOcc.checked;
});
el.togPts.addEventListener("change", () => {
  if (pointsObj) {
    pointsObj.visible = el.togPts.checked;
    if (occMesh) {
      const op = el.togPts.checked && el.togOcc.checked
        ? Math.min(Number(el.occOpacity.value), 0.35)
        : Number(el.occOpacity.value);
      occMesh.material.opacity = op;
      occMesh.material.transparent = op < 0.999;
      occMesh.material.needsUpdate = true;
    }
    if (el.togPts.checked && el.togOcc.checked) {
      setStatus("Points on · occ auto-dimmed (raise Opacity if needed)");
    }
  } else if (el.togPts.checked) {
    setStatus("No points in this scene — re-export with --export-points");
  }
});
el.togOdBoxes.addEventListener("change", () => {
  if (odBoxesGroup) odBoxesGroup.visible = el.togOdBoxes.checked;
});
el.togGrid.addEventListener("change", () => {
  gridHelper.visible = el.togGrid.checked;
});
el.togAxes.addEventListener("change", () => {
  if (axesGroup) axesGroup.visible = el.togAxes.checked;
});
el.occOpacity.addEventListener("input", () => {
  if (!occMesh) return;
  const op = Number(el.occOpacity.value);
  occMesh.material.opacity = op;
  occMesh.material.transparent = op < 0.999;
  occMesh.material.needsUpdate = true;
});
function rebuildOcc() {
  buildOccMesh();
  refreshCamProjections();
  setStatus(`voxel display size×=${el.occGrow.value} gap=${el.occGap.value}`);
}
el.occGap.addEventListener("input", rebuildOcc);
el.occGrow.addEventListener("input", rebuildOcc);
el.ptSize.addEventListener("input", () => {
  if (pointsObj) {
    pointsObj.material.size = Number(el.ptSize.value);
    pointsObj.material.needsUpdate = true;
  }
});
el.projMode.addEventListener("change", refreshCamProjections);
el.projRadius.addEventListener("input", () => {
  if (el.projRadiusValue) el.projRadiusValue.textContent = `${el.projRadius.value} px`;
  refreshCamProjections();
});
if (el.projAlpha) el.projAlpha.addEventListener("input", () => {
  if (el.projAlphaValue) el.projAlphaValue.textContent = Number(el.projAlpha.value).toFixed(2);
  refreshCamProjections();
});
if (el.projColorMode) el.projColorMode.addEventListener("change", refreshCamProjections);
if (el.projClassFilter) el.projClassFilter.addEventListener("change", refreshCamProjections);
if (el.projHeightMin) el.projHeightMin.addEventListener("input", refreshCamProjections);
if (el.projHeightMax) el.projHeightMax.addEventListener("input", refreshCamProjections);
if (el.projPointSource) el.projPointSource.addEventListener("change", refreshCamProjections);
if (el.projDistortion) el.projDistortion.addEventListener("change", refreshCamProjections);
el.btnRefreshProj.addEventListener("click", () => {
  refreshCamProjections();
  if (el.lightbox.classList.contains("open") && lb.cam && lb.img) {
    drawProjectionOnCanvas(
      el.lbCanvas,
      lb.img,
      lb.cam,
      el.projMode.value,
      Number(el.projRadius.value)
    );
  }
  setStatus(`projection refreshed · mode=${el.projMode.value}`);
});
el.btnRebuildOcc.addEventListener("click", () => {
  try {
    const v = Number(el.voxelSize.value);
    setStatus(`Rebuilding occ @ ${v}m…`);
    const t0 = performance.now();
    const out = voxelizeFromPoints(v);
    applyOccupancy(out.ijk, out.labels, out.voxel, "rebuilt from points");
    refreshCamProjections();
    el.sceneInfo.innerHTML = `
      <div>occ voxels: <b>${out.nOcc.toLocaleString()}</b></div>
      <div>active voxel: <b>${out.voxel}m</b> (exported ${currentMeta.voxel}m)</div>
      <div>grid snap: floor((p-origin)/v)</div>
      <div>x: [${currentMeta.x_range.join(", ")}]</div>
      <div>y: [${currentMeta.y_range.join(", ")}]</div>
      <div>z: [${currentMeta.z_range.join(", ")}]</div>
      <div>points: <b>${ptLabels ? ptLabels.length.toLocaleString() : "—"}</b></div>
    `;
    setStatus(
      `occ rebuilt · voxel=${out.voxel}m · n=${out.nOcc.toLocaleString()} · ${(
        performance.now() - t0
      ).toFixed(0)}ms`
    );
  } catch (e) {
    setStatus(String(e));
  }
});
el.btnResetOcc.addEventListener("click", () => {
  if (!exportedOcc || !currentMeta) {
    setStatus("No exported occupancy loaded");
    return;
  }
  applyOccupancy(
    exportedOcc.ijk,
    exportedOcc.labels,
    exportedOcc.voxel,
    "exported grid"
  );
  refreshCamProjections();
  el.sceneInfo.innerHTML = `
    <div>occ voxels: <b>${exportedOcc.labels.length.toLocaleString()}</b></div>
    <div>active voxel: <b>${exportedOcc.voxel}m</b> (exported ${currentMeta.voxel}m)</div>
    <div>grid snap: floor((p-origin)/v)</div>
    <div>x: [${currentMeta.x_range.join(", ")}]</div>
    <div>y: [${currentMeta.y_range.join(", ")}]</div>
    <div>z: [${currentMeta.z_range.join(", ")}]</div>
    <div>points: <b>${ptLabels ? ptLabels.length.toLocaleString() : "—"}</b></div>
  `;
  setStatus(`reset to exported · voxel=${exportedOcc.voxel}m · n=${exportedOcc.labels.length}`);
});
el.btnFit.addEventListener("click", () => {
  if (currentMeta) fitCamera(currentMeta);
});

function onColorOrFilterChange() {
  syncColorModeUi();
  rebuildColoredViews();
  setStatus(`color=${colorModeValue()} · ROI clip=${el.togRoiClip && el.togRoiClip.checked}`);
}

if (el.colorMode) el.colorMode.addEventListener("change", onColorOrFilterChange);
for (const id of ["togDyn", "togSta", "togFree", "togNoise", "togLid1", "togLid2", "togLid14"]) {
  if (el[id]) el[id].addEventListener("change", onColorOrFilterChange);
}
if (el.btnApplyRoi) {
  el.btnApplyRoi.addEventListener("click", () => {
    readRoiFromInputs();
    rebuildColoredViews();
    setStatus(
      `ROI x[${roi.x0},${roi.x1}] y[${roi.y0},${roi.y1}] z[${roi.z0},${roi.z1}]`
    );
  });
}
if (el.btnResetRoi) {
  el.btnResetRoi.addEventListener("click", () => {
    el.roiX0.value = -24;
    el.roiX1.value = 24;
    el.roiY0.value = -25;
    el.roiY1.value = 150;
    el.roiZ0.value = -5;
    el.roiZ1.value = 3;
    readRoiFromInputs();
    rebuildColoredViews();
    setStatus("ROI reset to default [-24,24]×[-25,150]×[-5,3]");
  });
}
if (el.togRoiBox) {
  el.togRoiBox.addEventListener("change", () => updateRoiHelper());
}
if (el.togRoiClip) {
  el.togRoiClip.addEventListener("change", () => {
    rebuildColoredViews();
  });
}
syncColorModeUi();
readRoiFromInputs();
updateRoiHelper();

let vidPollTimer = null;

function setVidStatus(msg) {
  if (el.vidStatus) el.vidStatus.textContent = msg || "";
}

function currentClipId() {
  if (el.clipSelect && el.clipSelect.value) return el.clipSelect.value;
  const m = String(sceneRoot || "").match(/\/scenes\/([^/]+)/);
  return m ? m[1] : "";
}

async function refreshVideoList() {
  if (!el.vidList) return;
  try {
    const cid = currentClipId();
    const r = await fetch(`/api/video/list?clip_id=${encodeURIComponent(cid)}`);
    const data = await r.json();
    const vids = data.videos || [];
    if (!vids.length) {
      el.vidList.innerHTML = "No videos yet.";
      return;
    }
    el.vidList.innerHTML = vids
      .map((v) => {
        const mb = (v.bytes / (1024 * 1024)).toFixed(1);
        return `<div><a href="${v.url}" download>${v.name}</a> · ${mb} MB</div>`;
      })
      .join("");
  } catch (e) {
    el.vidList.textContent = String(e);
  }
}

async function pollVideoJob() {
  try {
    const cid = currentClipId();
    const r = await fetch(`/api/video/status?clip_id=${encodeURIComponent(cid)}`);
    const s = await r.json();
    const pct = s.n ? Math.round((100 * (s.frame || 0)) / s.n) : Math.round(100 * (s.progress || 0));
    if (s.state === "running") {
      setVidStatus(`exporting… ${s.message || ""} (${pct}%)`);
      return true;
    }
    if (s.state === "done") {
      setVidStatus(`done · ${s.relpath || s.path || "ok"}`);
      await refreshVideoList();
      return false;
    }
    if (s.state === "error") {
      setVidStatus(`error · ${s.message || ""}`);
      return false;
    }
    setVidStatus(s.state || "idle");
    return false;
  } catch (e) {
    setVidStatus(String(e));
    return false;
  }
}

function startVidPoll() {
  if (vidPollTimer) clearInterval(vidPollTimer);
  vidPollTimer = setInterval(async () => {
    const keep = await pollVideoJob();
    if (!keep && vidPollTimer) {
      clearInterval(vidPollTimer);
      vidPollTimer = null;
    }
  }, 1000);
}

if (el.btnExportVid) {
  el.btnExportVid.addEventListener("click", async () => {
    setVidStatus("starting…");
    try {
      const body = {
        clip_id: currentClipId(),
        mode: el.vidMode.value,
        fps: Number(el.vidFps.value) || 5,
        max_frames: Number(el.vidMaxFrames.value) || 0,
        tile_w: 960,
        tile_h: 540,
      };
      const r = await fetch("/api/video/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await r.json();
      if (!r.ok) {
        setVidStatus(data.error || JSON.stringify(data));
        return;
      }
      setVidStatus(`started · job=${data.job_id || "?"}`);
      startVidPoll();
    } catch (e) {
      setVidStatus(String(e));
    }
  });
}
if (el.btnRefreshVid) {
  el.btnRefreshVid.addEventListener("click", () => {
    refreshVideoList();
    pollVideoJob();
  });
}
refreshVideoList();
pollVideoJob();

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

boot().catch((e) => {
  console.error(e);
  setStatus(String(e));
  el.sceneInfo.textContent = String(e);
});
