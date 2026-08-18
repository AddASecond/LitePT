import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const qs = new URLSearchParams(location.search);
const SCENE_ROOT = (qs.get("scene") || "/scene").replace(/\/$/, "");

const el = {
  frameSelect: document.getElementById("frameSelect"),
  titleMeta: document.getElementById("titleMeta"),
  sceneInfo: document.getElementById("sceneInfo"),
  status: document.getElementById("status"),
  cams: document.getElementById("cams"),
  togOcc: document.getElementById("togOcc"),
  togPts: document.getElementById("togPts"),
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
  projAlpha: document.getElementById("projAlpha"),
  btnRefreshProj: document.getElementById("btnRefreshProj"),
  btnFit: document.getElementById("btnFit"),
  classLegend: document.getElementById("classLegend"),
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
let gridHelper = null;
let axesGroup = null;
let classColors = null;
let classNames = null;

let occIjk = null; // Int32 ix,iy,iz
let occLabels = null;
let occCenters = null; // for projection
let activeVoxel = null;
let exportedOcc = null; // { voxel, ijk: Int32Array, labels: Uint8Array }
let ptXYZ = null;
let ptLabels = null;

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
 * Vehicle coords (from data files): x,y,z as stored.
 * Three.js: X right on screen when looking -Z, Y up, Z toward camera.
 * Map: Three(x,y,z) = (veh.x, veh.z, veh.y)
 * so ground XY (z≈0) lies on Three XZ with Y-up — default GridHelper plane.
 */
function vehToThree(x, y, z, out = new THREE.Vector3()) {
  return out.set(x, z, y);
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

function buildAxes() {
  if (axesGroup) scene.remove(axesGroup);
  axesGroup = new THREE.Group();
  // +X veh → Three +X
  axesGroup.add(makeAxisArrow(new THREE.Vector3(1, 0, 0), 0xff4d4d, 20));
  // +Y veh → Three +Z
  axesGroup.add(makeAxisArrow(new THREE.Vector3(0, 0, 1), 0x3dde6a, 20));
  // +Z veh → Three +Y
  axesGroup.add(makeAxisArrow(new THREE.Vector3(0, 1, 0), 0x4da3ff, 20));
  axesGroup.add(makeSprite("+X", new THREE.Vector3(22, 0.8, 0), "#ff4d4d"));
  axesGroup.add(makeSprite("+Y", new THREE.Vector3(0, 0.8, 22), "#3dde6a"));
  axesGroup.add(makeSprite("+Z", new THREE.Vector3(0, 22, 0), "#4da3ff"));
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

function colorFromLabel(label) {
  const i = label | 0;
  if (!classColors || i < 0 || i >= classColors.length || !classColors[i]) {
    return new THREE.Color(0.7, 0.7, 0.7);
  }
  const c = classColors[i];
  return new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255);
}

function rgbCss(label) {
  const i = label | 0;
  if (!classColors || i < 0 || i >= classColors.length || !classColors[i]) {
    return "rgb(180,180,180)";
  }
  const c = classColors[i];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function renderClassLegend() {
  if (!el.classLegend) return;
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
  const n = occLabels.length;
  const v = activeVoxel;
  const x0 = currentMeta.x_range[0];
  const y0 = currentMeta.y_range[0];
  const z0 = currentMeta.z_range[0];
  const gap = Math.max(0, Number(el.occGap.value));
  const grow = Number(el.occGrow.value);
  // size fills the cell; grow>1 seals hairline cracks from float/raster
  const size = Math.max(1e-4, v * (1.0 - gap) * grow);

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

  for (let i = 0; i < n; i++) {
    const ix = occIjk[i * 3];
    const iy = occIjk[i * 3 + 1];
    const iz = occIjk[i * 3 + 2];
    const vx = x0 + (ix + 0.5) * v;
    const vy = y0 + (iy + 0.5) * v;
    const vz = z0 + (iz + 0.5) * v;
    centers[i * 3] = vx;
    centers[i * 3 + 1] = vy;
    centers[i * 3 + 2] = vz;
    vehToThree(vx, vy, vz, dummy.position);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    color.copy(colorFromLabel(occLabels[i]));
    mesh.setColorAt(i, color);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.frustumCulled = false;
  occMesh = mesh;
  occMesh.visible = el.togOcc.checked;
  scene.add(mesh);
  occCenters = centers; // exact grid centers for projection
}

/**
 * Client-side grid voxelization from currently loaded points.
 * Same rule as Python: floor((p-origin)/v), first/majority label per cell.
 */
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
  const n = ptLabels.length;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const o = i * 3;
    positions[o] = ptXYZ[o];
    positions[o + 1] = ptXYZ[o + 2];
    positions[o + 2] = ptXYZ[o + 1];
    const c = colorFromLabel(ptLabels[i]);
    colors[o] = c.r;
    colors[o + 1] = c.g;
    colors[o + 2] = c.b;
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
  scene.add(pointsObj);
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

function projectVehToImage(xyz, labels, cam, maxN = 120000) {
  const K = cam.K;
  const T = cam.T_c_v;
  const dist = cam.dist5 || [0, 0, 0, 0, 0];
  const w = cam.width;
  const h = cam.height;
  const fx = K[0], fy = K[4], cx = K[2], cy = K[5];
  const k1 = dist[0] || 0, k2 = dist[1] || 0, p1 = dist[2] || 0, p2 = dist[3] || 0, k3 = dist[4] || 0;
  const nAll = labels.length;
  const step = nAll > maxN ? Math.ceil(nAll / maxN) : 1;
  const out = [];
  for (let i = 0; i < nAll; i += step) {
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
    out.push({ u, v, z: zc, lab: labels[i], fx, fy });
  }
  out.sort((a, b) => b.z - a.z);
  return out;
}

/**
 * Occupancy: filled squares with image size ≈ voxel_m * fx / depth
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
    if (occCenters && occLabels && occLabels.length && activeVoxel) {
      const pts = projectVehToImage(occCenters, occLabels, cam, maxN);
      const vox = activeVoxel;
      // keep photo readable: capped square + translucent fill
      for (const p of pts) {
        let side = (vox * p.fx) / Math.max(p.z, 0.3);
        side = Math.min(Math.max(side, 1.0), 64);
        const s = Math.max(1, side * scale);
        ctx.globalAlpha = alpha;
        ctx.fillStyle = rgbCss(p.lab);
        ctx.fillRect(ox + p.u * scale - s * 0.5, oy + p.v * scale - s * 0.5, s, s);
      }
      ctx.globalAlpha = 1;
      n += pts.length;
    }
  }

  if (mode === "points" || mode === "both") {
    if (ptXYZ && ptLabels && ptLabels.length) {
      const pts = projectVehToImage(ptXYZ, ptLabels, cam, maxN);
      const minPx = Math.max(1, ptMinPx | 0);
      const px = Math.max(minPx, Math.round(scale));
      for (const p of pts) {
        ctx.globalAlpha = Math.min(0.95, alpha + 0.25);
        ctx.fillStyle = rgbCss(p.lab);
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
    img.src = `${SCENE_ROOT}/${frameDir}/${cam.file}`;
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
  frameDir = frameEntry.dir;
  const meta = await fetchJson(`${SCENE_ROOT}/${frameDir}/meta.json`);
  currentMeta = meta;
  classColors = meta.class_colors_rgb;
  classNames = meta.class_names || null;
  renderClassLegend();
  const title = `${index.clip}  ·  ts=${meta.timestamp}  ·  voxel=${meta.voxel}m  ·  occ=${meta.n_occ}`;
  el.titleMeta.textContent = title;
  el.titleMeta.title = title;

  const ijkBuf = await fetchBin(`${SCENE_ROOT}/${frameDir}/${meta.occupancy.ijk}`);
  const labBuf = await fetchBin(`${SCENE_ROOT}/${frameDir}/${meta.occupancy.labels}`);
  const ijkArr = new Int32Array(ijkBuf);
  const labArr = new Uint8Array(labBuf);
  exportedOcc = {
    voxel: meta.voxel,
    ijk: ijkArr,
    labels: labArr,
  };
  applyOccupancy(ijkArr, labArr, meta.voxel, "exported grid");

  clearPoints();
  ptXYZ = null;
  ptLabels = null;
  let ptsNote = "not exported";
  if (meta.points) {
    try {
      const xyzBuf = await fetchBin(`${SCENE_ROOT}/${frameDir}/${meta.points.xyz}`);
      const plabBuf = await fetchBin(`${SCENE_ROOT}/${frameDir}/${meta.points.labels}`);
      ptXYZ = new Float32Array(xyzBuf);
      ptLabels = new Uint8Array(plabBuf);
      buildPoints();
      ptsNote = ptLabels.length.toLocaleString();
      // data-driven axis help
      let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity, zmin = Infinity, zmax = -Infinity;
      for (let i = 0; i < ptXYZ.length; i += 3) {
        xmin = Math.min(xmin, ptXYZ[i]); xmax = Math.max(xmax, ptXYZ[i]);
        ymin = Math.min(ymin, ptXYZ[i + 1]); ymax = Math.max(ymax, ptXYZ[i + 1]);
        zmin = Math.min(zmin, ptXYZ[i + 2]); zmax = Math.max(zmax, ptXYZ[i + 2]);
      }
      document.getElementById("axisHelp").innerHTML = `
        Arrows follow stored numeric signs.<br/>
        <span style="color:var(--x)">+X</span> x∈[${xmin.toFixed(1)}, ${xmax.toFixed(1)}]<br/>
        <span style="color:var(--y)">+Y</span> y∈[${ymin.toFixed(1)}, ${ymax.toFixed(1)}]<br/>
        <span style="color:var(--z)">+Z</span> z∈[${zmin.toFixed(1)}, ${zmax.toFixed(1)}]<br/>
        Three map: (x,z,y) so XY ground stays horizontal.
      `;
    } catch (e) {
      ptsNote = `export broken: ${e}`;
    }
  }

  el.sceneInfo.innerHTML = `
    <div>occ voxels: <b>${occLabels ? occLabels.length.toLocaleString() : meta.n_occ.toLocaleString()}</b></div>
    <div>active voxel: <b>${activeVoxel}m</b> (exported ${meta.voxel}m)</div>
    <div>grid snap: floor((p-origin)/v)</div>
    <div>x: [${meta.x_range.join(", ")}]</div>
    <div>y: [${meta.y_range.join(", ")}]</div>
    <div>z: [${meta.z_range.join(", ")}]</div>
    <div>points: <b>${ptsNote}</b></div>
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

async function boot() {
  index = await fetchJson(`${SCENE_ROOT}/index.json`);
  el.frameSelect.innerHTML = "";
  for (const fr of index.frames) {
    const opt = document.createElement("option");
    opt.value = fr.timestamp;
    opt.textContent = `${fr.timestamp}  (occ=${fr.n_occ})`;
    el.frameSelect.appendChild(opt);
  }
  if (!index.frames.length) {
    setStatus("No frames in index.json");
    return;
  }
  await loadFrame(index.frames[0]);
}

el.frameSelect.addEventListener("change", async () => {
  const fr = index.frames.find((f) => f.timestamp === el.frameSelect.value);
  if (fr) await loadFrame(fr);
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
el.projRadius.addEventListener("input", refreshCamProjections);
if (el.projAlpha) el.projAlpha.addEventListener("input", refreshCamProjections);
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

let vidPollTimer = null;

function setVidStatus(msg) {
  if (el.vidStatus) el.vidStatus.textContent = msg || "";
}

async function refreshVideoList() {
  if (!el.vidList) return;
  try {
    const r = await fetch("/api/video/list");
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
    const r = await fetch("/api/video/status");
    const s = await r.json();
    const pct = s.n ? Math.round(100 * (s.frame || 0) / s.n) : Math.round(100 * (s.progress || 0));
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
