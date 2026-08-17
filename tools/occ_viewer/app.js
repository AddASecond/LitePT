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
  ptSize: document.getElementById("ptSize"),
  projMode: document.getElementById("projMode"),
  projRadius: document.getElementById("projRadius"),
  btnFit: document.getElementById("btnFit"),
  wrap: document.getElementById("canvas-wrap"),
};

let index = null;
let currentMeta = null;
let frameDir = null;
let occMesh = null;
let pointsObj = null;
let gridHelper = null;
let axesGroup = null;
let classColors = null;

/** Cached arrays in vehicle frame for rebuild / projection */
let occCenters = null; // Float32Array xyz xyz ...
let occLabels = null; // Uint8Array
let ptXYZ = null;
let ptLabels = null;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x0b0e13, 1);
el.wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 2500);
camera.position.set(-40, 50, -60);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.0, 50);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(-30, 80, -20);
scene.add(sun);

/**
 * Vehicle frame: +X right, +Y forward, +Z up.
 * Three.js: +X right, +Y up, +Z toward viewer / depth.
 * Mapping: Three(x,y,z) = Vehicle(x, z, y)
 * Ground (veh XY, z≈0) → Three XZ plane with Y up — matches default GridHelper.
 */
function vehToThree(x, y, z, out = new THREE.Vector3()) {
  return out.set(x, z, y);
}

gridHelper = new THREE.GridHelper(300, 60, 0x445066, 0x243041);
// DO NOT rotate — keep on Three XZ (= vehicle XY ground)
scene.add(gridHelper);

function makeAxisArrow(dirThree, color, length = 12) {
  const origin = new THREE.Vector3(0, 0.05, 0);
  const arrow = new THREE.ArrowHelper(dirThree.clone().normalize(), origin, length, color, 1.8, 1.0);
  return arrow;
}

function buildAxes() {
  if (axesGroup) {
    scene.remove(axesGroup);
    axesGroup = null;
  }
  axesGroup = new THREE.Group();
  // Vehicle +X → Three +X
  axesGroup.add(makeAxisArrow(new THREE.Vector3(1, 0, 0), 0xff4d4d, 15));
  // Vehicle +Y forward → Three +Z
  axesGroup.add(makeAxisArrow(new THREE.Vector3(0, 0, 1), 0x3dde6a, 15));
  // Vehicle +Z up → Three +Y
  axesGroup.add(makeAxisArrow(new THREE.Vector3(0, 1, 0), 0x4da3ff, 15));

  const loader = makeSprite;
  axesGroup.add(loader("+X right", new THREE.Vector3(16, 0.5, 0), "#ff4d4d"));
  axesGroup.add(loader("+Y forward", new THREE.Vector3(0, 0.5, 16), "#3dde6a"));
  axesGroup.add(loader("+Z up", new THREE.Vector3(0, 16, 0), "#4da3ff"));
  axesGroup.visible = el.togAxes.checked;
  scene.add(axesGroup);
}

function makeSprite(text, pos, color) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, 256, 64);
  ctx.fillStyle = color;
  ctx.font = "bold 28px sans-serif";
  ctx.fillText(text, 8, 40);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const spr = new THREE.Sprite(mat);
  spr.position.copy(pos);
  spr.scale.set(8, 2, 1);
  return spr;
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
  if (!classColors || !classColors[label]) return new THREE.Color(0.7, 0.7, 0.7);
  const c = classColors[label];
  return new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255);
}

function rgbCss(label) {
  if (!classColors || !classColors[label]) return "rgb(180,180,180)";
  const c = classColors[label];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function clearOcc() {
  if (occMesh) {
    scene.remove(occMesh);
    occMesh.geometry.dispose();
    occMesh.material.dispose();
    occMesh = null;
  }
}

function clearPoints() {
  if (pointsObj) {
    scene.remove(pointsObj);
    pointsObj.geometry.dispose();
    pointsObj.material.dispose();
    pointsObj = null;
  }
}

function buildOccMesh(centers, labels, voxel, gap) {
  clearOcc();
  const n = labels.length;
  if (!n) return;

  // gap=0 → exact voxel size so neighbors share faces (tight pack)
  const size = Math.max(1e-4, voxel * (1.0 - Math.max(0, gap)));
  const geo = new THREE.BoxGeometry(size, size, size);
  const mat = new THREE.MeshLambertMaterial({
    transparent: Number(el.occOpacity.value) < 0.999,
    opacity: Number(el.occOpacity.value),
    depthWrite: true,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
  const mesh = new THREE.InstancedMesh(geo, mat, n);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  for (let i = 0; i < n; i++) {
    const o = i * 3;
    const vx = centers[o];
    const vy = centers[o + 1];
    const vz = centers[o + 2];
    vehToThree(vx, vy, vz, dummy.position);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    color.copy(colorFromLabel(labels[i]));
    mesh.setColorAt(i, color);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.frustumCulled = false;
  occMesh = mesh;
  occMesh.visible = el.togOcc.checked;
  scene.add(mesh);
}

function buildPoints(xyz, labels, size) {
  clearPoints();
  const n = labels.length;
  if (!n) return;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const o = i * 3;
    // vehicle → three
    positions[o] = xyz[o];
    positions[o + 1] = xyz[o + 2];
    positions[o + 2] = xyz[o + 1];
    const c = colorFromLabel(labels[i]);
    colors[o] = c.r;
    colors[o + 1] = c.g;
    colors[o + 2] = c.b;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.PointsMaterial({
    size: Number(size),
    vertexColors: true,
    sizeAttenuation: true,
  });
  pointsObj = new THREE.Points(geo, mat);
  pointsObj.visible = el.togPts.checked;
  scene.add(pointsObj);
}

function fitCamera(meta) {
  const y0 = meta.y_range[0];
  const y1 = meta.y_range[1];
  const midY = 0.5 * (y0 + y1);
  // Look at a point ahead on the ground (veh y = midY → three z)
  controls.target.set(0, 1.2, Math.max(10, Math.min(150, midY)));
  camera.position.set(-55, 45, midY - 70);
  controls.update();
}

/** Project vehicle-frame points with K / T_c_v (row-major 4x4). */
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
    // p_c = R * p_v + t
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
    if (u < 0 || v < 0 || u >= w || v >= h) continue;
    out.push({ u, v, z: zc, lab: labels[i] });
  }
  out.sort((a, b) => b.z - a.z);
  return out;
}

function drawProjectionOnCanvas(canvas, img, cam, mode, radius) {
  const ctx = canvas.getContext("2d");
  const cw = canvas.width;
  const ch = canvas.height;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, cw, ch);

  const scale = Math.min(cw / cam.width, ch / cam.height);
  const dw = cam.width * scale;
  const dh = cam.height * scale;
  const ox = (cw - dw) / 2;
  const oy = (ch - dh) / 2;
  if (img && img.complete) {
    ctx.drawImage(img, ox, oy, dw, dh);
  }

  const drawSet = (xyz, labels, alpha) => {
    if (!xyz || !labels) return;
    const pts = projectVehToImage(xyz, labels, cam);
    for (const p of pts) {
      ctx.beginPath();
      ctx.fillStyle = rgbCss(p.lab);
      ctx.globalAlpha = alpha;
      ctx.arc(ox + p.u * scale, oy + p.v * scale, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  };

  if (mode === "occ" || mode === "both") {
    drawSet(occCenters, occLabels, 0.9);
  }
  if (mode === "points" || mode === "both") {
    drawSet(ptXYZ, ptLabels, 0.75);
  }
}

function refreshCamProjections() {
  if (!currentMeta) return;
  const mode = el.projMode.value;
  const radius = Number(el.projRadius.value);
  const cards = el.cams.querySelectorAll(".cam-card");
  cards.forEach((card) => {
    const canvas = card.querySelector("canvas");
    const img = card.querySelector("img.base");
    const cam = card._cam;
    if (!canvas || !cam) return;
    drawProjectionOnCanvas(canvas, img, cam, mode, radius);
  });
}

function renderCams(meta) {
  el.cams.innerHTML = "";
  for (const cam of meta.cameras || []) {
    const card = document.createElement("div");
    card.className = "cam-card";
    card._cam = cam;

    const name = document.createElement("div");
    name.className = "name";
    name.innerHTML = `<span>${cam.name}</span><span>${cam.width}×${cam.height}</span>`;

    const stage = document.createElement("div");
    stage.className = "stage";
    const img = document.createElement("img");
    img.className = "base";
    img.crossOrigin = "anonymous";
    img.style.display = "none";
    img.src = `${SCENE_ROOT}/${frameDir}/${cam.file}`;
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 360;
    stage.appendChild(img);
    stage.appendChild(canvas);

    img.onload = () => {
      drawProjectionOnCanvas(canvas, img, cam, el.projMode.value, Number(el.projRadius.value));
    };

    card.appendChild(name);
    card.appendChild(stage);
    card.addEventListener("click", () => {
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
  el.titleMeta.textContent = `${index.clip}  ·  ts=${meta.timestamp}  ·  voxel=${meta.voxel}m  ·  occ=${meta.n_occ}`;
  el.sceneInfo.innerHTML = `
    <div>occ voxels: <b>${meta.n_occ.toLocaleString()}</b></div>
    <div>y forward: [${meta.y_range.join(", ")}] m</div>
    <div>x right: [${meta.x_range.join(", ")}] m</div>
    <div>z up: [${meta.z_range.join(", ")}] m</div>
    <div>cameras: ${meta.cameras.length}</div>
    <div>points: ${meta.points ? meta.points.n.toLocaleString() : "not exported"}</div>
  `;

  const occBuf = await fetchBin(`${SCENE_ROOT}/${frameDir}/${meta.occupancy.centers}`);
  const labBuf = await fetchBin(`${SCENE_ROOT}/${frameDir}/${meta.occupancy.labels}`);
  occCenters = new Float32Array(occBuf);
  occLabels = new Uint8Array(labBuf);
  buildOccMesh(occCenters, occLabels, meta.voxel, Number(el.occGap.value));

  clearPoints();
  ptXYZ = null;
  ptLabels = null;
  if (meta.points) {
    const xyzBuf = await fetchBin(`${SCENE_ROOT}/${frameDir}/${meta.points.xyz}`);
    const plabBuf = await fetchBin(`${SCENE_ROOT}/${frameDir}/${meta.points.labels}`);
    ptXYZ = new Float32Array(xyzBuf);
    ptLabels = new Uint8Array(plabBuf);
    buildPoints(ptXYZ, ptLabels, el.ptSize.value);
  } else if (el.projMode.value === "points" || el.projMode.value === "both") {
    setStatus("points not in export — use Occupancy projection or re-export with --export-points");
  }

  renderCams(meta);
  fitCamera(meta);
  setStatus(`ready · ${meta.n_occ} voxels · gap=${el.occGap.value}`);
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
  const ts = el.frameSelect.value;
  const fr = index.frames.find((f) => f.timestamp === ts);
  if (fr) await loadFrame(fr);
});

el.togOcc.addEventListener("change", () => {
  if (occMesh) occMesh.visible = el.togOcc.checked;
});
el.togPts.addEventListener("change", () => {
  if (pointsObj) pointsObj.visible = el.togPts.checked;
});
el.togGrid.addEventListener("change", () => {
  gridHelper.visible = el.togGrid.checked;
});
el.togAxes.addEventListener("change", () => {
  if (axesGroup) axesGroup.visible = el.togAxes.checked;
});
el.occOpacity.addEventListener("input", () => {
  if (occMesh) {
    const op = Number(el.occOpacity.value);
    occMesh.material.opacity = op;
    occMesh.material.transparent = op < 0.999;
    occMesh.material.needsUpdate = true;
  }
});
el.occGap.addEventListener("input", () => {
  if (!occCenters || !currentMeta) return;
  buildOccMesh(occCenters, occLabels, currentMeta.voxel, Number(el.occGap.value));
  setStatus(`gap=${el.occGap.value} · size=${(currentMeta.voxel * (1 - Number(el.occGap.value))).toFixed(4)}m`);
});
el.ptSize.addEventListener("input", () => {
  if (pointsObj) {
    pointsObj.material.size = Number(el.ptSize.value);
    pointsObj.material.needsUpdate = true;
  }
});
el.projMode.addEventListener("change", refreshCamProjections);
el.projRadius.addEventListener("input", refreshCamProjections);
el.btnFit.addEventListener("click", () => {
  if (currentMeta) fitCamera(currentMeta);
});

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
