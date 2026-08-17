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
  occOpacity: document.getElementById("occOpacity"),
  occGap: document.getElementById("occGap"),
  ptSize: document.getElementById("ptSize"),
  btnFit: document.getElementById("btnFit"),
  wrap: document.getElementById("canvas-wrap"),
};

let index = null;
let currentMeta = null;
let occMesh = null;
let pointsObj = null;
let gridHelper = null;
let classColors = null;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x0b0e13, 1);
el.wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 2000);
camera.position.set(-25, 35, -40);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 40);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 0.85);
sun.position.set(-40, 80, 20);
scene.add(sun);

gridHelper = new THREE.GridHelper(200, 40, 0x334155, 0x1f2937);
// Three grid is XZ; our vehicle frame is X-right, Y-forward, Z-up → rotate so +Y is depth on ground
gridHelper.rotation.x = Math.PI / 2;
scene.add(gridHelper);

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

  const size = Math.max(1e-4, voxel * (1.0 - gap));
  const geo = new THREE.BoxGeometry(size, size, size);
  const mat = new THREE.MeshLambertMaterial({
    transparent: true,
    opacity: Number(el.occOpacity.value),
    depthWrite: true,
  });
  const mesh = new THREE.InstancedMesh(geo, mat, n);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  for (let i = 0; i < n; i++) {
    const o = i * 3;
    // vehicle: x right, y forward, z up → Three: x, y=up, z=-forward? 
    // Map directly: Three X=veh X, Three Y=veh Z, Three Z=veh Y (forward into screen/depth)
    dummy.position.set(centers[o], centers[o + 2], centers[o + 1]);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    color.copy(colorFromLabel(labels[i]));
    mesh.setColorAt(i, color);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.frustumCulled = false;
  occMesh = mesh;
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
  controls.target.set(0, 1.5, Math.max(20, Math.min(120, midY)));
  camera.position.set(-45, 55, midY - 80);
  controls.update();
}

function renderCams(meta, frameDir) {
  el.cams.innerHTML = "";
  for (const cam of meta.cameras || []) {
    const card = document.createElement("div");
    card.className = "cam-card";
    const img = document.createElement("img");
    img.src = `${SCENE_ROOT}/${frameDir}/${cam.file}`;
    img.alt = cam.name;
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = `${cam.name}  ${cam.width}×${cam.height}`;
    card.appendChild(img);
    card.appendChild(name);
    card.addEventListener("click", () => {
      [...el.cams.children].forEach((c) => c.classList.remove("active"));
      card.classList.add("active");
    });
    el.cams.appendChild(card);
  }
}

async function loadFrame(frameEntry) {
  setStatus("Loading frame…");
  const frameDir = frameEntry.dir;
  const meta = await fetchJson(`${SCENE_ROOT}/${frameDir}/meta.json`);
  currentMeta = meta;
  classColors = meta.class_colors_rgb;
  el.titleMeta.textContent = `${index.clip}  ·  ts=${meta.timestamp}  ·  voxel=${meta.voxel}m  ·  occ=${meta.n_occ}`;
  el.sceneInfo.innerHTML = `
    <div>occ voxels: <b>${meta.n_occ.toLocaleString()}</b></div>
    <div>range y: [${meta.y_range.join(", ")}] m</div>
    <div>range x: [${meta.x_range.join(", ")}] m</div>
    <div>range z: [${meta.z_range.join(", ")}] m</div>
    <div>cameras: ${meta.cameras.length}</div>
    <div>points exported: ${meta.points ? meta.points.n.toLocaleString() : "no"}</div>
  `;

  const occBuf = await fetchBin(`${SCENE_ROOT}/${frameDir}/${meta.occupancy.centers}`);
  const labBuf = await fetchBin(`${SCENE_ROOT}/${frameDir}/${meta.occupancy.labels}`);
  const centers = new Float32Array(occBuf);
  const labels = new Uint8Array(labBuf);
  buildOccMesh(centers, labels, meta.voxel, Number(el.occGap.value));
  if (occMesh) occMesh.visible = el.togOcc.checked;

  clearPoints();
  if (meta.points) {
    const xyzBuf = await fetchBin(`${SCENE_ROOT}/${frameDir}/${meta.points.xyz}`);
    const plabBuf = await fetchBin(`${SCENE_ROOT}/${frameDir}/${meta.points.labels}`);
    buildPoints(new Float32Array(xyzBuf), new Uint8Array(plabBuf), el.ptSize.value);
  }

  renderCams(meta, frameDir);
  fitCamera(meta);
  setStatus(`ready · ${meta.n_occ} voxels`);
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
el.occOpacity.addEventListener("input", () => {
  if (occMesh) {
    occMesh.material.opacity = Number(el.occOpacity.value);
    occMesh.material.needsUpdate = true;
  }
});
el.occGap.addEventListener("change", async () => {
  if (!currentMeta) return;
  const frameDir = `frames/${currentMeta.timestamp}`;
  const occBuf = await fetchBin(`${SCENE_ROOT}/${frameDir}/${currentMeta.occupancy.centers}`);
  const labBuf = await fetchBin(`${SCENE_ROOT}/${frameDir}/${currentMeta.occupancy.labels}`);
  buildOccMesh(
    new Float32Array(occBuf),
    new Uint8Array(labBuf),
    currentMeta.voxel,
    Number(el.occGap.value)
  );
  if (occMesh) occMesh.visible = el.togOcc.checked;
});
el.ptSize.addEventListener("input", () => {
  if (pointsObj) {
    pointsObj.material.size = Number(el.ptSize.value);
    pointsObj.material.needsUpdate = true;
  }
});
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
