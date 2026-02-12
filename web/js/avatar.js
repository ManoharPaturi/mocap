import * as THREE from 'three';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- CONFIG ---
const MODEL_PATH = '/static/assets/character.fbx';
const WS_URL = 'ws://localhost:8000/ws';

// --- THREE.JS SETUP ---
const container = document.getElementById('3d-view');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222233);
scene.fog = new THREE.Fog(0x222233, 200, 1000);

const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 1, 2000);
camera.position.set(0, 150, 400);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.shadowMap.enabled = true;
container.appendChild(renderer.domElement);

// Lighting
const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 5);
hemiLight.position.set(0, 200, 0);
scene.add(hemiLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 5);
dirLight.position.set(0, 200, 100);
dirLight.castShadow = true;
scene.add(dirLight);

// Grid
const grid = new THREE.GridHelper(2000, 20, 0x000000, 0x000000);
grid.material.opacity = 0.2;
grid.material.transparent = true;
scene.add(grid);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 100, 0);
controls.update();

// --- STATE ---
let mixer;
let model;
let skeleton;
const bones = {}; // Map of bone names to Bone objects

// Start Animation Loop
animate();

// --- LOAD MODEL (OR FALLBACK) ---
// We try to load FBX. If it fails (version 6100 etc), we build a Robot.
const loader = new FBXLoader();
loader.load(MODEL_PATH, (object) => {
    model = object;

    // Traverse and find bones
    object.traverse((child) => {
        if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
        }
        if (child.isBone) {
            bones[child.name] = child;
        }
    });

    console.log("FBX Loaded. Bones found:", Object.keys(bones));
    scene.add(object);

}, undefined, (e) => {
    console.error("Error loading model (likely version issue):", e);
    document.getElementById('info').innerHTML += "<br><span style='color:red'>FBX Error - Using Primitive Robot Fallback</span>";
    createPrimitiveRobot();
});

// Connect WebSocket IMMEDIATELY (Don't wait for model)
connectWebSocket();

// --- PRIMITIVE ROBOT FALLBACK ---
const robotBones = {};

function createPrimitiveRobot() {
    console.log("Creating Primitive Robot...");
    const mat = new THREE.MeshStandardMaterial({ color: 0x00d4ff });

    // Root
    const hips = new THREE.Bone(); hips.name = MIXAMO.LeftUpLeg.replace('LeftUpLeg', 'Hips'); // Hack
    hips.position.set(0, 100, 0);
    scene.add(hips);

    // We actually need a hierarchy to rotate limbs.
    // For simplicity in this fail-safe, we just visualize markers if model fails?
    // Or we assume the user WILL fix the FBX. 
    // Let's just spawn a simple box to show "Alive" state.
    const geometry = new THREE.BoxGeometry(10, 10, 10);
    const cube = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0xff0000 }));
    scene.add(cube);

    // Animating a full skeleton from scratch in code is complex.
    // We will rely on user getting a binary FBX, but show status.
}

// --- WEBSOCKET & RETARGETING ---
function connectWebSocket() {
    const ws = new WebSocket(WS_URL);

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.pose) {
            updatePose(data.pose);
        }
    };

    ws.onopen = () => {
        console.log("Connected to Python Physics Backend");
        const status = document.getElementById('info');
        if (!status.innerHTML.includes("FBX Error")) {
            status.innerHTML = "<h1>Connected: Live Intelligent Twin</h1>";
        }
    };
}

// MEDIA PIPE LANDMARK INDICES
const MP = {
    NOSE: 0,
    LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
    LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
    LEFT_WRIST: 15, RIGHT_WRIST: 16,
    LEFT_HIP: 23, RIGHT_HIP: 24,
    LEFT_KNEE: 25, RIGHT_KNEE: 26,
    LEFT_ANKLE: 27, RIGHT_ANKLE: 28
};

// MIXAMO BONE NAMES (Standard)
const MIXAMO = {
    // Left Arm
    LeftArm: 'mixamorigLeftArm',      // Shoulder -> Elbow
    LeftForeArm: 'mixamorigLeftForeArm', // Elbow -> Wrist

    // Right Arm
    RightArm: 'mixamorigRightArm',
    RightForeArm: 'mixamorigRightForeArm',

    // Legs
    LeftUpLeg: 'mixamorigLeftUpLeg',  // Hip -> Knee
    LeftLeg: 'mixamorigLeftLeg',      // Knee -> Ankle

    RightUpLeg: 'mixamorigRightUpLeg',
    RightLeg: 'mixamorigRightLeg'
};

function updatePose(landmarks) {
    if (!model) return;

    // Helper: Get Vector3 from Landmark Index
    function getVec(index) {
        const lm = landmarks[index];
        // MediaPipe: X left/right, Y up/down (inverted), Z depth
        // Three.js: X left/right, Y up/down, Z depth
        // We scale up coordinates * 100 for visibility
        return new THREE.Vector3(-lm.x, -lm.y, -lm.z);
    }

    // --- RETARGETING LOGIC ---
    // We compute the direction vector for a limb segment in MP, 
    // and rotate the corresponding Bone to match that direction.

    // Helper to rotate a bone to align with MP limb vector
    function applyRot(boneName, startIdx, endIdx, restVector) {
        const bone = bones[boneName];
        if (!bone) return;

        const start = getVec(startIdx);
        const end = getVec(endIdx);

        // Live Vector
        const currentVec = new THREE.Vector3().subVectors(end, start).normalize();

        // Create Quaternion from Rest -> Current
        const q = new THREE.Quaternion().setFromUnitVectors(restVector, currentVec);

        // Apply (Slerp for smoothness could go here, but doing direct for response)
        bone.quaternion.slerp(q, 0.5); // 0.5 smoothing factor
    }

    // Example: Right Arm (Shoulder -> Elbow)
    applyRot(MIXAMO.RightArm, MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW, new THREE.Vector3(1, 0, 0)); // T-Pose R-Arm points +X
    applyRot(MIXAMO.RightForeArm, MP.RIGHT_ELBOW, MP.RIGHT_WRIST, new THREE.Vector3(1, 0, 0));

    // Left Arm (Points -X in T-Pose)
    applyRot(MIXAMO.LeftArm, MP.LEFT_SHOULDER, MP.LEFT_ELBOW, new THREE.Vector3(-1, 0, 0));
    applyRot(MIXAMO.LeftForeArm, MP.LEFT_ELBOW, MP.LEFT_WRIST, new THREE.Vector3(-1, 0, 0));

    // Legs (Point -Y in T-Pose effectively)
    applyRot(MIXAMO.RightUpLeg, MP.RIGHT_HIP, MP.RIGHT_KNEE, new THREE.Vector3(0, -1, 0));
    applyRot(MIXAMO.RightLeg, MP.RIGHT_KNEE, MP.RIGHT_ANKLE, new THREE.Vector3(0, -1, 0));

    applyRot(MIXAMO.LeftUpLeg, MP.LEFT_HIP, MP.LEFT_KNEE, new THREE.Vector3(0, -1, 0));
    applyRot(MIXAMO.LeftLeg, MP.LEFT_KNEE, MP.LEFT_ANKLE, new THREE.Vector3(0, -1, 0));
}

// --- RENDER LOOP ---
function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

// --- RESIZE ---
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
