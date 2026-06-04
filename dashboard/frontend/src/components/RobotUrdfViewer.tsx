import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import URDFLoader, { URDFRobot } from "urdf-loader";
import { ArmSide, VRStatus } from "../api";
import { Badge, Card, Toggle } from "./ui";

const URDF_URL = "/robot_assets/xlerobot/xlerobot_front.urdf";
const JOINTS = [
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_roll",
  "gripper",
] as const;
const SIDES: ArmSide[] = ["left", "right"];

const XLEROBOT_ARM_JOINTS: Record<ArmSide, Record<(typeof JOINTS)[number], string>> = {
  right: {
    shoulder_pan: "Rotation",
    shoulder_lift: "Pitch",
    elbow_flex: "Elbow",
    wrist_flex: "Wrist_Pitch",
    wrist_roll: "Wrist_Roll",
    gripper: "Jaw",
  },
  left: {
    shoulder_pan: "Rotation_2",
    shoulder_lift: "Pitch_2",
    elbow_flex: "Elbow_2",
    wrist_flex: "Wrist_Pitch_2",
    wrist_roll: "Wrist_Roll_2",
    gripper: "Jaw_2",
  },
};

// Viewer frame:
//   X = robot right/left  (ROS -Y; keeps the frame right-handed)
//   Y = up                (ROS +Z)
//   Z = view depth        (ROS -X, so robot forward faces the camera)
const ROS_TO_VIEW = new THREE.Quaternion().setFromRotationMatrix(
  new THREE.Matrix4().makeBasis(
    new THREE.Vector3(0, 0, -1), // ROS +X
    new THREE.Vector3(-1, 0, 0), // ROS +Y
    new THREE.Vector3(0, 1, 0),  // ROS +Z
  ),
);
type PoseKind = "present" | "target";

function degToRad(value: number): number {
  return value * Math.PI / 180.0;
}

function so101MotorDegToXLeRobotRad(
  status: VRStatus | undefined,
  side: ArmSide,
  joint: string,
  value: number | undefined,
): number | null {
  if (value === undefined || value === null || !Number.isFinite(value)) return null;

  // xlerobot_front.urdf uses the simulation/old SO101 arm convention for the
  // lift/elbow pair. Live robot readings are LeRobot calibrated motor degrees.
  if (joint === "shoulder_lift") return degToRad(90.0 - value);
  if (joint === "elbow_flex") return degToRad(value + 90.0);
  if (joint === "shoulder_pan") {
    const homeKey = `${side}_arm_shoulder_pan`;
    const homePan = status?.arms?.[side]?.home?.joints?.[homeKey];
    const straightReference = Number.isFinite(homePan) ? Number(homePan) : 0.0;
    return degToRad(value - straightReference);
  }
  if (joint === "wrist_roll") return degToRad(90.0 - value);

  // XLeRobot Jaw is a radian hinge. The real gripper is reported as 0..100.
  if (joint === "gripper") return Math.max(0.0, Math.min(100.0, value)) / 100.0 * 1.7;

  return degToRad(value);
}

function jointValue(status: VRStatus | undefined, side: ArmSide, joint: string, kind: PoseKind): number | null {
  const key = `${side}_arm_${joint}`;
  if (kind === "target") {
    const target = status?.arms?.[side]?.joint_target?.[key];
    const rad = so101MotorDegToXLeRobotRad(status, side, joint, target);
    if (rad !== null) return rad;
  }
  const present = so101MotorDegToXLeRobotRad(status, side, joint, status?.joint_present?.[key]);
  if (present !== null) return present;
  return so101MotorDegToXLeRobotRad(status, side, joint, status?.arms?.[side]?.home?.joints?.[key]);
}

function applyRobotPose(robot: URDFRobot | null, status: VRStatus | undefined, kind: PoseKind) {
  if (!robot) return;
  for (const side of SIDES) {
    for (const joint of JOINTS) {
      const value = jointValue(status, side, joint, kind);
      const xleJoint = XLEROBOT_ARM_JOINTS[side][joint];
      if (value !== null && robot.joints[xleJoint]) {
        robot.setJointValue(xleJoint, value);
      }
    }
  }
}

function hasTargetPose(status: VRStatus | undefined, side: ArmSide): boolean {
  const arm = status?.arms?.[side];
  if (!arm) return false;
  return JOINTS.some((joint) => arm.joint_target?.[`${side}_arm_${joint}`] !== undefined);
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
    const material = mesh.material;
    if (Array.isArray(material)) {
      material.forEach((m) => m.dispose());
    } else if (material) {
      material.dispose();
    }
  });
}

function prepareRobot(robot: URDFRobot, kind: PoseKind) {
  robot.quaternion.copy(ROS_TO_VIEW);
  robot.position.set(0.0, 0.0, 0.0);
  robot.updateMatrixWorld(true);
  robot.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    if (kind === "target") {
      mesh.material = new THREE.MeshBasicMaterial({
        color: 0x39a9ff,
        transparent: true,
        opacity: 0.55,
        wireframe: true,
        depthWrite: false,
      });
      mesh.renderOrder = 2;
    } else {
      const material = mesh.material;
      const mats = Array.isArray(material) ? material : [material];
      mesh.material = mats.map((m) => {
        if (m instanceof THREE.MeshStandardMaterial) {
          const clone = m.clone();
          clone.roughness = 0.7;
          clone.metalness = 0.05;
          return clone;
        }
        return new THREE.MeshStandardMaterial({
          color: 0xd7dadd,
          roughness: 0.7,
          metalness: 0.05,
        });
      });
      if (!Array.isArray(material)) {
        mesh.material = (mesh.material as THREE.Material[])[0];
      }
      mesh.renderOrder = 1;
    }
  });
}

function sceneBackgroundColor() {
  return "#101114";
}

export default function RobotUrdfViewer({ status }: { status: VRStatus | undefined }) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const presentRef = useRef<URDFRobot | null>(null);
  const targetRef = useRef<URDFRobot | null>(null);
  const [showTarget, setShowTarget] = useState(false);
  const [loadState, setLoadState] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string>("");

  const targetAvailable = useMemo(
    () => SIDES.some((side) => hasTargetPose(status, side)),
    [status],
  );

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let disposed = false;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(sceneBackgroundColor());

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 6);
    camera.position.set(0.0, 1.0, 1.55);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.autoClear = false;
    renderer.setClearColor(scene.background as THREE.Color, 1);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0.0, 0.72, -0.06);
    controls.minDistance = 0.45;
    controls.maxDistance = 2.6;

    const hemi = new THREE.HemisphereLight(0xffffff, 0x2a2d34, 2.2);
    scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 2.6);
    key.position.set(0.5, 1.2, 0.45);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    scene.add(key);

    const fill = new THREE.DirectionalLight(0x9fc7ff, 0.9);
    fill.position.set(-0.5, 0.5, -0.45);
    scene.add(fill);

    const grid = new THREE.GridHelper(1.6, 24, 0x5f666f, 0x30343a);
    grid.position.y = -0.002;
    scene.add(grid);

    const axes = new THREE.AxesHelper(0.12);
    axes.position.set(0.02, 0.004, 0.02);
    scene.add(axes);

    const loader = new URDFLoader();
    loader.parseCollision = false;
    loader.parseVisual = true;

    Promise.all([
      loader.loadAsync(URDF_URL),
      loader.loadAsync(URDF_URL),
    ])
      .then(([targetRobot, presentRobot]) => {
        if (disposed) {
          [targetRobot, presentRobot].forEach(disposeObject);
          return;
        }
        prepareRobot(targetRobot, "target");
        prepareRobot(presentRobot, "present");
        targetRobot.visible = false;
        targetRef.current = targetRobot;
        presentRef.current = presentRobot;
        scene.add(targetRobot);
        scene.add(presentRobot);
        setLoadState("ready");
      })
      .catch((e) => {
        if (!disposed) {
          setLoadState("error");
          setError(e instanceof Error ? e.message : String(e));
        }
      });

    const resize = () => {
      const rect = mount.getBoundingClientRect();
      const width = Math.max(1, Math.floor(rect.width));
      const height = Math.max(1, Math.floor(rect.height));
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);
    resize();

    const animate = () => {
      if (disposed) return;
      controls.update();
      renderer.clear(true, true, true);
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    };
    animate();

    return () => {
      disposed = true;
      observer.disconnect();
      controls.dispose();
      if (presentRef.current) disposeObject(presentRef.current);
      if (targetRef.current) disposeObject(targetRef.current);
      presentRef.current = null;
      targetRef.current = null;
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  useEffect(() => {
    applyRobotPose(presentRef.current, status, "present");
    applyRobotPose(targetRef.current, status, "target");
  }, [status]);

  useEffect(() => {
    if (targetRef.current) targetRef.current.visible = targetAvailable && showTarget;
  }, [showTarget, targetAvailable, loadState]);

  return (
    <Card>
      <div className="space-y-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-sm font-semibold">OpenPIBot 3D View</h2>
            <Badge tone={loadState === "ready" ? "success" : loadState === "error" ? "danger" : "neutral"}>
              {loadState}
            </Badge>
            {SIDES.map((side) => {
              const connected = !!status?.arms?.[side]?.connected;
              const active = status?.active_arm === side || !!status?.dual_mode;
              return (
                <Badge key={side} tone={connected ? active ? "danger" : "success" : "neutral"}>
                  {side} {connected ? active ? "active" : "connected" : "offline"}
                </Badge>
              );
            })}
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>target overlay</span>
            <Toggle
              checked={showTarget}
              onCheckedChange={setShowTarget}
              disabled={!targetAvailable && loadState === "ready"}
            />
          </div>
        </div>

        <div
          ref={mountRef}
          className="h-[420px] min-h-80 w-full overflow-hidden rounded-md border border-border bg-[#101114]"
        />

        {loadState === "error" ? (
          <p className="text-xs text-danger">URDF load failed: {error}</p>
        ) : (
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="warning">solid present</Badge>
            <Badge tone={showTarget && targetAvailable ? "info" : "neutral"}>
              {showTarget && targetAvailable ? "wireframe targets" : "targets hidden"}
            </Badge>
            <span className="mono text-xs text-muted-foreground">
              XLeRobot front URDF · {targetAvailable ? "target stream available" : "waiting for target"}
            </span>
          </div>
        )}
      </div>
    </Card>
  );
}
