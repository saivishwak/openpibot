/** Thin API client. SWR is used in components for cache + revalidation. */

export type CheckStatus = "ok" | "warn" | "fail" | "info";
export interface DoctorCheck { name: string; status: CheckStatus; detail: string }

export interface RobotProfile {
  id: string;
  name?: string | null;
  active: boolean;
  port_left_base?: string | null;
  port_right_head?: string | null;
  left_arm_id?: string | null;
  right_arm_id?: string | null;
  home_pose?: Record<string, number>;
}

export interface RobotsResponse {
  active_robot: string;
  robots: RobotProfile[];
}

export interface Job {
  id: string;
  command: string[];
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled" | string;
  returncode: number | null;
  started_at: number | null;
  finished_at: number | null;
  log: string[];
}

export interface CameraSpec {
  name: string;
  path: string;
  width: number; height: number; fps: number; fourcc: string;
  role: string | null;
  by_path: string | null;
  card: string;
  available?: boolean;
  capture?: boolean | null;
}

export type ArmSide = "left" | "right";

export interface VRControllerPose {
  /** present, in metres (lerobot-frame XYZ) */
  position: [number, number, number] | null;
  /** quaternion x,y,z,w from VR — null if no goal received yet */
  rotation: [number, number, number, number] | null;
  trigger: boolean;
  thumbstick: { x: number; y: number } | null;
  /** ms since last goal arrived from the controller */
  age_ms: number | null;
  /** "idle" | "position" | "reset" */
  mode: string;
}

export interface RobotVerificationSample {
  label: string;
  robot_start: [number, number, number];
  robot_end: [number, number, number];
  robot_delta: [number, number, number];
  vr_start: [number, number, number];
  vr_end: [number, number, number];
  vr_delta: [number, number, number];
  robot_motion_m: number;
  vr_motion_m: number;
  captured_at: string;
}

export interface RobotVerificationSampleResidual {
  index: number;
  label: string;
  residual_cm: number;
  direction_error_deg: number | null;
  robot_motion_cm: number;
  vr_motion_cm: number;
  target_robot_delta?: [number, number, number];
  predicted_robot_delta: [number, number, number];
  error_vector_cm?: [number, number, number];
}

export interface RobotVerificationLive {
  ready: boolean;
  state: string;
  message: string;
  sample_label: string;
  controller_age_ms: number | null;
  target_robot_delta: [number, number, number] | null;
  predicted_robot_delta: [number, number, number] | null;
  vr_delta: [number, number, number] | null;
  target_motion_m: number | null;
  predicted_motion_m: number | null;
  vr_motion_m: number | null;
  direction_error_deg: number | null;
  direction_match: number | null;
  magnitude_ratio: number | null;
  position_error_cm: number | null;
  scale_estimate: number;
  preview_source?: "verified" | "provisional_lstsq" | "stage1_scaled" | string;
}

export interface CalibrationProfileSummary {
  name: string;
  left_saved: boolean;
  right_saved: boolean;
  left_robot_verified: boolean;
  right_robot_verified: boolean;
  updated_at: string | null;
}

export interface CalibrationProfilesStatus {
  active_profile: string;
  profiles: CalibrationProfileSummary[];
}

/** Per-arm state in the new bimanual view. */
export interface VRArmState {
  connected: boolean;
  /** When false, motors are limp — user is hand-posing this arm. Drive loop
   *  skips arms with torque off. */
  torque_enabled: boolean;
  calibrated: boolean;
  joint_target: Record<string, number>;
  controller: VRControllerPose;
  /** Calibration diagnostics — populated after the user squeezes grip on this
   *  arm's controller. Shows what the backend thinks the mapping is doing. */
  calibration: {
    /** Robot EE position (m, robot base frame) at the moment of RESET. */
    anchor_ee_pos: [number, number, number];
    /** Cumulative offset from anchor in robot base frame (m, unclamped). */
    offset_robot: [number, number, number];
    /** Final EE target (m, robot base frame) — anchor + offset, clamped. */
    target_ee_pos: [number, number, number];
    /** Yaw (deg) of the active session VR→robot frame relative to default. */
    session_yaw_deg: number;
    /** Guided 3-vector + optional wrist pitch/roll calibration wizard state. */
    wizard_state:
      | "idle"
      | "awaiting_anchor_fwd"  | "motioning_fwd"
      | "awaiting_anchor_up"   | "motioning_up"
      | "awaiting_anchor_left" | "motioning_left"
      | "awaiting_anchor_wrist_verify" | "motioning_wrist_verify"
      | "awaiting_anchor_wrist_pitch" | "motioning_wrist_pitch"
      | "awaiting_anchor_wrist_roll" | "motioning_wrist_roll";
    /** Live motion magnitude (m) accumulated during the current motion-capture. */
    wizard_motion_m: number;
    wizard_target_m: number;
    wizard_min_m: number;
    /** Most-recently-captured forward / up / left motion magnitudes (m). */
    wizard_last_fwd_m: number;
    wizard_last_up_m: number;
    wizard_last_left_m: number;
    wizard_fwd_captured: boolean;
    wizard_up_captured: boolean;
    wizard_left_captured: boolean;
    /** Step 4: live wrist rotation since anchor (degrees). */
    wizard_wrist_verify_deg: number;
    wizard_wrist_pitch_verify_deg: number;
    wizard_wrist_roll_verify_deg: number;
    wizard_wrist_verify_target_deg: number;
    wizard_wrist_verify_min_deg: number;
    wizard_wrist_captured: boolean;
    wizard_wrist_pitch_captured: boolean;
    wizard_wrist_roll_captured: boolean;
    /** Empirical raw controller-anchor-local wrist axes, if captured. */
    wrist_pitch_canonical: [number, number, number] | null;
    wrist_roll_canonical: [number, number, number] | null;
    /** Result of the lateral-check step. True = matrix mirroring detected,
     *  invert_lateral was auto-flipped on by the wizard. */
    invert_lateral: boolean;
    /** "good" = captured motion vectors well-separated, matrix is robust.
     *  "poor" = vectors too parallel (cos > 0.6); re-run wizard for better
     *  results. */
    confidence: "good" | "poor";
    /** Whether the most-recent calibration has been written to
     *  `config/vr_calibration.yaml` and reload on next startup. */
    persisted: {
      saved: boolean;
      calibration_mode?: string | null;
      calibrated_at: string | null;
      forward_motion_m: number;
      up_motion_m: number;
      has_empirical_wrist_canonical?: boolean;
      has_empirical_wrist_pitch_canonical?: boolean;
      has_empirical_wrist_roll_canonical?: boolean;
      robot_verified?: boolean;
      verified_at?: string | null;
      fit_error_cm?: number | null;
      translation_scale?: number;
      calibration_quality?: string | null;
      needs_recapture?: boolean;
      verified_sample_count?: number;
    };
    vr_ctrl_to_ee_ready?: boolean;
    diagnostics?: Record<string, unknown>;
    quality?: {
      offset_step_m: number;
      offset_speed_ema_mps: number;
      ik_reject_fraction: number;
      samples: number;
    };
    robot_verification: {
      state: string;
      sample_count: number;
      samples: RobotVerificationSample[];
      current_label: string;
      robot_start: [number, number, number] | null;
      robot_end: [number, number, number] | null;
      vr_start: [number, number, number] | null;
      translation_scale: number;
      fit_error_cm: number | null;
      sample_residuals: RobotVerificationSampleResidual[];
      worst_residuals?: RobotVerificationSampleResidual[];
      residual_hint?: string;
      quality: "unverified" | "collecting" | "good" | "warn" | "needs_recapture" | "poor" | string;
      verified_at: string | null;
      min_samples: number;
      min_motion_m: number;
      pass_error_cm: number;
      warn_error_cm: number;
      required_labels?: string[];
      missing_labels?: string[];
      needs_recapture?: boolean;
      has_verified_matrix: boolean;
      readiness: "stage1_only" | "verified_test_pending" | "ready_to_record" | string;
      live: RobotVerificationLive;
      test_active: boolean;
      test_completed: boolean;
      test_scale: number;
    };
  };
  /** Per-arm home pose state — read from config/xlerobot.yaml + live flag. */
  home: {
    captured: boolean;
    joints: Record<string, number>;
    homing: boolean;
  };
}

export interface VROperatorArmPanel {
  connected: boolean;
  torque_enabled: boolean;
  anchored: boolean;
  wrist_aligned: boolean;
  active: boolean;
  controller_age_ms: number | null;
  ee_speed_cm_s: number;
  ik_reject_fraction: number;
  recording_readiness: string;
}

export interface VROperatorStatus {
  stage:
    | "connect_required"
    | "mirror_waiting_robot"
    | "mirror_ready"
    | "teleop_head_only"
    | "teleop_arms"
    | "suspended"
    | string;
  guidance: string;
  ready_blockers: string[];
  recording_blockers: string[];
  connection: {
    backend_ready: boolean;
    https_ready: boolean;
    websocket_ready: boolean;
    websocket_clients: number;
    native_quest_ready?: boolean;
    native_quest_clients?: number;
    native_quest_last_seen_ms?: number | null;
    connected_arms: ArmSide[];
  };
  camera_roles: Record<string, {
    configured: boolean;
    name?: string;
    stream_url?: string;
    error?: string;
  }>;
  head_camera_url: string | null;
  arm_panels: Record<ArmSide, VROperatorArmPanel>;
  recording: {
    active: boolean;
    frames: number;
    episodes_saved: number;
    task: string;
    ready: boolean;
  };
  updated_at: number;
}

export interface VRStatus {
  /** New: per-arm state, keyed by side. */
  arms: { left: VRArmState; right: VRArmState };
  /** Condensed Reachy-style state for the in-headset operator UI. */
  operator?: VROperatorStatus;
  calibration_profiles: CalibrationProfilesStatus;
  /** Sides that currently have a motor connection. */
  connected_sides: ArmSide[];
  /** Which arm VR is currently driving (engage-gated bimanual). null = none. */
  active_arm: ArmSide | null;
  /** True when left-controller Y has enabled simultaneous left + right drive. */
  dual_mode: boolean;
  /** Global engage gate. Motors move when engaged and either active_arm or dual_mode is set. */
  engaged: boolean;
  /** Whether dataset recording is currently active (toggled by B button or UI). */
  recording: boolean;
  /** Detail on the LeRobotDataset state. Useful for the UI's Recording card. */
  recording_info: {
    active: boolean;
    episodes_saved: number;
    frames_in_current_episode: number;
    last_episode_index: number | null;
    last_episode_frames: number;
    repo_id: string | null;
    /** Most-recent task description (from UI or previous session). */
    last_task: string;
    /** Absolute filesystem path where datasets are/will be written. */
    root: string;
    calibration_ready: boolean;
    calibration_blockers: string[];
  };
  /** 0.1..1.0 — multiplier on VR delta caps */
  scale: number;
  /** ms since the last drive-loop tick */
  last_tick_age_ms: number | null;
  /** session-level error message, set after a failure (connect, send, etc.) */
  last_error: string | null;
  /** present joints for ALL connected arms (prefixed keys) */
  joint_present: Record<string, number>;
  /** calibration bounds in degrees, per joint */
  joint_bounds: Record<string, [number, number]>;
  /** URL the user should open on the Quest browser */
  vr_endpoint: string | null;
}

export interface QuestVideoStream {
  role: "head" | "left_wrist" | "right_wrist" | string;
  camera_name: string;
  device_path: string;
  width: number;
  height: number;
  fps: number;
  fourcc: string;
  mount: string;
  gst_launch: string;
  udp_port: number;
  receiver_pipeline: string;
  active_gst_launch: string | null;
  running: boolean;
  pid: number | null;
  last_error: string | null;
}

export interface QuestVideoStatus {
  ready: boolean;
  transport: "gstreamer-rtp-h264" | string;
  gst_available: boolean;
  running: boolean;
  quest_host: string | null;
  started_at: number | null;
  roles: string[];
  base_port: number;
  bitrate_kbps: number;
  running_roles: string[];
  missing_roles: string[];
  receive_health: Record<string, {
    role: string;
    received_at: number;
    state: string;
    fps: number;
    latency_ms: number;
    frames: number;
    error: string;
  }>;
  streams: QuestVideoStream[];
}

export interface QuestBridgeStatus {
  clients: number;
  last_seen_ms: number | null;
  endpoint: string;
  ws_url: string;
  coordinate_frame: string;
  pairing_required: boolean;
  max_packet_bytes: number;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, { headers: { "content-type": "application/json" }, ...init });
  if (!r.ok) {
    const text = await r.text().catch(() => r.statusText);
    try {
      const data = JSON.parse(text);
      if (data?.error) {
        const message = typeof data.error === "string" ? data.error : data.error.message;
        const requestId = typeof data.error === "object" && data.error.request_id ? ` [${data.error.request_id}]` : "";
        throw new Error(`${r.status} ${r.statusText}: ${message}${requestId}`);
      }
    } catch (e) {
      if (e instanceof Error && e.message.startsWith(`${r.status}`)) throw e;
    }
    throw new Error(`${r.status} ${r.statusText}: ${text}`);
  }
  return r.json();
}

export const fetcher = <T,>(url: string) => req<T>(url);

export const api = {
  robots: () => req<RobotsResponse>("/api/config/robots"),
  doctor: () => req<{ checks: DoctorCheck[] }>("/api/doctor"),
  cameras: () => req<{ cameras: CameraSpec[]; roles: string[] }>("/api/cameras"),
  assign:  (by_path: string, role: string | null) =>
    req<{ ok: true; cameras: CameraSpec[] }>("/api/cameras/assign", {
      method: "POST", body: JSON.stringify({ by_path, role }),
    }),

  vrStatus: () => req<VRStatus>("/api/vr/status"),
  questBridgeStatus: () => req<QuestBridgeStatus>("/api/vr/quest/status"),
  questVideoStatus: () => req<QuestVideoStatus>("/api/vr/quest/video/status"),
  questVideoStart: (questHost: string, token: string, roles?: string[]) =>
    req<QuestVideoStatus>("/api/vr/quest/video/start", {
      method: "POST",
      headers: { "x-quest-pairing-token": token },
      body: JSON.stringify({ quest_host: questHost, roles }),
    }),
  questVideoStop: (token: string) =>
    req<QuestVideoStatus>("/api/vr/quest/video/stop", {
      method: "POST",
      headers: { "x-quest-pairing-token": token },
    }),
  vrConnect: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/connect", { method: "POST", body: JSON.stringify({ arm }) }),
  /** Pass `arm` to disconnect one side; omit to disconnect both. */
  vrDisconnect: (arm?: ArmSide) =>
    req<VRStatus>("/api/vr/disconnect", {
      method: "POST",
      body: JSON.stringify(arm ? { arm } : {}),
    }),
  /** When both arms are connected, `active_arm` must be set to choose which one VR drives. */
  vrEngage: (engaged: boolean, scale: number, active_arm?: ArmSide) =>
    req<VRStatus>("/api/vr/engage", {
      method: "POST",
      body: JSON.stringify({ engaged, scale, ...(active_arm ? { active_arm } : {}) }),
    }),
  vrSelectCalibrationProfile: (profile: string) =>
    req<VRStatus>("/api/vr/calibration_profile/select", {
      method: "POST", body: JSON.stringify({ profile }),
    }),
  vrCreateCalibrationProfile: (profile: string, copy_from_active = true) =>
    req<VRStatus>("/api/vr/calibration_profile/create", {
      method: "POST", body: JSON.stringify({ profile, copy_from_active }),
    }),
  vrDeleteCalibrationProfile: (profile: string) =>
    req<VRStatus>("/api/vr/calibration_profile/delete", {
      method: "POST", body: JSON.stringify({ profile }),
    }),
  vrEmergencyStop: () =>
    req<VRStatus>("/api/vr/emergency_stop", { method: "POST" }),
  /** UI-side mirror of the B button on the right Quest controller. Pass the
   *  per-episode task description; LeRobot v2 stores it on every frame and
   *  uses it as conditioning input for VLA training. The optional `root` arg
   *  overrides where the dataset is written (null/empty = HF default). */
  vrSetRecording: (enabled: boolean, task: string = "", root: string = "") =>
    req<VRStatus>("/api/vr/recording", {
      method: "POST", body: JSON.stringify({ enabled, task, root }),
    }),
  /** Cache the current UI task text so Quest B-button recording starts use the
   *  same prompt. Empty text clears the cached prompt and start is rejected. */
  vrSetRecordingTask: (task: string) =>
    req<VRStatus>("/api/vr/recording/task", {
      method: "POST", body: JSON.stringify({ task }),
    }),
  /** Persist dataset.root and optionally dataset.repo_id in config/xlerobot.yaml.
   *  Empty root clears the override and returns to the Hugging Face LeRobot
   *  cache default for the saved repo id. */
  vrSetRecordingRoot: (root?: string, repoId?: string) =>
    req<VRStatus>("/api/vr/recording/root", {
      method: "POST", body: JSON.stringify({ root, repo_id: repoId }),
    }),
  /** Delete the most recently saved episode from the active recording dataset. */
  vrDeleteLastRecordingEpisode: () =>
    req<VRStatus>("/api/vr/recording/delete_last", {
      method: "POST", body: JSON.stringify({}),
    }),
  /** Begin a guided motion-based calibration for one arm. */
  vrCalibrateStart: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/calibrate/start", {
      method: "POST", body: JSON.stringify({ arm }),
    }),
  vrCalibrateCancel: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/calibrate/cancel", {
      method: "POST", body: JSON.stringify({ arm }),
    }),
  /** End the wizard's optional wrist pitch/roll motion without capturing an
   *  empirical canonical for the remaining wrist axes. The runtime falls back to the standard
   *  Quest analytical canonical. */
  vrCalibrateSkipWristVerify: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/calibrate/skip_wrist_verify", {
      method: "POST", body: JSON.stringify({ arm }),
    }),
  vrRobotVerifyStart: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/calibrate/robot_verify/start", {
      method: "POST", body: JSON.stringify({ arm, release_torque: false }),
    }),
  vrRobotVerifyCancel: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/calibrate/robot_verify/cancel", {
      method: "POST", body: JSON.stringify({ arm }),
    }),
  vrRobotVerifyRobotPose: (arm: ArmSide, point: "start" | "end", label = "") =>
    req<VRStatus>("/api/vr/calibrate/robot_verify/robot_pose", {
      method: "POST", body: JSON.stringify({ arm, point, label }),
    }),
  vrRobotVerifyVrPose: (arm: ArmSide, point: "start" | "end", label = "") =>
    req<VRStatus>("/api/vr/calibrate/robot_verify/vr_pose", {
      method: "POST", body: JSON.stringify({ arm, point, label }),
    }),
  vrRobotVerifyDiscardLast: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/calibrate/robot_verify/discard_last", {
      method: "POST", body: JSON.stringify({ arm }),
    }),
  vrRobotVerifySolve: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/calibrate/robot_verify/solve", {
      method: "POST", body: JSON.stringify({ arm }),
    }),
  vrRobotVerifyTestStart: (arm: ArmSide, scale = 0.2) =>
    req<VRStatus>("/api/vr/calibrate/robot_verify/test_start", {
      method: "POST", body: JSON.stringify({ arm, scale }),
    }),
  vrRobotVerifyTestStop: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/calibrate/robot_verify/test_stop", {
      method: "POST", body: JSON.stringify({ arm }),
    }),
  /** Read present joints for one arm (or all connected if arm omitted) and
   *  write to config/xlerobot.yaml's home_pose block. */
  vrHomeCapture: (arm?: ArmSide) =>
    req<VRStatus>("/api/vr/home/capture", {
      method: "POST", body: JSON.stringify(arm ? { arm } : {}),
    }),
  vrHomeGo: (arm?: ArmSide) =>
    req<VRStatus>("/api/vr/home/go", {
      method: "POST", body: JSON.stringify(arm ? { arm } : {}),
    }),
  vrHomeCancel: (arm?: ArmSide) =>
    req<VRStatus>("/api/vr/home/cancel", {
      method: "POST", body: JSON.stringify(arm ? { arm } : {}),
    }),
  /** Disable torque on one arm so the user can hand-pose it. */
  vrTorqueRelease: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/torque/release", {
      method: "POST", body: JSON.stringify({ arm }),
    }),
  /** Re-enable torque on one arm at its CURRENT position (no snap-back). */
  vrTorqueLock: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/torque/lock", {
      method: "POST", body: JSON.stringify({ arm }),
    }),
  recentLogs: (lines = 200) =>
    req<{ path: string; lines: string[] }>(`/api/logs/recent?lines=${lines}`),
  jobs: () => req<{ jobs: Job[] }>("/api/jobs"),
  job: (id: string) => req<{ job: Job }>(`/api/jobs/${id}`),
  cancelJob: (id: string) =>
    req<{ job: Job }>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  startTraining: (args: string[]) =>
    req<{ job: Job }>("/api/jobs/train/pi05", { method: "POST", body: JSON.stringify({ args }) }),
  startInference: (args: string[]) =>
    req<{ job: Job }>("/api/jobs/inference/pi05", { method: "POST", body: JSON.stringify({ args }) }),
  startPi05Server: () =>
    req<{ job: Job }>("/api/jobs/pi05/server", { method: "POST", body: JSON.stringify({}) }),
  pushDataset: (args: string[]) =>
    req<{ job: Job }>("/api/jobs/dataset/push", { method: "POST", body: JSON.stringify({ args }) }),
};
