using System;
using UnityEngine;

namespace XLeRobot.QuestTeleop
{
    [Serializable]
    public sealed class XLeRobotQuestSettings
    {
        public string workstationHost = "192.168.0.113";
        public int workstationPort = 5000;
        public bool useTls = false;
        public string pairingToken = "dev-quest-token";

        public string HttpBaseUrl
        {
            get
            {
                string scheme = useTls ? "https" : "http";
                return $"{scheme}://{workstationHost}:{workstationPort}";
            }
        }

        public string WebSocketBaseUrl
        {
            get
            {
                string scheme = useTls ? "wss" : "ws";
                return $"{scheme}://{workstationHost}:{workstationPort}";
            }
        }
    }

    public enum XLeRobotOperatorStage
    {
        Unknown,
        ConnectRequired,
        MirrorWaitingRobot,
        MirrorReady,
        TeleopHeadOnly,
        TeleopArms,
        Suspended
    }

    public sealed class XLeRobotQuestStatus
    {
        public string stage = "";
        public string guidance = "";
        public string last_error = "";
        public string ws_url = "";
        public string coordinate_frame = "";
        public bool native_quest_ready;
        public int native_quest_clients;
        public int native_quest_last_seen_ms;
        public bool recording_active;
        public int recording_frames;
        public int recording_episodes_saved;
        public bool recording_ready;
        public bool dual_mode;
        public bool engaged;
        public bool video_ready;
        public bool video_running;
        public string video_transport = "";
        public int video_base_port = 5600;
        public int video_bitrate_kbps;
        public string[] video_running_roles = Array.Empty<string>();
        public string[] ready_blockers = Array.Empty<string>();
        public string[] recording_blockers = Array.Empty<string>();
        public string[] connected_arms = Array.Empty<string>();
        public string active_arm = "";
        public bool calibration_active;
        public string calibration_side = "";
        public string calibration_state = "";
        public float calibration_motion_m;
        public float calibration_target_m = 0.10f;
        public float calibration_min_m = 0.05f;
        public float calibration_wrist_pitch_deg;
        public float calibration_wrist_roll_deg;
        public float calibration_wrist_target_deg = 30.0f;
        public float calibration_wrist_min_deg = 15.0f;
        public string calibration_confidence = "";
        public bool robot_verification_active;
        public string robot_verification_side = "";
        public string robot_verification_state = "";
        public string robot_verification_label = "";
        public int robot_verification_sample_count;
        public int robot_verification_min_samples = 6;
        public string robot_verification_quality = "";
        public string robot_verification_message = "";
        public string robot_verification_live_state = "";
        public bool robot_verification_ready;
        public float robot_verification_direction_error_deg;
        public float robot_verification_magnitude_ratio;
        public float robot_verification_position_error_cm;
        public float robot_verification_vr_motion_m;
        public float robot_verification_target_motion_m;
        public float[] robot_verification_target_robot_delta = Array.Empty<float>();
        public float[] robot_verification_predicted_robot_delta = Array.Empty<float>();
        public float[] robot_verification_vr_delta = Array.Empty<float>();
        public float robot_verification_fit_error_cm;
        public string robot_verification_residual_hint = "";
        public string robot_verification_controls = "";

        public bool NativeQuestReady => native_quest_ready;
        public bool RecordingActive => recording_active;
        public bool RecordingReady => recording_ready;
        public bool VideoReady => video_ready;
        public bool VideoRunning => video_running;
        public bool CalibrationActive => calibration_active;
        public bool RobotVerificationActive => robot_verification_active;

        public XLeRobotOperatorStage Stage
        {
            get { return ParseStage(stage); }
        }

        public string LastError
        {
            get { return last_error ?? ""; }
        }

        public static XLeRobotQuestStatus FromJson(string json)
        {
            if (string.IsNullOrEmpty(json))
            {
                return new XLeRobotQuestStatus();
            }
            return JsonUtility.FromJson<XLeRobotQuestStatus>(json);
        }

        private static XLeRobotOperatorStage ParseStage(string value)
        {
            switch (value)
            {
                case "connect_required": return XLeRobotOperatorStage.ConnectRequired;
                case "mirror_waiting_robot": return XLeRobotOperatorStage.MirrorWaitingRobot;
                case "mirror_ready": return XLeRobotOperatorStage.MirrorReady;
                case "teleop_head_only": return XLeRobotOperatorStage.TeleopHeadOnly;
                case "teleop_arms": return XLeRobotOperatorStage.TeleopArms;
                case "suspended": return XLeRobotOperatorStage.Suspended;
                default: return XLeRobotOperatorStage.Unknown;
            }
        }
    }

    [Serializable]
    public sealed class XLeRobotVideoReceiveStatus
    {
        public string role = "";
        public string state = "";
        public int port;
        public int frames;
        public bool running;
        public string error = "";

        public bool HasFrames => frames > 0;
    }
}
