using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace XLeRobot.QuestTeleop
{
    public sealed class XLeRobotQuestRuntimeOrchestrator : MonoBehaviour
    {
        [SerializeField] private XLeRobotQuestClient questClient;
        [SerializeField] private XLeRobotStateClient stateClient;
        [SerializeField] private XLeRobotVideoBridgeClient videoBridgeClient;
        [SerializeField] private XLeRobotAndroidRtpVideoReceiver rtpVideoReceiver;
        [SerializeField] private Text diagnosticsText;
        [SerializeField] private Text headVideoStatusText;
        [SerializeField] private Text leftVideoStatusText;
        [SerializeField] private Text rightVideoStatusText;
        [SerializeField] private GameObject calibrationPanel;
        [SerializeField] private Text calibrationTitleText;
        [SerializeField] private Text calibrationInstructionText;
        [SerializeField] private Text calibrationProgressText;
        [SerializeField] private Image calibrationProgressFill;
        [SerializeField] private GameObject[] calibrationHiddenObjects;
        [SerializeField] private float videoStartRetrySeconds = 5.0f;
        [SerializeField] private float videoStatusRefreshSeconds = 2.0f;
        [SerializeField] private float diagnosticsRefreshSeconds = 0.5f;

        private const float RobotVerificationArrowPixels = 112.0f;
        private const float RobotVerificationMaxArrowPixels = 145.0f;

        private readonly Dictionary<string, XLeRobotVideoReceiveStatus> receiveStatuses = new();
        private XLeRobotQuestStatus latestStatus;
        private string websocketState = "connecting";
        private string lastQuestError = "";
        private string lastStateError = "";
        private string lastVideoError = "";
        private string lastVideoStatusJson = "";
        private float nextVideoStartAttempt;
        private float nextVideoStatusRefresh;
        private float nextDiagnosticsRefresh;
        private bool subscribed;
        private RectTransform robotVerificationVisualRoot;
        private Image robotVerificationTargetShaft;
        private Image robotVerificationTargetHead;
        private Image robotVerificationLiveShaft;
        private Image robotVerificationLiveHead;
        private Image robotVerificationStartMarker;
        private Image robotVerificationTargetMarker;
        private Image robotVerificationLiveMarker;
        private Text robotVerificationTargetLegend;
        private Text robotVerificationLiveLegend;
        private Image robotVerificationDirectionGauge;
        private Image robotVerificationDistanceGauge;
        private Image robotVerificationPositionGauge;
        private Text robotVerificationDirectionText;
        private Text robotVerificationDistanceText;
        private Text robotVerificationPositionText;

        public void Bind(
            XLeRobotQuestClient quest,
            XLeRobotStateClient state,
            XLeRobotVideoBridgeClient bridge,
            XLeRobotAndroidRtpVideoReceiver receiver,
            Text diagnostics,
            Text headStatus,
            Text leftStatus,
            Text rightStatus,
            GameObject calPanel,
            Text calTitle,
            Text calInstruction,
            Text calProgress,
            Image calProgressFillImage,
            GameObject[] hiddenDuringCalibration)
        {
            Unsubscribe();
            questClient = quest;
            stateClient = state;
            videoBridgeClient = bridge;
            rtpVideoReceiver = receiver;
            diagnosticsText = diagnostics;
            headVideoStatusText = headStatus;
            leftVideoStatusText = leftStatus;
            rightVideoStatusText = rightStatus;
            calibrationPanel = calPanel;
            calibrationTitleText = calTitle;
            calibrationInstructionText = calInstruction;
            calibrationProgressText = calProgress;
            calibrationProgressFill = calProgressFillImage;
            calibrationHiddenObjects = hiddenDuringCalibration ?? new GameObject[0];
            Subscribe();
            RenderDiagnostics();
        }

        private void OnEnable()
        {
            Subscribe();
        }

        private void OnDisable()
        {
            Unsubscribe();
        }

        private void Update()
        {
            float now = Time.unscaledTime;
            if (videoBridgeClient != null && now >= nextVideoStatusRefresh)
            {
                videoBridgeClient.RefreshStatus();
                nextVideoStatusRefresh = now + videoStatusRefreshSeconds;
            }

            MaybeStartVideo(now);

            if (now >= nextDiagnosticsRefresh)
            {
                RenderDiagnostics();
                nextDiagnosticsRefresh = now + diagnosticsRefreshSeconds;
            }
        }

        private void Subscribe()
        {
            if (subscribed)
            {
                return;
            }
            if (questClient != null)
            {
                questClient.onConnectionState.AddListener(ApplyQuestConnectionState);
                questClient.onError.AddListener(ApplyQuestError);
            }
            if (stateClient != null)
            {
                stateClient.onStatus.AddListener(ApplyStatus);
                stateClient.onError.AddListener(ApplyStateError);
            }
            if (videoBridgeClient != null)
            {
                videoBridgeClient.onVideoStatusJson.AddListener(ApplyVideoStatusJson);
                videoBridgeClient.onError.AddListener(ApplyVideoError);
            }
            if (rtpVideoReceiver != null)
            {
                rtpVideoReceiver.onReceiveStatus.AddListener(ApplyReceiveStatus);
            }
            subscribed = true;
        }

        private void Unsubscribe()
        {
            if (!subscribed)
            {
                return;
            }
            if (questClient != null)
            {
                questClient.onConnectionState.RemoveListener(ApplyQuestConnectionState);
                questClient.onError.RemoveListener(ApplyQuestError);
            }
            if (stateClient != null)
            {
                stateClient.onStatus.RemoveListener(ApplyStatus);
                stateClient.onError.RemoveListener(ApplyStateError);
            }
            if (videoBridgeClient != null)
            {
                videoBridgeClient.onVideoStatusJson.RemoveListener(ApplyVideoStatusJson);
                videoBridgeClient.onError.RemoveListener(ApplyVideoError);
            }
            if (rtpVideoReceiver != null)
            {
                rtpVideoReceiver.onReceiveStatus.RemoveListener(ApplyReceiveStatus);
            }
            subscribed = false;
        }

        private void MaybeStartVideo(float now)
        {
            if (stateClient == null || videoBridgeClient == null || latestStatus == null)
            {
                return;
            }
            if (latestStatus.VideoReady || videoBridgeClient.StartInFlight)
            {
                return;
            }
            if (now < nextVideoStartAttempt)
            {
                return;
            }

            videoBridgeClient.EnsureVideoStarted();
            nextVideoStartAttempt = now + videoStartRetrySeconds;
        }

        private void ApplyStatus(XLeRobotQuestStatus status)
        {
            latestStatus = status;
            lastStateError = "";
            RenderDiagnostics();
        }

        private void ApplyStateError(string error)
        {
            lastStateError = error ?? "";
            RenderDiagnostics();
        }

        private void ApplyVideoStatusJson(string json)
        {
            lastVideoStatusJson = json ?? "";
            lastVideoError = "";
            RenderDiagnostics();
        }

        private void ApplyVideoError(string error)
        {
            lastVideoError = error ?? "";
            RenderDiagnostics();
        }

        private void ApplyQuestConnectionState(string state)
        {
            websocketState = string.IsNullOrEmpty(state) ? "unknown" : state;
            RenderDiagnostics();
        }

        private void ApplyQuestError(string error)
        {
            lastQuestError = error ?? "";
            RenderDiagnostics();
        }

        private void ApplyReceiveStatus(XLeRobotVideoReceiveStatus status)
        {
            if (status == null || string.IsNullOrEmpty(status.role))
            {
                return;
            }
            receiveStatuses[status.role] = status;
            RenderDiagnostics();
        }

        private void RenderDiagnostics()
        {
            RenderCalibrationGuide();
            SetVideoLabel(headVideoStatusText, "Head", "head");
            SetVideoLabel(leftVideoStatusText, "Left wrist", "left_wrist");
            SetVideoLabel(rightVideoStatusText, "Right wrist", "right_wrist");

            if (diagnosticsText == null)
            {
                return;
            }

            if (latestStatus == null)
            {
                diagnosticsText.text = "Backend: waiting for status | WS: " + websocketState;
                diagnosticsText.color = Color.white;
                return;
            }

            string roles = FormatRoles(latestStatus.video_running_roles);
            string videoLine = latestStatus.VideoRunning
                ? "Video: running " + roles + " @ UDP " + latestStatus.video_base_port
                : "Video: starting @ UDP " + latestStatus.video_base_port;
            string teleopLine = "WS: " + websocketState
                + " | Quest clients: " + latestStatus.native_quest_clients
                + " | Stage: " + (latestStatus.stage ?? "unknown");
            string recordLine = "Arms: " + FormatRoles(latestStatus.connected_arms)
                + " | Active: " + EmptyToDash(latestStatus.active_arm)
                + " | Recording: " + (latestStatus.recording_active ? latestStatus.recording_frames + " frames" : "idle");
            string errorLine = FirstNonEmpty(lastQuestError, lastStateError, lastVideoError, latestStatus.LastError);

            diagnosticsText.text = teleopLine + "\n" + videoLine + "\n" + recordLine
                + (string.IsNullOrEmpty(errorLine) ? "" : "\n" + Trim(errorLine, 135));
            diagnosticsText.color = string.IsNullOrEmpty(errorLine) ? Color.white : new Color(1.0f, 0.78f, 0.35f);
        }

        private void RenderCalibrationGuide()
        {
            bool active = latestStatus != null && (latestStatus.CalibrationActive || latestStatus.RobotVerificationActive);
            if (calibrationPanel != null && calibrationPanel.activeSelf != active)
            {
                calibrationPanel.SetActive(active);
            }
            if (calibrationHiddenObjects != null)
            {
                foreach (GameObject obj in calibrationHiddenObjects)
                {
                    if (obj != null && obj.activeSelf == active)
                    {
                        obj.SetActive(!active);
                    }
                }
            }
            if (!active)
            {
                SetRobotVerificationVisualsActive(false);
                return;
            }

            if (latestStatus.RobotVerificationActive && !latestStatus.CalibrationActive)
            {
                RenderRobotVerificationGuide();
                return;
            }
            SetRobotVerificationVisualsActive(false);

            string side = EmptyToDash(latestStatus.calibration_side);
            string state = latestStatus.calibration_state ?? "";
            int step = CalibrationStep(state);
            int stepCount = 5;
            float motion = MotionForState(latestStatus, state);
            float target = TargetForState(latestStatus, state);
            float progress = Mathf.Clamp01(target > 0.001f ? motion / target : 0.0f);
            float totalProgress = Mathf.Clamp01(((float)(step - 1) + progress) / stepCount);

            if (calibrationTitleText != null)
            {
                calibrationTitleText.text = "VR calibration | " + side + " arm | step " + step + "/" + stepCount;
            }
            if (calibrationInstructionText != null)
            {
                calibrationInstructionText.text = CalibrationInstruction(side, state);
            }
            if (calibrationProgressText != null)
            {
                string metric = IsWristState(state)
                    ? motion.ToString("0") + " deg / " + target.ToString("0") + " deg"
                    : (motion * 100.0f).ToString("0.0") + " cm / " + (target * 100.0f).ToString("0.0") + " cm";
                string confidence = string.IsNullOrEmpty(latestStatus.calibration_confidence)
                    ? ""
                    : " | confidence: " + latestStatus.calibration_confidence;
                calibrationProgressText.text = "Progress: " + metric + confidence;
            }
            if (calibrationProgressFill != null)
            {
                calibrationProgressFill.fillAmount = totalProgress;
            }
        }

        private void RenderRobotVerificationGuide()
        {
            string side = EmptyToDash(latestStatus.robot_verification_side);
            string label = EmptyToDash(latestStatus.robot_verification_label);
            string state = latestStatus.robot_verification_state ?? "";
            string liveState = latestStatus.robot_verification_live_state ?? "";
            float sampleProgress = Mathf.Clamp01(
                latestStatus.robot_verification_min_samples > 0
                    ? (float)latestStatus.robot_verification_sample_count / latestStatus.robot_verification_min_samples
                    : 0.0f);

            if (calibrationTitleText != null)
            {
                calibrationTitleText.text = "Robot verification | " + side + " arm | " + latestStatus.robot_verification_sample_count
                    + "/" + latestStatus.robot_verification_min_samples;
            }
            if (calibrationInstructionText != null)
            {
                calibrationInstructionText.text = RobotVerificationInstruction(label, state, liveState);
            }
            if (calibrationProgressText != null)
            {
                calibrationProgressText.text = RobotVerificationMetrics(latestStatus);
            }
            if (calibrationProgressFill != null)
            {
                calibrationProgressFill.fillAmount = sampleProgress;
            }
            RenderRobotVerificationVisuals(latestStatus, state, liveState);
        }

        private void RenderRobotVerificationVisuals(XLeRobotQuestStatus status, string state, string liveState)
        {
            EnsureRobotVerificationVisuals();
            if (robotVerificationVisualRoot == null)
            {
                return;
            }
            SetRobotVerificationVisualsActive(true);

            bool hasTarget = TryProjectRobotDelta(status.robot_verification_target_robot_delta, out Vector2 targetProjected);
            bool hasLive = TryProjectRobotDelta(status.robot_verification_predicted_robot_delta, out Vector2 liveProjected);
            Vector2 origin = new Vector2(-170.0f, -20.0f);
            float targetScale = RobotVerificationArrowPixels / Mathf.Max(0.001f, hasTarget ? targetProjected.magnitude : 0.0f);
            float liveScale = hasTarget
                ? targetScale
                : RobotVerificationArrowPixels / Mathf.Max(0.001f, hasLive ? liveProjected.magnitude : 0.0f);
            if (hasLive && liveProjected.magnitude * liveScale > RobotVerificationMaxArrowPixels)
            {
                liveScale = RobotVerificationMaxArrowPixels / Mathf.Max(0.001f, liveProjected.magnitude);
            }

            Color targetColor = new Color(0.38f, 0.82f, 1.0f, 0.95f);
            Color liveColor = QualityColor(status.robot_verification_direction_error_deg, 12.0f, 25.0f, false);
            if (liveState != "good" && liveState != "adjust")
            {
                liveColor = new Color(1.0f, 0.82f, 0.28f, 0.95f);
            }
            if (liveState == "good")
            {
                liveColor = new Color(0.38f, 1.0f, 0.52f, 0.95f);
            }

            Vector2 targetEnd = origin;
            if (hasTarget)
            {
                targetEnd = origin + targetProjected * targetScale;
            }
            Vector2 liveEnd = origin;
            if (hasLive)
            {
                liveEnd = origin + liveProjected * liveScale;
            }

            SetArrow(robotVerificationTargetShaft, robotVerificationTargetHead, origin, targetEnd, targetColor, hasTarget);
            SetArrow(robotVerificationLiveShaft, robotVerificationLiveHead, origin, liveEnd, liveColor, hasLive);
            SetMarker(robotVerificationStartMarker, origin, new Color(1.0f, 1.0f, 1.0f, 0.95f), hasTarget || state == "vr_start_captured");
            SetMarker(robotVerificationTargetMarker, targetEnd, targetColor, hasTarget);
            SetMarker(robotVerificationLiveMarker, liveEnd, liveColor, hasLive);

            if (robotVerificationTargetLegend != null)
            {
                robotVerificationTargetLegend.text = hasTarget ? "target robot move" : "target pending";
                robotVerificationTargetLegend.color = targetColor;
            }
            if (robotVerificationLiveLegend != null)
            {
                robotVerificationLiveLegend.text = hasLive ? "your mapped move" : "move with grip";
                robotVerificationLiveLegend.color = liveColor;
            }

            SetQualityGauge(
                robotVerificationDirectionGauge,
                robotVerificationDirectionText,
                "DIR",
                status.robot_verification_direction_error_deg,
                "0 deg",
                12.0f,
                25.0f,
                false,
                liveState == "good" || liveState == "adjust");
            SetQualityGauge(
                robotVerificationDistanceGauge,
                robotVerificationDistanceText,
                "DIST",
                status.robot_verification_magnitude_ratio,
                "0.00x",
                0.15f,
                0.40f,
                true,
                liveState == "good" || liveState == "adjust");
            SetQualityGauge(
                robotVerificationPositionGauge,
                robotVerificationPositionText,
                "POS",
                status.robot_verification_position_error_cm,
                "0.0 cm",
                2.0f,
                5.0f,
                false,
                liveState == "good" || liveState == "adjust");
        }

        private void EnsureRobotVerificationVisuals()
        {
            if (robotVerificationVisualRoot != null || calibrationPanel == null)
            {
                return;
            }

            GameObject root = new GameObject("RobotVerificationVisuals");
            root.transform.SetParent(calibrationPanel.transform, false);
            robotVerificationVisualRoot = root.AddComponent<RectTransform>();
            robotVerificationVisualRoot.anchoredPosition = new Vector2(0.0f, -54.0f);
            robotVerificationVisualRoot.sizeDelta = new Vector2(720.0f, 168.0f);

            robotVerificationTargetLegend = CreateVisualText(root.transform, "TargetLegend", new Vector2(-230.0f, 62.0f), new Vector2(190.0f, 28.0f), 15);
            robotVerificationLiveLegend = CreateVisualText(root.transform, "LiveLegend", new Vector2(-30.0f, 62.0f), new Vector2(190.0f, 28.0f), 15);

            robotVerificationTargetShaft = CreateVisualImage(root.transform, "TargetArrowShaft", targetGraphic: true);
            robotVerificationTargetHead = CreateVisualImage(root.transform, "TargetArrowHead", targetGraphic: true);
            robotVerificationLiveShaft = CreateVisualImage(root.transform, "LiveArrowShaft", targetGraphic: true);
            robotVerificationLiveHead = CreateVisualImage(root.transform, "LiveArrowHead", targetGraphic: true);
            robotVerificationStartMarker = CreateVisualImage(root.transform, "StartMarker", targetGraphic: true);
            robotVerificationTargetMarker = CreateVisualImage(root.transform, "TargetMarker", targetGraphic: true);
            robotVerificationLiveMarker = CreateVisualImage(root.transform, "LiveMarker", targetGraphic: true);

            CreateGauge(
                root.transform,
                "DirectionGauge",
                new Vector2(165.0f, 4.0f),
                out robotVerificationDirectionGauge,
                out robotVerificationDirectionText);
            CreateGauge(
                root.transform,
                "DistanceGauge",
                new Vector2(255.0f, 4.0f),
                out robotVerificationDistanceGauge,
                out robotVerificationDistanceText);
            CreateGauge(
                root.transform,
                "PositionGauge",
                new Vector2(345.0f, 4.0f),
                out robotVerificationPositionGauge,
                out robotVerificationPositionText);

            SetRobotVerificationVisualsActive(false);
        }

        private void SetRobotVerificationVisualsActive(bool active)
        {
            if (robotVerificationVisualRoot != null && robotVerificationVisualRoot.gameObject.activeSelf != active)
            {
                robotVerificationVisualRoot.gameObject.SetActive(active);
            }
        }

        private static Text CreateVisualText(Transform parent, string name, Vector2 position, Vector2 size, int fontSize)
        {
            GameObject obj = new GameObject(name);
            obj.transform.SetParent(parent, false);
            Text text = obj.AddComponent<Text>();
            text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            text.fontSize = fontSize;
            text.alignment = TextAnchor.MiddleCenter;
            text.horizontalOverflow = HorizontalWrapMode.Wrap;
            text.verticalOverflow = VerticalWrapMode.Truncate;
            RectTransform rect = obj.GetComponent<RectTransform>();
            rect.anchoredPosition = position;
            rect.sizeDelta = size;
            return text;
        }

        private static Image CreateVisualImage(Transform parent, string name, bool targetGraphic)
        {
            GameObject obj = new GameObject(name);
            obj.transform.SetParent(parent, false);
            Image image = obj.AddComponent<Image>();
            image.raycastTarget = targetGraphic;
            RectTransform rect = obj.GetComponent<RectTransform>();
            rect.sizeDelta = new Vector2(1.0f, 1.0f);
            return image;
        }

        private static void CreateGauge(Transform parent, string name, Vector2 position, out Image fill, out Text label)
        {
            GameObject gauge = new GameObject(name);
            gauge.transform.SetParent(parent, false);
            RectTransform gaugeRect = gauge.AddComponent<RectTransform>();
            gaugeRect.anchoredPosition = position;
            gaugeRect.sizeDelta = new Vector2(74.0f, 74.0f);

            Image background = gauge.AddComponent<Image>();
            background.color = new Color(1.0f, 1.0f, 1.0f, 0.12f);
            background.raycastTarget = false;

            GameObject fillObject = new GameObject("Fill");
            fillObject.transform.SetParent(gauge.transform, false);
            fill = fillObject.AddComponent<Image>();
            fill.color = new Color(1.0f, 0.82f, 0.28f, 0.95f);
            fill.type = Image.Type.Filled;
            fill.fillMethod = Image.FillMethod.Radial360;
            fill.fillOrigin = 2;
            fill.fillClockwise = true;
            fill.fillAmount = 0.0f;
            fill.raycastTarget = false;
            RectTransform fillRect = fillObject.GetComponent<RectTransform>();
            fillRect.anchorMin = Vector2.zero;
            fillRect.anchorMax = Vector2.one;
            fillRect.offsetMin = Vector2.zero;
            fillRect.offsetMax = Vector2.zero;

            label = CreateVisualText(gauge.transform, "Label", Vector2.zero, new Vector2(70.0f, 46.0f), 13);
            label.color = Color.white;
        }

        private static bool TryProjectRobotDelta(float[] values, out Vector2 projected)
        {
            projected = Vector2.zero;
            if (values == null || values.Length < 3)
            {
                return false;
            }
            float x = values[0];
            float y = values[1];
            float z = values[2];
            if (!IsFinite(x) || !IsFinite(y) || !IsFinite(z))
            {
                return false;
            }
            if (Mathf.Abs(x) + Mathf.Abs(y) + Mathf.Abs(z) < 0.0001f)
            {
                return false;
            }

            projected = new Vector2(-y + 0.45f * x, z + 0.28f * x);
            if (projected.sqrMagnitude < 0.0000001f)
            {
                projected = new Vector2(0.0f, Mathf.Sign(z == 0.0f ? x : z) * 0.0001f);
            }
            return true;
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }

        private static void SetArrow(Image shaft, Image head, Vector2 start, Vector2 end, Color color, bool visible)
        {
            if (shaft == null || head == null)
            {
                return;
            }
            shaft.gameObject.SetActive(visible);
            head.gameObject.SetActive(visible);
            if (!visible)
            {
                return;
            }

            Vector2 delta = end - start;
            float length = delta.magnitude;
            if (length < 8.0f)
            {
                shaft.gameObject.SetActive(false);
                head.gameObject.SetActive(false);
                return;
            }

            float angle = Mathf.Atan2(delta.y, delta.x) * Mathf.Rad2Deg;
            Vector2 direction = delta / length;
            RectTransform shaftRect = shaft.GetComponent<RectTransform>();
            shaftRect.anchoredPosition = start + direction * (length * 0.5f - 7.0f);
            shaftRect.sizeDelta = new Vector2(Mathf.Max(4.0f, length - 18.0f), 7.0f);
            shaftRect.localRotation = Quaternion.Euler(0.0f, 0.0f, angle);
            shaft.color = color;

            RectTransform headRect = head.GetComponent<RectTransform>();
            headRect.anchoredPosition = end;
            headRect.sizeDelta = new Vector2(17.0f, 17.0f);
            headRect.localRotation = Quaternion.Euler(0.0f, 0.0f, angle + 45.0f);
            head.color = color;
        }

        private static void SetMarker(Image marker, Vector2 position, Color color, bool visible)
        {
            if (marker == null)
            {
                return;
            }
            marker.gameObject.SetActive(visible);
            if (!visible)
            {
                return;
            }
            RectTransform rect = marker.GetComponent<RectTransform>();
            rect.anchoredPosition = position;
            rect.sizeDelta = new Vector2(16.0f, 16.0f);
            rect.localRotation = Quaternion.identity;
            marker.color = color;
        }

        private static void SetQualityGauge(
            Image fill,
            Text label,
            string title,
            float value,
            string format,
            float greenLimit,
            float yellowLimit,
            bool centeredOnOne,
            bool hasValue)
        {
            if (fill == null || label == null)
            {
                return;
            }
            if (!hasValue || !IsFinite(value))
            {
                fill.fillAmount = 0.0f;
                fill.color = new Color(0.7f, 0.74f, 0.78f, 0.45f);
                label.text = title + "\n--";
                label.color = new Color(0.86f, 0.9f, 0.95f, 0.85f);
                return;
            }

            float error = centeredOnOne ? Mathf.Abs(value - 1.0f) : Mathf.Max(0.0f, value);
            Color color = QualityColor(error, greenLimit, yellowLimit, true);
            float fillAmount = Mathf.Clamp01(1.0f - (error / Mathf.Max(0.001f, yellowLimit)));
            fill.fillAmount = Mathf.Max(0.12f, fillAmount);
            fill.color = color;
            label.text = title + "\n" + value.ToString(format);
            label.color = Color.white;
        }

        private static Color QualityColor(float value, float greenLimit, float yellowLimit, bool lowerIsBetter)
        {
            if (!IsFinite(value))
            {
                return new Color(0.7f, 0.74f, 0.78f, 0.65f);
            }
            float score = lowerIsBetter ? Mathf.Max(0.0f, value) : value;
            if (score <= greenLimit)
            {
                return new Color(0.38f, 1.0f, 0.52f, 0.95f);
            }
            if (score <= yellowLimit)
            {
                return new Color(1.0f, 0.82f, 0.28f, 0.95f);
            }
            return new Color(1.0f, 0.35f, 0.28f, 0.95f);
        }

        private void SetVideoLabel(Text label, string displayName, string role)
        {
            if (label == null)
            {
                return;
            }

            receiveStatuses.TryGetValue(role, out XLeRobotVideoReceiveStatus receiveStatus);
            bool backendRunning = BackendRoleRunning(role);
            if (receiveStatus != null && !string.IsNullOrEmpty(receiveStatus.error))
            {
                label.text = displayName + ": " + Trim(receiveStatus.error, 42);
                label.color = new Color(1.0f, 0.45f, 0.35f);
                return;
            }
            if (receiveStatus != null && receiveStatus.frames > 0)
            {
                label.text = displayName + ": " + receiveStatus.frames + " frames";
                label.color = new Color(0.45f, 1.0f, 0.55f);
                return;
            }
            if (backendRunning)
            {
                int port = receiveStatus != null ? receiveStatus.port : PortForRole(role);
                label.text = displayName + ": waiting for RTP " + port;
                label.color = new Color(1.0f, 0.86f, 0.35f);
                return;
            }
            if (latestStatus != null && !latestStatus.VideoReady && !string.IsNullOrEmpty(lastVideoError))
            {
                label.text = displayName + ": video error";
                label.color = new Color(1.0f, 0.45f, 0.35f);
                return;
            }

            label.text = displayName + ": starting";
            label.color = new Color(0.78f, 0.82f, 0.88f);
        }

        private bool BackendRoleRunning(string role)
        {
            if (latestStatus == null)
            {
                return false;
            }
            string[] roles = latestStatus.video_running_roles;
            if (roles == null || roles.Length == 0)
            {
                return latestStatus.VideoRunning;
            }
            foreach (string runningRole in roles)
            {
                if (runningRole == role)
                {
                    return true;
                }
            }
            return false;
        }

        private int PortForRole(string role)
        {
            int basePort = latestStatus != null && latestStatus.video_base_port > 0
                ? latestStatus.video_base_port
                : (rtpVideoReceiver != null ? rtpVideoReceiver.BasePort : 5600);
            switch (role)
            {
                case "left_wrist": return basePort + 1;
                case "right_wrist": return basePort + 2;
                default: return basePort;
            }
        }

        private static string FormatRoles(string[] roles)
        {
            if (roles == null || roles.Length == 0)
            {
                return "-";
            }
            return string.Join(", ", roles);
        }

        private static string EmptyToDash(string value)
        {
            return string.IsNullOrEmpty(value) ? "-" : value;
        }

        private static string FirstNonEmpty(params string[] values)
        {
            foreach (string value in values)
            {
                if (!string.IsNullOrEmpty(value))
                {
                    return value;
                }
            }
            return "";
        }

        private static string Trim(string value, int maxChars)
        {
            if (string.IsNullOrEmpty(value) || value.Length <= maxChars)
            {
                return value ?? "";
            }
            return value.Substring(0, maxChars - 3) + "...";
        }

        private static int CalibrationStep(string state)
        {
            switch (state)
            {
                case "awaiting_anchor_up":
                case "motioning_up":
                    return 2;
                case "awaiting_anchor_left":
                case "motioning_left":
                    return 3;
                case "awaiting_anchor_wrist_verify":
                case "awaiting_anchor_wrist_pitch":
                case "motioning_wrist_verify":
                case "motioning_wrist_pitch":
                    return 4;
                case "awaiting_anchor_wrist_roll":
                case "motioning_wrist_roll":
                    return 5;
                default:
                    return 1;
            }
        }

        private static bool IsWristState(string state)
        {
            return state != null && state.Contains("wrist");
        }

        private static float MotionForState(XLeRobotQuestStatus status, string state)
        {
            if (status == null)
            {
                return 0.0f;
            }
            if (state == "motioning_wrist_roll" || state == "awaiting_anchor_wrist_roll")
            {
                return Mathf.Abs(status.calibration_wrist_roll_deg);
            }
            if (state == "motioning_wrist_pitch" || state == "motioning_wrist_verify"
                || state == "awaiting_anchor_wrist_pitch" || state == "awaiting_anchor_wrist_verify")
            {
                return Mathf.Abs(status.calibration_wrist_pitch_deg);
            }
            return Mathf.Max(0.0f, status.calibration_motion_m);
        }

        private static float TargetForState(XLeRobotQuestStatus status, string state)
        {
            if (status == null)
            {
                return 1.0f;
            }
            if (IsWristState(state))
            {
                return Mathf.Max(1.0f, status.calibration_wrist_target_deg);
            }
            return Mathf.Max(0.001f, status.calibration_target_m);
        }

        private static string CalibrationInstruction(string side, string state)
        {
            switch (state)
            {
                case "awaiting_anchor_fwd":
                    return "Hold the " + side + " grip to mark the start point for forward motion.";
                case "motioning_fwd":
                    return "Keep grip held. Move the " + side + " hand forward about 10 cm, then release.";
                case "awaiting_anchor_up":
                    return "Hold the " + side + " grip to mark the start point for upward motion.";
                case "motioning_up":
                    return "Keep grip held. Move the " + side + " hand upward about 10 cm, then release.";
                case "awaiting_anchor_left":
                    return "Hold the " + side + " grip to mark the start point for leftward motion.";
                case "motioning_left":
                    return "Keep grip held. Move the " + side + " hand left about 10 cm, then release.";
                case "awaiting_anchor_wrist_verify":
                case "awaiting_anchor_wrist_pitch":
                    return "Hold the " + side + " grip, then pitch the wrist upward clearly.";
                case "motioning_wrist_verify":
                case "motioning_wrist_pitch":
                    return "Keep grip held. Pitch the " + side + " wrist upward, then release.";
                case "awaiting_anchor_wrist_roll":
                    return "Hold the " + side + " grip, then roll the wrist to the right clearly.";
                case "motioning_wrist_roll":
                    return "Keep grip held. Roll the " + side + " wrist right, then release.";
                default:
                    return "Follow the calibration step shown in the web dashboard.";
            }
        }

        private static string RobotVerificationInstruction(string label, string state, string liveState)
        {
            if (string.IsNullOrEmpty(label) || label == "-")
            {
                label = "selected";
            }
            if (state == "collecting" || state == "robot_start_captured")
            {
                return "Use the web UI to capture robot start and robot end for the " + label + " sample.";
            }
            if (state == "robot_end_captured")
            {
                return "Hold grip with the controller natural, then press A/X to set VR start for " + label + ".";
            }
            if (state == "vr_start_captured")
            {
                if (liveState == "good")
                {
                    return "Good match. Keep grip held and press A/X before releasing to save VR end for " + label + ".";
                }
                return "Keep grip held. Move the controller like the robot " + label + " motion, then press A/X before releasing.";
            }
            return "Follow robot verification in the web UI. Grip+A/X captures the next VR point.";
        }

        private static string RobotVerificationMetrics(XLeRobotQuestStatus status)
        {
            string controls = string.IsNullOrEmpty(status.robot_verification_controls)
                ? "Grip+A/X captures VR point"
                : status.robot_verification_controls;
            string message = string.IsNullOrEmpty(status.robot_verification_message)
                ? ""
                : status.robot_verification_message + "\n";
            string residualHint = string.IsNullOrEmpty(status.robot_verification_residual_hint)
                ? ""
                : status.robot_verification_residual_hint + "\n";
            return message + residualHint + controls + "\n" + RobotVerificationCorrection(status);
        }

        private static string RobotVerificationCorrection(XLeRobotQuestStatus status)
        {
            string liveState = status.robot_verification_live_state ?? "";
            if (liveState == "good")
            {
                return "Quality green. Keep grip held and capture VR end.";
            }
            if (liveState == "adjust")
            {
                if (status.robot_verification_direction_error_deg > 25.0f)
                {
                    return "Correction: turn your movement toward the target arrow.";
                }
                if (status.robot_verification_magnitude_ratio < 0.60f)
                {
                    return "Correction: move farther along the same direction.";
                }
                if (status.robot_verification_magnitude_ratio > 1.60f)
                {
                    return "Correction: make a shorter controller move.";
                }
                return "Correction: align the live arrow endpoint with the target marker.";
            }
            if (liveState == "move_vr")
            {
                return "Move with grip until the live arrow reaches the target direction.";
            }
            if (liveState == "need_vr_neutral")
            {
                return "Hold grip and press A/X with the controller natural to place the start marker.";
            }
            return "State: " + EmptyToDash(liveState);
        }
    }
}
