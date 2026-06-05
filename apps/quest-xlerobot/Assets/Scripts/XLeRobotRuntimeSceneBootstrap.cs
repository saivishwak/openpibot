using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;

namespace XLeRobot.QuestTeleop
{
    /// <summary>
    /// Runtime scene composer for the standalone Quest build.
    ///
    /// The repository intentionally does not hand-author Unity YAML references to
    /// scripts because those references depend on generated .meta GUIDs. This
    /// bootstrap creates the minimal Reachy-style operator surface at startup:
    /// an operator origin, headset camera, status/suspension panel, video
    /// surfaces, and the backend clients wired together.
    /// </summary>
    public sealed class XLeRobotRuntimeSceneBootstrap : MonoBehaviour
    {
        private const string PlayerPrefsHost = "xlerobot.quest.host";
        private const string PlayerPrefsPort = "xlerobot.quest.port";
        private const string PlayerPrefsTls = "xlerobot.quest.tls";
        private const string PlayerPrefsToken = "xlerobot.quest.token";
        private const string DefaultLocalHost = "192.168.0.113";
        private const int DefaultLocalPort = 5000;
        private const string DefaultLocalToken = "dev-quest-token";

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void EnsureRuntimeScene()
        {
            if (FindObjectOfType<XLeRobotQuestClient>() != null)
            {
                return;
            }

            GameObject root = new GameObject("XLeRobot Quest Runtime");
            DontDestroyOnLoad(root);
            root.AddComponent<XLeRobotRuntimeSceneBootstrap>().Build(root.transform);
        }

        private void Build(Transform root)
        {
            Transform operatorOrigin = CreateOperatorOrigin(root);
            Camera headsetCamera = CreateHeadsetCamera(operatorOrigin);
            ConfigurePassthrough(root, headsetCamera);
            Canvas canvas = CreateWorldCanvas(headsetCamera.transform);
            Text stageText = CreateText(canvas.transform, "StageText", new Vector2(0.0f, 155.0f), 28, "Connecting");
            Text guidanceText = CreateText(canvas.transform, "GuidanceText", new Vector2(0.0f, 115.0f), 18, "Waiting for backend status...");
            Image indicator = CreateIndicator(canvas.transform);
            GameObject suspensionOverlay = CreateSuspensionOverlay(canvas.transform);

            RawImage head = CreateVideoPanel(canvas.transform, "HeadVideo", new Vector2(0.0f, -25.0f), new Vector2(380.0f, 210.0f));
            Text headVideoStatus = CreateText(canvas.transform, "HeadVideoStatus", new Vector2(0.0f, -150.0f), 13, "Head: starting");
            RawImage left = CreateVideoPanel(canvas.transform, "LeftWristVideo", new Vector2(-200.0f, -245.0f), new Vector2(180.0f, 108.0f));
            RawImage right = CreateVideoPanel(canvas.transform, "RightWristVideo", new Vector2(200.0f, -245.0f), new Vector2(180.0f, 108.0f));
            Text leftVideoStatus = CreateText(canvas.transform, "LeftWristVideoStatus", new Vector2(-200.0f, -315.0f), 12, "Left wrist: starting");
            Text rightVideoStatus = CreateText(canvas.transform, "RightWristVideoStatus", new Vector2(200.0f, -315.0f), 12, "Right wrist: starting");
            Text diagnosticsText = CreateText(canvas.transform, "QuestDiagnostics", new Vector2(0.0f, -380.0f), 13, "Backend: waiting for status");
            ConfigureTextRect(diagnosticsText, new Vector2(820.0f, 92.0f), TextAnchor.UpperLeft);
            GameObject calibrationPanel = CreateCalibrationPanel(
                canvas.transform,
                out Text calibrationTitle,
                out Text calibrationInstruction,
                out Text calibrationProgress,
                out Image calibrationProgressFill);

            GameObject systems = new GameObject("Quest Systems");
            systems.transform.SetParent(root, false);
            XLeRobotQuestClient questClient = systems.AddComponent<XLeRobotQuestClient>();
            XLeRobotStateClient stateClient = systems.AddComponent<XLeRobotStateClient>();
            XLeRobotOperatorFlowManager flow = systems.AddComponent<XLeRobotOperatorFlowManager>();
            XLeRobotVideoSurfaceBinder videoBinder = systems.AddComponent<XLeRobotVideoSurfaceBinder>();
            XLeRobotGStreamerTextureAdapter textureAdapter = systems.AddComponent<XLeRobotGStreamerTextureAdapter>();
            XLeRobotGStreamerWebRtcReceiver webRtcReceiver = systems.AddComponent<XLeRobotGStreamerWebRtcReceiver>();
            XLeRobotAndroidRtpVideoReceiver rtpVideoReceiver = systems.AddComponent<XLeRobotAndroidRtpVideoReceiver>();
            XLeRobotVideoBridgeClient videoBridge = systems.AddComponent<XLeRobotVideoBridgeClient>();
            XLeRobotQuestBootstrap bootstrap = systems.AddComponent<XLeRobotQuestBootstrap>();
            XLeRobotEventManager eventManager = systems.AddComponent<XLeRobotEventManager>();
            XLeRobotScenesManager scenesManager = systems.AddComponent<XLeRobotScenesManager>();
            XLeRobotUserTrackerManager userTracker = systems.AddComponent<XLeRobotUserTrackerManager>();
            XLeRobotMirrorSceneManager mirrorScene = systems.AddComponent<XLeRobotMirrorSceneManager>();
            XLeRobotTeleoperationSceneManager teleoperationScene = systems.AddComponent<XLeRobotTeleoperationSceneManager>();
            XLeRobotSuspensionUIManager suspension = systems.AddComponent<XLeRobotSuspensionUIManager>();
            XLeRobotEmergencyStopInput emergencyStop = systems.AddComponent<XLeRobotEmergencyStopInput>();
            XLeRobotQuestRuntimeOrchestrator orchestrator = systems.AddComponent<XLeRobotQuestRuntimeOrchestrator>();

            flow.BindUi(stageText, guidanceText, indicator);
            videoBinder.BindRawImages(head, left, right);
            textureAdapter.Bind(videoBinder);
            videoBridge.Configure(stateClient);
            webRtcReceiver.Bind(null, textureAdapter, videoBridge);
            rtpVideoReceiver.Bind(textureAdapter, videoBridge);
            userTracker.BindHeadset(headsetCamera.transform);
            mirrorScene.Bind(bootstrap);
            suspension.BindOverlay(suspensionOverlay);
            emergencyStop.Bind(stateClient);
            questClient.BindOperatorOrigin(operatorOrigin);
            questClient.onRawServerMessage.AddListener(stateClient.ApplyServerMessage);
            questClient.onError.AddListener(flow.ApplyConnectionError);
            stateClient.onStatus.AddListener(scenesManager.ApplyStatus);
            stateClient.onStatus.AddListener(suspension.ApplyStatus);
            stateClient.onStatus.AddListener(rtpVideoReceiver.ApplyStatus);
            stateClient.onError.AddListener(flow.ApplyConnectionError);
            orchestrator.Bind(
                questClient,
                stateClient,
                videoBridge,
                rtpVideoReceiver,
                diagnosticsText,
                headVideoStatus,
                leftVideoStatus,
                rightVideoStatus,
                calibrationPanel,
                calibrationTitle,
                calibrationInstruction,
                calibrationProgress,
                calibrationProgressFill,
                new GameObject[]
                {
                    stageText.gameObject,
                    guidanceText.gameObject,
                    indicator.gameObject,
                    head.gameObject,
                    left.gameObject,
                    right.gameObject,
                    headVideoStatus.gameObject,
                    leftVideoStatus.gameObject,
                    rightVideoStatus.gameObject,
                    diagnosticsText.gameObject,
                });

            string host = PlayerPrefs.GetString(PlayerPrefsHost, DefaultLocalHost);
            int port = PlayerPrefs.GetInt(PlayerPrefsPort, DefaultLocalPort);
            bool tls = PlayerPrefs.GetInt(PlayerPrefsTls, 0) == 1;
            string token = PlayerPrefs.GetString(PlayerPrefsToken, DefaultLocalToken);
            bootstrap.BindSceneComponents(questClient, stateClient, flow, headsetCamera.transform);
            bootstrap.ConfigureEndpoint(host, port, tls, token);

            EnsureEventSystem(root);
            _ = eventManager;
            _ = teleoperationScene;
        }

        private static Transform CreateOperatorOrigin(Transform root)
        {
            GameObject origin = new GameObject("XR Origin");
            origin.transform.SetParent(root, false);
            origin.transform.localPosition = Vector3.zero;
            origin.transform.localRotation = Quaternion.identity;
            return origin.transform;
        }

        private static Camera CreateHeadsetCamera(Transform operatorOrigin)
        {
            Camera camera = Camera.main;
            if (camera == null)
            {
                GameObject cameraObject = new GameObject("Main Camera");
                cameraObject.tag = "MainCamera";
                cameraObject.transform.SetParent(operatorOrigin, false);
                cameraObject.transform.localPosition = new Vector3(0.0f, 1.6f, 0.0f);
                camera = cameraObject.AddComponent<Camera>();
                cameraObject.AddComponent<AudioListener>();
            }
            else if (camera.transform.parent == null)
            {
                camera.transform.SetParent(operatorOrigin, true);
            }

            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.0f, 0.0f, 0.0f, 0.0f);
            return camera;
        }

        private static void ConfigurePassthrough(Transform root, Camera headsetCamera)
        {
            if (FindObjectOfType<ARSession>() == null)
            {
                GameObject sessionObject = new GameObject("Quest Passthrough Session");
                sessionObject.transform.SetParent(root, false);
                sessionObject.AddComponent<ARSession>();
            }

            ARCameraManager cameraManager = headsetCamera.GetComponent<ARCameraManager>();
            if (cameraManager == null)
            {
                cameraManager = headsetCamera.gameObject.AddComponent<ARCameraManager>();
            }
            cameraManager.enabled = true;
        }

        private static Canvas CreateWorldCanvas(Transform headset)
        {
            GameObject canvasObject = new GameObject("Operator Status Canvas");
            canvasObject.transform.SetParent(headset, false);
            canvasObject.transform.localPosition = new Vector3(0.0f, 0.0f, 1.8f);
            canvasObject.transform.localRotation = Quaternion.identity;
            canvasObject.transform.localScale = Vector3.one * 0.0025f;

            Canvas canvas = canvasObject.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvasObject.AddComponent<GraphicRaycaster>();
            RectTransform rect = canvasObject.GetComponent<RectTransform>();
            rect.sizeDelta = new Vector2(900.0f, 760.0f);
            return canvas;
        }

        private static Text CreateText(Transform parent, string name, Vector2 position, int size, string value)
        {
            GameObject obj = new GameObject(name);
            obj.transform.SetParent(parent, false);
            Text text = obj.AddComponent<Text>();
            text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            text.fontSize = size;
            text.alignment = TextAnchor.MiddleCenter;
            text.color = Color.white;
            text.horizontalOverflow = HorizontalWrapMode.Wrap;
            text.verticalOverflow = VerticalWrapMode.Truncate;
            text.text = value;
            RectTransform rect = obj.GetComponent<RectTransform>();
            rect.anchoredPosition = position;
            rect.sizeDelta = new Vector2(780.0f, 48.0f);
            return text;
        }

        private static void ConfigureTextRect(Text text, Vector2 size, TextAnchor alignment)
        {
            text.alignment = alignment;
            RectTransform rect = text.GetComponent<RectTransform>();
            rect.sizeDelta = size;
        }

        private static Image CreateIndicator(Transform parent)
        {
            GameObject obj = new GameObject("StageIndicator");
            obj.transform.SetParent(parent, false);
            Image image = obj.AddComponent<Image>();
            image.color = Color.gray;
            RectTransform rect = obj.GetComponent<RectTransform>();
            rect.anchoredPosition = new Vector2(-410.0f, 155.0f);
            rect.sizeDelta = new Vector2(24.0f, 24.0f);
            return image;
        }

        private static GameObject CreateSuspensionOverlay(Transform parent)
        {
            GameObject obj = new GameObject("SuspensionOverlay");
            obj.transform.SetParent(parent, false);
            Image image = obj.AddComponent<Image>();
            image.color = new Color(0.3f, 0.02f, 0.02f, 0.65f);
            RectTransform rect = obj.GetComponent<RectTransform>();
            rect.anchoredPosition = new Vector2(0.0f, 0.0f);
            rect.sizeDelta = new Vector2(840.0f, 620.0f);
            obj.SetActive(false);
            return obj;
        }

        private static GameObject CreateCalibrationPanel(
            Transform parent,
            out Text title,
            out Text instruction,
            out Text progress,
            out Image progressFill)
        {
            GameObject panel = new GameObject("CalibrationGuidePanel");
            panel.transform.SetParent(parent, false);
            Image background = panel.AddComponent<Image>();
            background.color = new Color(0.02f, 0.04f, 0.06f, 0.78f);
            RectTransform panelRect = panel.GetComponent<RectTransform>();
            panelRect.anchoredPosition = new Vector2(0.0f, 0.0f);
            panelRect.sizeDelta = new Vector2(820.0f, 520.0f);

            title = CreateText(panel.transform, "CalibrationTitle", new Vector2(0.0f, 195.0f), 30, "VR calibration");
            ConfigureTextRect(title, new Vector2(740.0f, 56.0f), TextAnchor.MiddleCenter);

            instruction = CreateText(panel.transform, "CalibrationInstruction", new Vector2(0.0f, 112.0f), 23, "");
            ConfigureTextRect(instruction, new Vector2(720.0f, 96.0f), TextAnchor.MiddleCenter);

            progress = CreateText(panel.transform, "CalibrationProgressText", new Vector2(0.0f, -176.0f), 18, "");
            ConfigureTextRect(progress, new Vector2(720.0f, 62.0f), TextAnchor.MiddleCenter);

            GameObject bar = new GameObject("CalibrationProgressBar");
            bar.transform.SetParent(panel.transform, false);
            Image barBackground = bar.AddComponent<Image>();
            barBackground.color = new Color(1.0f, 1.0f, 1.0f, 0.14f);
            RectTransform barRect = bar.GetComponent<RectTransform>();
            barRect.anchoredPosition = new Vector2(0.0f, -222.0f);
            barRect.sizeDelta = new Vector2(660.0f, 18.0f);

            GameObject fill = new GameObject("CalibrationProgressFill");
            fill.transform.SetParent(bar.transform, false);
            progressFill = fill.AddComponent<Image>();
            progressFill.color = new Color(0.35f, 0.88f, 0.95f, 0.95f);
            progressFill.type = Image.Type.Filled;
            progressFill.fillMethod = Image.FillMethod.Horizontal;
            progressFill.fillOrigin = 0;
            progressFill.fillAmount = 0.0f;
            RectTransform fillRect = fill.GetComponent<RectTransform>();
            fillRect.anchorMin = Vector2.zero;
            fillRect.anchorMax = Vector2.one;
            fillRect.offsetMin = Vector2.zero;
            fillRect.offsetMax = Vector2.zero;

            panel.SetActive(false);
            return panel;
        }

        private static RawImage CreateVideoPanel(Transform parent, string name, Vector2 position, Vector2 size)
        {
            GameObject obj = new GameObject(name);
            obj.transform.SetParent(parent, false);
            RawImage image = obj.AddComponent<RawImage>();
            image.color = Color.white;
            RectTransform rect = obj.GetComponent<RectTransform>();
            rect.anchoredPosition = position;
            rect.sizeDelta = size;
            return image;
        }

        private static void EnsureEventSystem(Transform root)
        {
            if (FindObjectOfType<EventSystem>() != null)
            {
                return;
            }

            GameObject eventSystem = new GameObject("EventSystem");
            eventSystem.transform.SetParent(root, false);
            eventSystem.AddComponent<EventSystem>();
            eventSystem.AddComponent<StandaloneInputModule>();
        }
    }
}
