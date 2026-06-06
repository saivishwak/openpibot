using UnityEngine;
using UnityEngine.Events;

namespace XLeRobot.QuestTeleop
{
    /// <summary>
    /// Android/Quest RTP H.264 receiver for the backend GStreamer UDP streams.
    ///
    /// Backend streams are produced by `gst-launch-1.0 ... rtph264pay ! udpsink`
    /// on consecutive UDP ports. On Android this component delegates RTP
    /// depacketization and MediaCodec decode to `RtpH264Receiver`, then uploads
    /// decoded RGBA frames into Unity `Texture2D` objects.
    /// </summary>
    public sealed class XLeRobotAndroidRtpVideoReceiver : MonoBehaviour
    {
        [SerializeField] private int basePort = 5600;
        [SerializeField] private int width = 640;
        [SerializeField] private int height = 480;
        [SerializeField] private bool startOnEnable = true;
        [SerializeField] private XLeRobotGStreamerTextureAdapter textureAdapter;
        [SerializeField] private XLeRobotVideoBridgeClient videoBridgeClient;

        public UnityEvent<XLeRobotVideoReceiveStatus> onReceiveStatus = new UnityEvent<XLeRobotVideoReceiveStatus>();

#if UNITY_ANDROID && !UNITY_EDITOR
        private AndroidJavaObject headReceiver;
        private AndroidJavaObject leftReceiver;
        private AndroidJavaObject rightReceiver;
#endif
        private Texture2D headTexture;
        private Texture2D leftTexture;
        private Texture2D rightTexture;
        private int headFrames;
        private int leftFrames;
        private int rightFrames;
        private float nextHeadHealthReport;
        private float nextLeftHealthReport;
        private float nextRightHealthReport;

        public int BasePort => basePort;
        public int HeadFrames => headFrames;
        public int LeftFrames => leftFrames;
        public int RightFrames => rightFrames;

        public void Bind(
            XLeRobotGStreamerTextureAdapter adapter,
            XLeRobotVideoBridgeClient bridgeClient,
            int configuredBasePort = 5600)
        {
            textureAdapter = adapter;
            videoBridgeClient = bridgeClient;
            basePort = configuredBasePort;
        }

        private void OnEnable()
        {
            if (startOnEnable)
            {
                StartReceivers();
            }
        }

        private void OnDisable()
        {
            StopReceivers();
        }

        public void StartReceivers()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            StopReceivers();
            headReceiver = StartReceiver(basePort);
            leftReceiver = StartReceiver(basePort + 1);
            rightReceiver = StartReceiver(basePort + 2);
            EmitReceiveStatus("head", "listening", basePort, true, headFrames, "");
            EmitReceiveStatus("left_wrist", "listening", basePort + 1, true, leftFrames, "");
            EmitReceiveStatus("right_wrist", "listening", basePort + 2, true, rightFrames, "");
#else
            Debug.Log("XLeRobot RTP video receiver is active only on Android Quest builds.");
            EmitReceiveStatus("head", "android_only", basePort, false, headFrames, "RTP receiver runs only in Android Quest builds");
            EmitReceiveStatus("left_wrist", "android_only", basePort + 1, false, leftFrames, "RTP receiver runs only in Android Quest builds");
            EmitReceiveStatus("right_wrist", "android_only", basePort + 2, false, rightFrames, "RTP receiver runs only in Android Quest builds");
#endif
        }

        public void ApplyStatus(XLeRobotQuestStatus status)
        {
            if (status == null || status.video_base_port <= 0 || status.video_base_port == basePort)
            {
                return;
            }
            basePort = status.video_base_port;
            if (enabled)
            {
                StartReceivers();
            }
        }

        public void StopReceivers()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            StopReceiver(headReceiver);
            StopReceiver(leftReceiver);
            StopReceiver(rightReceiver);
            headReceiver = null;
            leftReceiver = null;
            rightReceiver = null;
#endif
            EmitReceiveStatus("head", "stopped", basePort, false, headFrames, "");
            EmitReceiveStatus("left_wrist", "stopped", basePort + 1, false, leftFrames, "");
            EmitReceiveStatus("right_wrist", "stopped", basePort + 2, false, rightFrames, "");
        }

        private void Update()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            UpdateReceiver(headReceiver, "head", ref headTexture, ref headFrames);
            UpdateReceiver(leftReceiver, "left_wrist", ref leftTexture, ref leftFrames);
            UpdateReceiver(rightReceiver, "right_wrist", ref rightTexture, ref rightFrames);
#endif
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        private AndroidJavaObject StartReceiver(int port)
        {
            AndroidJavaObject receiver = new AndroidJavaObject("com.openpibot.questteleop.RtpH264Receiver");
            receiver.Call("start", port, width, height);
            return receiver;
        }

        private static void StopReceiver(AndroidJavaObject receiver)
        {
            if (receiver == null)
            {
                return;
            }
            receiver.Call("stop");
            receiver.Dispose();
        }

        private void UpdateReceiver(
            AndroidJavaObject receiver,
            string role,
            ref Texture2D texture,
            ref int lastFrameCount)
        {
            if (receiver == null)
            {
                return;
            }

            int frameCount = receiver.Call<int>("getFrameCount");
            if (frameCount <= lastFrameCount)
            {
                ReportHealthIfDue(receiver, role, frameCount);
                return;
            }

            byte[] rgba = receiver.Call<byte[]>("getLatestRgba");
            if (rgba == null || rgba.Length == 0)
            {
                return;
            }

            int decodedWidth = receiver.Call<int>("getFrameWidth");
            int decodedHeight = receiver.Call<int>("getFrameHeight");
            if (texture == null || texture.width != decodedWidth || texture.height != decodedHeight)
            {
                texture = new Texture2D(decodedWidth, decodedHeight, TextureFormat.RGBA32, false);
                ApplyTexture(role, texture);
            }

            texture.LoadRawTextureData(rgba);
            texture.Apply(false);
            lastFrameCount = frameCount;
            ReportHealthIfDue(receiver, role, frameCount);
        }

        private void ReportHealthIfDue(AndroidJavaObject receiver, string role, int frameCount)
        {
            if (videoBridgeClient == null)
            {
                return;
            }
            float now = Time.unscaledTime;
            if (role == "head" && now < nextHeadHealthReport) return;
            if (role == "left_wrist" && now < nextLeftHealthReport) return;
            if (role == "right_wrist" && now < nextRightHealthReport) return;
            if (role == "head") nextHeadHealthReport = now + 2.0f;
            if (role == "left_wrist") nextLeftHealthReport = now + 2.0f;
            if (role == "right_wrist") nextRightHealthReport = now + 2.0f;
            string error = receiver.Call<string>("getLastError") ?? "";
            bool running = receiver.Call<bool>("isRunning");
            string state = running ? (frameCount > 0 ? "receiving" : "waiting_frames") : "stopped";
            EmitReceiveStatus(role, state, PortForRole(role), running, frameCount, error);
            videoBridgeClient.ReportReceiveHealth(
                role,
                state,
                0.0f,
                0.0f,
                frameCount,
                error);
        }
#endif

        private int PortForRole(string role)
        {
            switch (role)
            {
                case "left_wrist": return basePort + 1;
                case "right_wrist": return basePort + 2;
                default: return basePort;
            }
        }

        private void EmitReceiveStatus(string role, string state, int port, bool running, int frames, string error)
        {
            onReceiveStatus.Invoke(new XLeRobotVideoReceiveStatus
            {
                role = role,
                state = state,
                port = port,
                running = running,
                frames = frames,
                error = error ?? ""
            });
        }

        private void ApplyTexture(string role, Texture texture)
        {
            if (textureAdapter == null)
            {
                return;
            }
            switch (role)
            {
                case "head":
                    textureAdapter.OnHeadTextureReady(texture);
                    break;
                case "left_wrist":
                    textureAdapter.OnLeftTextureReady(texture);
                    break;
                case "right_wrist":
                    textureAdapter.OnRightTextureReady(texture);
                    break;
            }
        }
    }
}
