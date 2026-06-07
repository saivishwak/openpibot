using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.XR;

namespace XLeRobot.QuestTeleop
{
    public sealed class XLeRobotQuestClient : MonoBehaviour
    {
        [Header("Backend")]
        [SerializeField] private string workstationHost = "192.168.0.113";
        [SerializeField] private int workstationPort = 5000;
        [SerializeField] private bool useTls = false;
        [SerializeField] private string pairingToken = "dev-quest-token";

        [Header("Streaming")]
        [SerializeField] private float sendHz = 60.0f;
        [SerializeField] private Transform operatorOrigin;

        public UnityEvent<string> onRawServerMessage = new UnityEvent<string>();
        public UnityEvent<string> onConnectionState = new UnityEvent<string>();
        public UnityEvent<string> onError = new UnityEvent<string>();

        private ClientWebSocket socket;
        private CancellationTokenSource cancelSource;
        private float nextSendTime;
        private bool sendInFlight;
        private readonly List<InputDevice> leftDevices = new();
        private readonly List<InputDevice> rightDevices = new();
        private InputDevice leftController;
        private InputDevice rightController;
        private readonly Queue<string> pendingServerMessages = new();
        private readonly Queue<string> pendingConnectionStates = new();
        private readonly Queue<string> pendingErrors = new();
        private readonly object pendingLock = new();
        private bool operatorOriginCentered;

        public bool IsConnected => socket != null && socket.State == WebSocketState.Open;

        private Uri WebSocketUri
        {
            get
            {
                string scheme = useTls ? "wss" : "ws";
                string escapedToken = Uri.EscapeDataString(pairingToken ?? "");
                return new Uri($"{scheme}://{workstationHost}:{workstationPort}/api/vr/quest/ws?token={escapedToken}");
            }
        }

        private void OnEnable()
        {
            RefreshControllers();
            InputDevices.deviceConnected += OnDeviceChanged;
            InputDevices.deviceDisconnected += OnDeviceChanged;
            cancelSource = new CancellationTokenSource();
            _ = ConnectLoop(cancelSource.Token);
        }

        private async void OnDisable()
        {
            InputDevices.deviceConnected -= OnDeviceChanged;
            InputDevices.deviceDisconnected -= OnDeviceChanged;
            if (cancelSource != null)
            {
                cancelSource.Cancel();
                cancelSource.Dispose();
                cancelSource = null;
            }
            if (socket != null)
            {
                try
                {
                    if (socket.State == WebSocketState.Open)
                    {
                        await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "disabled", CancellationToken.None);
                    }
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"Quest teleop socket close failed: {e.Message}");
                }
                socket.Dispose();
                socket = null;
            }
        }

        private void Update()
        {
            DrainQueuedEvents();
            if (Time.unscaledTime < nextSendTime)
            {
                return;
            }
            nextSendTime = Time.unscaledTime + 1.0f / Mathf.Max(1.0f, sendHz);

            if (socket == null || socket.State != WebSocketState.Open || sendInFlight)
            {
                return;
            }

            _ = SendSnapshotAsync();
        }

        public void ConfigureEndpoint(string host, int port, bool tls, string token = "")
        {
            workstationHost = host;
            workstationPort = port;
            useTls = tls;
            pairingToken = token ?? "";
        }

        public void BindOperatorOrigin(Transform origin)
        {
            if (operatorOrigin != origin)
            {
                operatorOriginCentered = false;
            }
            operatorOrigin = origin;
        }

        public void RecenterOperatorOrigin(Transform headset)
        {
            if (operatorOrigin == null || headset == null)
            {
                return;
            }
            if (operatorOriginCentered)
            {
                Debug.Log("Quest operator origin already centered for this app session; preserving calibration frame.");
                return;
            }

            Vector3 yawOnly = headset.rotation.eulerAngles;
            operatorOrigin.rotation = Quaternion.Euler(0.0f, yawOnly.y, 0.0f);

            // Reachy2-style mirror recenter: user origin follows head yaw and is
            // placed slightly behind/below the headset, near shoulder height.
            Vector3 headPosition = headset.position - headset.forward * 0.10f;
            operatorOrigin.position = new Vector3(headPosition.x, headPosition.y - 0.15f, headPosition.z);
            operatorOriginCentered = true;
        }

        private async Task ConnectLoop(CancellationToken token)
        {
            while (!token.IsCancellationRequested)
            {
                try
                {
                    if (socket != null && socket.State == WebSocketState.Open)
                    {
                        await Task.Delay(TimeSpan.FromSeconds(1), token);
                        continue;
                    }
                    socket?.Dispose();
                    socket = new ClientWebSocket();
                    await socket.ConnectAsync(WebSocketUri, token);
                    EnqueueConnectionState("connected");
                    _ = ReceiveLoop(socket, token);
                    Debug.Log($"Connected Quest teleop socket: {WebSocketUri}");
                }
                catch (OperationCanceledException)
                {
                    return;
                }
                catch (Exception e)
                {
                    EnqueueConnectionState("disconnected");
                    EnqueueError($"Quest teleop socket connect failed: {e.Message}");
                    Debug.LogWarning($"Quest teleop socket connect failed: {e.Message}");
                    await Task.Delay(TimeSpan.FromSeconds(2), token);
                }
            }
        }

        private async Task ReceiveLoop(ClientWebSocket activeSocket, CancellationToken token)
        {
            byte[] buffer = new byte[8192];
            while (!token.IsCancellationRequested && activeSocket == socket && activeSocket.State == WebSocketState.Open)
            {
                try
                {
                    using MemoryStream message = new();
                    WebSocketReceiveResult result;
                    do
                    {
                        result = await activeSocket.ReceiveAsync(new ArraySegment<byte>(buffer), token);
                        if (result.MessageType == WebSocketMessageType.Close)
                        {
                            EnqueueConnectionState("disconnected");
                            activeSocket.Abort();
                            return;
                        }
                        message.Write(buffer, 0, result.Count);
                    }
                    while (!result.EndOfMessage);

                    if (result.MessageType == WebSocketMessageType.Text)
                    {
                        EnqueueServerMessage(Encoding.UTF8.GetString(message.ToArray()));
                    }
                }
                catch (OperationCanceledException)
                {
                    return;
                }
                catch (Exception e)
                {
                    EnqueueConnectionState("disconnected");
                    EnqueueError($"Quest teleop socket receive failed: {e.Message}");
                    if (activeSocket == socket)
                    {
                        socket?.Abort();
                    }
                    return;
                }
            }
        }

        private async Task SendSnapshotAsync()
        {
            sendInFlight = true;
            try
            {
                string payload = BuildPacketJson();
                byte[] bytes = Encoding.UTF8.GetBytes(payload);
                await socket.SendAsync(bytes, WebSocketMessageType.Text, true, CancellationToken.None);
            }
            catch (Exception e)
            {
                EnqueueConnectionState("disconnected");
                EnqueueError($"Quest teleop send failed: {e.Message}");
                Debug.LogWarning($"Quest teleop send failed: {e.Message}");
                socket?.Abort();
                socket?.Dispose();
                socket = null;
            }
            finally
            {
                sendInFlight = false;
            }
        }

        private void EnqueueServerMessage(string message)
        {
            lock (pendingLock)
            {
                pendingServerMessages.Enqueue(message ?? "");
            }
        }

        private void EnqueueConnectionState(string state)
        {
            lock (pendingLock)
            {
                pendingConnectionStates.Enqueue(state ?? "");
            }
        }

        private void EnqueueError(string error)
        {
            lock (pendingLock)
            {
                pendingErrors.Enqueue(error ?? "");
            }
        }

        private void DrainQueuedEvents()
        {
            while (true)
            {
                string message = null;
                string state = null;
                string error = null;
                lock (pendingLock)
                {
                    if (pendingServerMessages.Count > 0)
                    {
                        message = pendingServerMessages.Dequeue();
                    }
                    else if (pendingConnectionStates.Count > 0)
                    {
                        state = pendingConnectionStates.Dequeue();
                    }
                    else if (pendingErrors.Count > 0)
                    {
                        error = pendingErrors.Dequeue();
                    }
                    else
                    {
                        return;
                    }
                }

                if (message != null)
                {
                    onRawServerMessage.Invoke(message);
                }
                if (state != null)
                {
                    onConnectionState.Invoke(state);
                }
                if (error != null)
                {
                    onError.Invoke(error);
                }
            }
        }

        private string BuildPacketJson()
        {
            RefreshControllers();
            StringBuilder sb = new();
            sb.Append("{\"timestamp\":");
            sb.Append(Time.realtimeSinceStartupAsDouble.ToString("F6", CultureInfo.InvariantCulture));
            sb.Append(",\"controllers\":{");
            AppendController(sb, "left", leftController, true);
            sb.Append(",");
            AppendController(sb, "right", rightController, false);
            sb.Append("}}");
            return sb.ToString();
        }

        private void AppendController(StringBuilder sb, string side, InputDevice device, bool left)
        {
            Vector3 position = Vector3.zero;
            Quaternion rotation = Quaternion.identity;
            float trigger = 0.0f;
            float grip = 0.0f;
            Vector2 thumbstick = Vector2.zero;
            bool primary = false;
            bool secondary = false;
            bool valid = device.isValid;

            if (valid)
            {
                device.TryGetFeatureValue(CommonUsages.devicePosition, out position);
                device.TryGetFeatureValue(CommonUsages.deviceRotation, out rotation);
                device.TryGetFeatureValue(CommonUsages.trigger, out trigger);
                device.TryGetFeatureValue(CommonUsages.grip, out grip);
                device.TryGetFeatureValue(CommonUsages.primary2DAxis, out thumbstick);
                device.TryGetFeatureValue(CommonUsages.primaryButton, out primary);
                device.TryGetFeatureValue(CommonUsages.secondaryButton, out secondary);
            }

            if (operatorOrigin != null)
            {
                position = Quaternion.Inverse(operatorOrigin.rotation) * (position - operatorOrigin.position);
                rotation = Quaternion.Inverse(operatorOrigin.rotation) * rotation;
            }

            if (XLeRobotPassthroughHoldController.PassthroughHeld ||
                XLeRobotPassthroughHoldController.IsPassthroughChordPressed())
            {
                trigger = 0.0f;
            }

            sb.Append("\"");
            sb.Append(side);
            sb.Append("\":{");
            sb.Append("\"valid\":");
            sb.Append(valid ? "true" : "false");
            sb.Append(",\"position\":");
            AppendVector3(sb, position);
            sb.Append(",\"rotation\":");
            AppendQuaternion(sb, rotation);
            sb.Append(",\"grip\":");
            sb.Append(grip > 0.5f ? "true" : "false");
            sb.Append(",\"trigger\":");
            sb.Append(trigger.ToString("F4", CultureInfo.InvariantCulture));
            sb.Append(",\"thumbstick\":{\"x\":");
            sb.Append(thumbstick.x.ToString("F4", CultureInfo.InvariantCulture));
            sb.Append(",\"y\":");
            sb.Append(thumbstick.y.ToString("F4", CultureInfo.InvariantCulture));
            sb.Append("},\"buttons\":{");
            if (left)
            {
                sb.Append("\"X\":");
                sb.Append(primary ? "true" : "false");
                sb.Append(",\"Y\":");
                sb.Append(secondary ? "true" : "false");
            }
            else
            {
                sb.Append("\"A\":");
                sb.Append(primary ? "true" : "false");
                sb.Append(",\"B\":");
                sb.Append(secondary ? "true" : "false");
            }
            sb.Append("}}");
        }

        private static void AppendVector3(StringBuilder sb, Vector3 v)
        {
            sb.Append("[");
            sb.Append(v.x.ToString("F6", CultureInfo.InvariantCulture));
            sb.Append(",");
            sb.Append(v.y.ToString("F6", CultureInfo.InvariantCulture));
            sb.Append(",");
            sb.Append(v.z.ToString("F6", CultureInfo.InvariantCulture));
            sb.Append("]");
        }

        private static void AppendQuaternion(StringBuilder sb, Quaternion q)
        {
            sb.Append("[");
            sb.Append(q.x.ToString("F6", CultureInfo.InvariantCulture));
            sb.Append(",");
            sb.Append(q.y.ToString("F6", CultureInfo.InvariantCulture));
            sb.Append(",");
            sb.Append(q.z.ToString("F6", CultureInfo.InvariantCulture));
            sb.Append(",");
            sb.Append(q.w.ToString("F6", CultureInfo.InvariantCulture));
            sb.Append("]");
        }

        private void OnDeviceChanged(InputDevice _)
        {
            RefreshControllers();
        }

        private void RefreshControllers()
        {
            InputDevices.GetDevicesAtXRNode(XRNode.LeftHand, leftDevices);
            InputDevices.GetDevicesAtXRNode(XRNode.RightHand, rightDevices);
            if (leftDevices.Count > 0)
            {
                leftController = leftDevices[0];
            }
            if (rightDevices.Count > 0)
            {
                rightController = rightDevices[0];
            }
        }
    }
}
