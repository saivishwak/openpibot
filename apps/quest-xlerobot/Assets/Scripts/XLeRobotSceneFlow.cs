using System.Collections.Generic;
using System.Collections;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Networking;
using UnityEngine.XR;

namespace XLeRobot.QuestTeleop
{
    public enum XLeRobotQuestEvent
    {
        EnterMirrorScene,
        EnterTeleoperationScene,
        SuspendTeleoperation,
        ResumeTeleoperation,
        EmergencyStop
    }

    public sealed class XLeRobotEventManager : MonoBehaviour
    {
        private static readonly Dictionary<XLeRobotQuestEvent, UnityEvent> Events = new Dictionary<XLeRobotQuestEvent, UnityEvent>();

        public static void StartListening(XLeRobotQuestEvent eventName, UnityAction listener)
        {
            if (!Events.TryGetValue(eventName, out UnityEvent evt))
            {
                evt = new UnityEvent();
                Events[eventName] = evt;
            }
            evt.AddListener(listener);
        }

        public static void StopListening(XLeRobotQuestEvent eventName, UnityAction listener)
        {
            if (Events.TryGetValue(eventName, out UnityEvent evt))
            {
                evt.RemoveListener(listener);
            }
        }

        public static void Trigger(XLeRobotQuestEvent eventName)
        {
            if (Events.TryGetValue(eventName, out UnityEvent evt))
            {
                evt.Invoke();
            }
        }
    }

    public sealed class XLeRobotScenesManager : MonoBehaviour
    {
        public UnityEvent onMirrorScene = new UnityEvent();
        public UnityEvent onTeleoperationScene = new UnityEvent();
        public UnityEvent onSuspended = new UnityEvent();

        public XLeRobotOperatorStage CurrentStage { get; private set; } = XLeRobotOperatorStage.Unknown;

        public void ApplyStatus(XLeRobotQuestStatus status)
        {
            if (status == null || status.Stage == CurrentStage)
            {
                return;
            }

            CurrentStage = status.Stage;
            switch (CurrentStage)
            {
                case XLeRobotOperatorStage.MirrorReady:
                    XLeRobotEventManager.Trigger(XLeRobotQuestEvent.EnterMirrorScene);
                    onMirrorScene.Invoke();
                    break;
                case XLeRobotOperatorStage.TeleopHeadOnly:
                case XLeRobotOperatorStage.TeleopArms:
                    XLeRobotEventManager.Trigger(XLeRobotQuestEvent.EnterTeleoperationScene);
                    onTeleoperationScene.Invoke();
                    break;
                case XLeRobotOperatorStage.Suspended:
                    XLeRobotEventManager.Trigger(XLeRobotQuestEvent.SuspendTeleoperation);
                    onSuspended.Invoke();
                    break;
            }
        }
    }

    public sealed class XLeRobotUserTrackerManager : MonoBehaviour
    {
        [SerializeField] private Transform headset;

        public Vector3 HeadPosition => headset != null ? headset.position : Vector3.zero;
        public Quaternion HeadRotation => headset != null ? headset.rotation : Quaternion.identity;

        public void BindHeadset(Transform headsetTransform)
        {
            headset = headsetTransform;
        }
    }

    public sealed class XLeRobotMirrorSceneManager : MonoBehaviour
    {
        [SerializeField] private XLeRobotQuestBootstrap bootstrap;

        public UnityEvent onMirrorReadyConfirmed = new UnityEvent();

        public void Bind(XLeRobotQuestBootstrap questBootstrap)
        {
            bootstrap = questBootstrap;
        }

        public void ConfirmMirrorReady()
        {
            if (bootstrap != null)
            {
                bootstrap.Recenter();
            }
            onMirrorReadyConfirmed.Invoke();
        }
    }

    public sealed class XLeRobotTeleoperationSceneManager : MonoBehaviour
    {
        public bool TeleoperationVisible { get; private set; }

        private void OnEnable()
        {
            XLeRobotEventManager.StartListening(XLeRobotQuestEvent.EnterTeleoperationScene, ShowTeleoperation);
            XLeRobotEventManager.StartListening(XLeRobotQuestEvent.SuspendTeleoperation, HideTeleoperation);
        }

        private void OnDisable()
        {
            XLeRobotEventManager.StopListening(XLeRobotQuestEvent.EnterTeleoperationScene, ShowTeleoperation);
            XLeRobotEventManager.StopListening(XLeRobotQuestEvent.SuspendTeleoperation, HideTeleoperation);
        }

        private void ShowTeleoperation()
        {
            TeleoperationVisible = true;
        }

        private void HideTeleoperation()
        {
            TeleoperationVisible = false;
        }
    }

    public sealed class XLeRobotSuspensionUIManager : MonoBehaviour
    {
        [SerializeField] private GameObject overlay;

        public void BindOverlay(GameObject overlayObject)
        {
            overlay = overlayObject;
        }

        public void ApplyStatus(XLeRobotQuestStatus status)
        {
            bool suspended = status == null || status.Stage == XLeRobotOperatorStage.Suspended;
            if (overlay != null)
            {
                overlay.SetActive(suspended);
            }
        }
    }

    public sealed class XLeRobotEmergencyStopInput : MonoBehaviour
    {
        [SerializeField] private XLeRobotStateClient stateClient;
        [SerializeField] private float holdSeconds = 0.75f;

        private readonly List<InputDevice> devices = new List<InputDevice>();
        private float heldSince = -1.0f;
        private bool sent;

        public void Bind(XLeRobotStateClient client)
        {
            stateClient = client;
        }

        private void Update()
        {
            InputDevices.GetDevicesWithCharacteristics(InputDeviceCharacteristics.Controller, devices);
            bool chordHeld = false;
            foreach (InputDevice device in devices)
            {
                bool primary = false;
                bool secondary = false;
                device.TryGetFeatureValue(CommonUsages.primaryButton, out primary);
                device.TryGetFeatureValue(CommonUsages.secondaryButton, out secondary);
                chordHeld |= primary && secondary;
            }

            if (!chordHeld)
            {
                heldSince = -1.0f;
                sent = false;
                return;
            }

            if (heldSince < 0.0f)
            {
                heldSince = Time.unscaledTime;
            }
            if (!sent && Time.unscaledTime - heldSince >= holdSeconds)
            {
                sent = true;
                XLeRobotEventManager.Trigger(XLeRobotQuestEvent.EmergencyStop);
                StartCoroutine(PostEmergencyStop());
            }
        }

        private IEnumerator PostEmergencyStop()
        {
            if (stateClient == null)
            {
                yield break;
            }

            using UnityWebRequest req = new UnityWebRequest(stateClient.Settings.HttpBaseUrl + "/api/vr/emergency_stop", "POST");
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("x-quest-pairing-token", stateClient.Settings.pairingToken);
            yield return req.SendWebRequest();
        }
    }
}
