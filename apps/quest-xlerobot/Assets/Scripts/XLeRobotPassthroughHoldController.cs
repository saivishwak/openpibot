using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.ARFoundation;

namespace XLeRobot.QuestTeleop
{
    public sealed class XLeRobotPassthroughHoldController : MonoBehaviour
    {
        [SerializeField] private float pressThreshold = 0.72f;
        [SerializeField] private float releaseThreshold = 0.45f;
        [SerializeField] private ARSession arSession;
        [SerializeField] private ARCameraManager cameraManager;
        [SerializeField] private ARCameraBackground cameraBackground;
        [SerializeField] private Camera headsetCamera;
        [SerializeField] private GameObject passthroughOverlay;

        private GameObject[] normalSceneObjects = new GameObject[0];
        private readonly Dictionary<GameObject, bool> previousActiveStates = new Dictionary<GameObject, bool>();
        private bool passthroughActive;

        public static bool PassthroughHeld { get; private set; }

        public static bool IsPassthroughChordPressed(float threshold = 0.72f)
        {
            return ReadTrigger(XRNode.LeftHand) > threshold && ReadTrigger(XRNode.RightHand) > threshold;
        }

        public void Bind(
            ARSession session,
            ARCameraManager manager,
            ARCameraBackground background,
            Camera camera,
            GameObject[] sceneObjects,
            GameObject overlay)
        {
            arSession = session;
            cameraManager = manager;
            cameraBackground = background;
            headsetCamera = camera;
            normalSceneObjects = sceneObjects ?? new GameObject[0];
            passthroughOverlay = overlay;
            SetPassthrough(false, true);
        }

        private void OnDisable()
        {
            SetPassthrough(false, true);
        }

        private void Update()
        {
            float leftTrigger = ReadTrigger(XRNode.LeftHand);
            float rightTrigger = ReadTrigger(XRNode.RightHand);
            bool shouldShow = passthroughActive
                ? leftTrigger > releaseThreshold && rightTrigger > releaseThreshold
                : leftTrigger > pressThreshold && rightTrigger > pressThreshold;
            SetPassthrough(shouldShow, false);
        }

        private static float ReadTrigger(XRNode node)
        {
            InputDevice device = InputDevices.GetDeviceAtXRNode(node);
            if (!device.isValid)
            {
                return 0.0f;
            }
            return device.TryGetFeatureValue(CommonUsages.trigger, out float value) ? value : 0.0f;
        }

        private void SetPassthrough(bool enabled, bool force)
        {
            if (!force && enabled == passthroughActive)
            {
                PassthroughHeld = enabled;
                return;
            }

            passthroughActive = enabled;
            PassthroughHeld = enabled;

            if (enabled)
            {
                previousActiveStates.Clear();
                foreach (GameObject obj in normalSceneObjects)
                {
                    if (obj == null || previousActiveStates.ContainsKey(obj))
                    {
                        continue;
                    }
                    previousActiveStates[obj] = obj.activeSelf;
                    obj.SetActive(false);
                }
            }
            else
            {
                foreach (KeyValuePair<GameObject, bool> entry in previousActiveStates)
                {
                    if (entry.Key != null)
                    {
                        entry.Key.SetActive(entry.Value);
                    }
                }
                previousActiveStates.Clear();
            }

            if (arSession != null)
            {
                arSession.gameObject.SetActive(enabled);
                arSession.enabled = enabled;
            }
            if (cameraManager != null)
            {
                cameraManager.enabled = enabled;
            }
            if (cameraBackground != null)
            {
                cameraBackground.enabled = enabled;
            }
            if (headsetCamera != null)
            {
                headsetCamera.clearFlags = CameraClearFlags.SolidColor;
                headsetCamera.backgroundColor = enabled
                    ? new Color(0.0f, 0.0f, 0.0f, 0.0f)
                    : new Color(0.0f, 0.0f, 0.0f, 1.0f);
            }
            if (passthroughOverlay != null)
            {
                passthroughOverlay.SetActive(enabled);
            }
        }
    }
}
