using System.Reflection;
using UnityEngine;
using UnityEngine.Events;

namespace XLeRobot.QuestTeleop
{
    /// <summary>
    /// Adapter for the optional Pollen/Reachy GstreamerWebRTC Unity package.
    ///
    /// The package API has changed across branches, so this class avoids a hard
    /// compile-time dependency on concrete plugin types. If a scene contains a
    /// component with UnityEvent<Texture> fields such as
    /// `event_LeftVideoTextureReady`, `event_RightVideoTextureReady`, or
    /// `event_HeadVideoTextureReady`, they are connected to XLeRobot's video
    /// surface binder.
    /// </summary>
    public sealed class XLeRobotGStreamerWebRtcReceiver : MonoBehaviour
    {
        [SerializeField] private Component gstreamerPlugin;
        [SerializeField] private XLeRobotGStreamerTextureAdapter textureAdapter;
        [SerializeField] private XLeRobotVideoBridgeClient videoBridgeClient;

        public void Bind(
            Component plugin,
            XLeRobotGStreamerTextureAdapter adapter,
            XLeRobotVideoBridgeClient bridgeClient)
        {
            gstreamerPlugin = plugin;
            textureAdapter = adapter;
            videoBridgeClient = bridgeClient;
            WireTextureEvents();
        }

        private void OnEnable()
        {
            WireTextureEvents();
        }

        private void WireTextureEvents()
        {
            if (gstreamerPlugin == null || textureAdapter == null)
            {
                return;
            }

            AddTextureListener("event_HeadVideoTextureReady", textureAdapter.OnHeadTextureReady);
            AddTextureListener("event_LeftVideoTextureReady", textureAdapter.OnLeftTextureReady);
            AddTextureListener("event_RightVideoTextureReady", textureAdapter.OnRightTextureReady);
            videoBridgeClient?.ReportReceiveHealth("head", "plugin_bound", 0.0f, 0.0f, 0);
        }

        private void AddTextureListener(string eventFieldName, UnityAction<Texture> listener)
        {
            FieldInfo field = gstreamerPlugin.GetType().GetField(
                eventFieldName,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (field?.GetValue(gstreamerPlugin) is UnityEvent<Texture> evt)
            {
                evt.RemoveListener(listener);
                evt.AddListener(listener);
            }
        }
    }
}
