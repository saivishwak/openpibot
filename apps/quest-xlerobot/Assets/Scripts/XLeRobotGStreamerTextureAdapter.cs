using UnityEngine;

namespace XLeRobot.QuestTeleop
{
    public sealed class XLeRobotGStreamerTextureAdapter : MonoBehaviour
    {
        [SerializeField] private XLeRobotVideoSurfaceBinder surfaceBinder;

        public void Bind(XLeRobotVideoSurfaceBinder binder)
        {
            surfaceBinder = binder;
        }

        public void OnHeadTextureReady(Texture texture)
        {
            if (surfaceBinder != null)
            {
                surfaceBinder.SetHeadTexture(texture);
            }
        }

        public void OnLeftTextureReady(Texture texture)
        {
            if (surfaceBinder != null)
            {
                surfaceBinder.SetLeftWristTexture(texture);
            }
        }

        public void OnRightTextureReady(Texture texture)
        {
            if (surfaceBinder != null)
            {
                surfaceBinder.SetRightWristTexture(texture);
            }
        }
    }
}
