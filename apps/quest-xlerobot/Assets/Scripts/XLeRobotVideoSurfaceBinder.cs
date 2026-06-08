using UnityEngine;
using UnityEngine.UI;

namespace XLeRobot.QuestTeleop
{
    public sealed class XLeRobotVideoSurfaceBinder : MonoBehaviour
    {
        [Header("RawImage Targets")]
        [SerializeField] private RawImage headImage;
        [SerializeField] private RawImage leftWristImage;
        [SerializeField] private RawImage rightWristImage;

        [Header("Renderer Targets")]
        [SerializeField] private Renderer headRenderer;
        [SerializeField] private Renderer leftWristRenderer;
        [SerializeField] private Renderer rightWristRenderer;
        [SerializeField] private bool flipHeadHorizontally = true;
        [SerializeField] private bool flipLeftWristHorizontally = true;
        [SerializeField] private bool flipRightWristHorizontally = true;

        public void BindRawImages(RawImage head, RawImage leftWrist, RawImage rightWrist)
        {
            headImage = head;
            leftWristImage = leftWrist;
            rightWristImage = rightWrist;
        }

        public void BindRenderers(Renderer head, Renderer leftWrist, Renderer rightWrist)
        {
            headRenderer = head;
            leftWristRenderer = leftWrist;
            rightWristRenderer = rightWrist;
        }

        public void SetHeadTexture(Texture texture)
        {
            Apply(headImage, headRenderer, texture, flipHeadHorizontally);
        }

        public void SetLeftWristTexture(Texture texture)
        {
            Apply(leftWristImage, leftWristRenderer, texture, flipLeftWristHorizontally);
        }

        public void SetRightWristTexture(Texture texture)
        {
            Apply(rightWristImage, rightWristRenderer, texture, flipRightWristHorizontally);
        }

        private static void Apply(RawImage image, Renderer rendererTarget, Texture texture, bool flipHorizontally)
        {
            if (image != null)
            {
                image.texture = texture;
                image.color = Color.white;
                image.uvRect = flipHorizontally
                    ? new Rect(1.0f, 0.0f, -1.0f, 1.0f)
                    : new Rect(0.0f, 0.0f, 1.0f, 1.0f);
            }
            if (rendererTarget != null && rendererTarget.material != null)
            {
                rendererTarget.material.mainTexture = texture;
                rendererTarget.material.color = Color.white;
                rendererTarget.material.mainTextureScale = flipHorizontally
                    ? new Vector2(-1.0f, 1.0f)
                    : Vector2.one;
                rendererTarget.material.mainTextureOffset = flipHorizontally
                    ? new Vector2(1.0f, 0.0f)
                    : Vector2.zero;
            }
        }
    }
}
