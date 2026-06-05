using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

namespace XLeRobot.QuestTeleop
{
    public sealed class XLeRobotOperatorFlowManager : MonoBehaviour
    {
        [Header("Status UI")]
        [SerializeField] private Text stageText;
        [SerializeField] private Text guidanceText;
        [SerializeField] private Image stageIndicator;

        [Header("Flow Events")]
        public UnityEvent onConnectionRequired = new UnityEvent();
        public UnityEvent onMirrorReady = new UnityEvent();
        public UnityEvent onTeleopActive = new UnityEvent();
        public UnityEvent onSuspended = new UnityEvent();

        public XLeRobotOperatorStage CurrentStage { get; private set; } = XLeRobotOperatorStage.Unknown;

        public void BindUi(Text stage, Text guidance, Image indicator)
        {
            stageText = stage;
            guidanceText = guidance;
            stageIndicator = indicator;
        }

        public void ApplyStatus(XLeRobotQuestStatus status)
        {
            if (status == null)
            {
                return;
            }

            XLeRobotOperatorStage stage = status.Stage;
            CurrentStage = stage;
            if (stageText != null)
            {
                stageText.text = HumanizeStage(stage);
            }
            if (guidanceText != null)
            {
                guidanceText.text = string.IsNullOrEmpty(status.guidance) ? status.LastError : status.guidance;
            }
            if (stageIndicator != null)
            {
                stageIndicator.color = StageColor(stage);
            }

            switch (stage)
            {
                case XLeRobotOperatorStage.ConnectRequired:
                    onConnectionRequired.Invoke();
                    break;
                case XLeRobotOperatorStage.MirrorReady:
                    onMirrorReady.Invoke();
                    break;
                case XLeRobotOperatorStage.TeleopHeadOnly:
                case XLeRobotOperatorStage.TeleopArms:
                    onTeleopActive.Invoke();
                    break;
                case XLeRobotOperatorStage.Suspended:
                    onSuspended.Invoke();
                    break;
            }
        }

        public void ApplyConnectionError(string error)
        {
            if (stageText != null)
            {
                stageText.text = "Connection";
            }
            if (guidanceText != null)
            {
                guidanceText.text = string.IsNullOrEmpty(error) ? "Waiting for backend status..." : error;
            }
            if (stageIndicator != null)
            {
                stageIndicator.color = StageColor(XLeRobotOperatorStage.ConnectRequired);
            }
        }

        private static string HumanizeStage(XLeRobotOperatorStage stage)
        {
            switch (stage)
            {
                case XLeRobotOperatorStage.ConnectRequired: return "Connection";
                case XLeRobotOperatorStage.MirrorWaitingRobot: return "Waiting";
                case XLeRobotOperatorStage.MirrorReady: return "Mirror Ready";
                case XLeRobotOperatorStage.TeleopHeadOnly: return "Teleop Head";
                case XLeRobotOperatorStage.TeleopArms: return "Teleop Arms";
                case XLeRobotOperatorStage.Suspended: return "Suspended";
                default: return "Unknown";
            }
        }

        private static Color StageColor(XLeRobotOperatorStage stage)
        {
            switch (stage)
            {
                case XLeRobotOperatorStage.MirrorReady: return new Color(0.2f, 0.55f, 1.0f);
                case XLeRobotOperatorStage.TeleopHeadOnly:
                case XLeRobotOperatorStage.TeleopArms: return new Color(0.15f, 0.8f, 0.35f);
                case XLeRobotOperatorStage.Suspended: return new Color(1.0f, 0.25f, 0.25f);
                default: return new Color(0.75f, 0.75f, 0.75f);
            }
        }
    }
}
