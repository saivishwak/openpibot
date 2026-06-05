using UnityEngine;

namespace XLeRobot.QuestTeleop
{
    public sealed class XLeRobotQuestBootstrap : MonoBehaviour
    {
        [Header("Endpoint")]
        [SerializeField] private string workstationHost = "192.168.0.113";
        [SerializeField] private int workstationPort = 5000;
        [SerializeField] private bool useTls = false;
        [SerializeField] private string pairingToken = "dev-quest-token";

        [Header("Scene Components")]
        [SerializeField] private XLeRobotQuestClient questClient;
        [SerializeField] private XLeRobotStateClient stateClient;
        [SerializeField] private XLeRobotOperatorFlowManager flowManager;
        [SerializeField] private Transform headset;

        private void Awake()
        {
            ApplyEndpoint();
            if (stateClient != null && flowManager != null)
            {
                stateClient.onStatus.AddListener(flowManager.ApplyStatus);
            }
        }

        public void ConfigureEndpoint(string host, int port, bool tls, string token)
        {
            workstationHost = host;
            workstationPort = port;
            useTls = tls;
            pairingToken = token ?? "";
            ApplyEndpoint();
        }

        public void BindSceneComponents(
            XLeRobotQuestClient quest,
            XLeRobotStateClient state,
            XLeRobotOperatorFlowManager flow,
            Transform headsetTransform)
        {
            questClient = quest;
            stateClient = state;
            flowManager = flow;
            headset = headsetTransform;
            ApplyEndpoint();
            if (stateClient != null && flowManager != null)
            {
                stateClient.onStatus.RemoveListener(flowManager.ApplyStatus);
                stateClient.onStatus.AddListener(flowManager.ApplyStatus);
            }
        }

        public void ApplyEndpoint()
        {
            if (questClient != null)
            {
                questClient.ConfigureEndpoint(workstationHost, workstationPort, useTls, pairingToken);
            }
            if (stateClient != null)
            {
                stateClient.ConfigureEndpoint(workstationHost, workstationPort, useTls, pairingToken);
            }
        }

        public void Recenter()
        {
            if (questClient != null)
            {
                questClient.RecenterOperatorOrigin(headset);
            }
        }
    }
}
