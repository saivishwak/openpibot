using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Networking;

namespace XLeRobot.QuestTeleop
{
    public sealed class XLeRobotVideoBridgeClient : MonoBehaviour
    {
        [SerializeField] private XLeRobotStateClient stateClient;
        [SerializeField] private string questHostOverride = "";

        public UnityEvent<string> onVideoStatusJson = new UnityEvent<string>();
        public UnityEvent<string> onError = new UnityEvent<string>();

        public bool StartInFlight { get; private set; }
        public float LastStartAttemptTime { get; private set; } = -1000.0f;
        public string LastStatusJson { get; private set; } = "";
        public string LastError { get; private set; } = "";

        public void Configure(XLeRobotStateClient client, string hostOverride = "")
        {
            stateClient = client;
            questHostOverride = hostOverride ?? "";
        }

        public void EnsureVideoStarted()
        {
            if (StartInFlight)
            {
                return;
            }
            StartVideo();
        }

        public void StartVideo()
        {
            if (stateClient == null)
            {
                RaiseError("stateClient is not assigned");
                return;
            }
            if (StartInFlight)
            {
                return;
            }
            string json = string.IsNullOrWhiteSpace(questHostOverride)
                ? "{}"
                : "{\"quest_host\":\"" + EscapeJson(questHostOverride) + "\"}";
            StartInFlight = true;
            LastStartAttemptTime = Time.unscaledTime;
            StartCoroutine(PostJson("/api/vr/quest/video/start", json, true));
        }

        public void StopVideo()
        {
            StartCoroutine(PostJson("/api/vr/quest/video/stop", "{}"));
        }

        public void RefreshStatus()
        {
            if (stateClient == null)
            {
                RaiseError("stateClient is not assigned");
                return;
            }
            StartCoroutine(GetJson("/api/vr/quest/video/status"));
        }

        public void ReportReceiveHealth(string role, string state, float fps, float latencyMs, int frames, string error = "")
        {
            string json = "{"
                + "\"role\":\"" + EscapeJson(role) + "\","
                + "\"state\":\"" + EscapeJson(state) + "\","
                + "\"fps\":" + fps.ToString(System.Globalization.CultureInfo.InvariantCulture) + ","
                + "\"latency_ms\":" + latencyMs.ToString(System.Globalization.CultureInfo.InvariantCulture) + ","
                + "\"frames\":" + frames + ","
                + "\"error\":\"" + EscapeJson(error) + "\""
                + "}";
            StartCoroutine(PostJson("/api/vr/quest/video/health", json));
        }

        private IEnumerator GetJson(string path)
        {
            using UnityWebRequest req = UnityWebRequest.Get(stateClient.Settings.HttpBaseUrl + path);
            req.SetRequestHeader("x-quest-pairing-token", stateClient.Settings.pairingToken);
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
            {
                RaiseError(req.error);
                yield break;
            }
            RaiseStatus(req.downloadHandler.text);
        }

        private IEnumerator PostJson(string path, string json, bool isStartRequest = false)
        {
            if (stateClient == null)
            {
                RaiseError("stateClient is not assigned");
                if (isStartRequest)
                {
                    StartInFlight = false;
                }
                yield break;
            }

            byte[] body = Encoding.UTF8.GetBytes(json);
            using UnityWebRequest req = new UnityWebRequest(stateClient.Settings.HttpBaseUrl + path, "POST");
            req.uploadHandler = new UploadHandlerRaw(body);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("content-type", "application/json");
            req.SetRequestHeader("x-quest-pairing-token", stateClient.Settings.pairingToken);
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                RaiseError(req.error + ": " + req.downloadHandler.text);
                if (isStartRequest)
                {
                    StartInFlight = false;
                }
                yield break;
            }
            RaiseStatus(req.downloadHandler.text);
            if (isStartRequest)
            {
                StartInFlight = false;
            }
        }

        private void RaiseStatus(string json)
        {
            LastStatusJson = json ?? "";
            LastError = "";
            onVideoStatusJson.Invoke(LastStatusJson);
        }

        private void RaiseError(string error)
        {
            LastError = error ?? "";
            onError.Invoke(LastError);
        }

        private static string EscapeJson(string value)
        {
            return (value ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"");
        }
    }
}
