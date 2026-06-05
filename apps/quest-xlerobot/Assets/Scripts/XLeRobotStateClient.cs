using System.Collections;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Networking;

namespace XLeRobot.QuestTeleop
{
    public sealed class XLeRobotStateClient : MonoBehaviour
    {
        [SerializeField] private XLeRobotQuestSettings settings = new XLeRobotQuestSettings();
        [SerializeField] private float pollHz = 1.0f;

        public UnityEvent<XLeRobotQuestStatus> onStatus = new UnityEvent<XLeRobotQuestStatus>();
        public UnityEvent<string> onRawStatusJson = new UnityEvent<string>();
        public UnityEvent<string> onError = new UnityEvent<string>();

        private Coroutine pollRoutine;
        private XLeRobotQuestStatus lastStatus = new XLeRobotQuestStatus();

        public XLeRobotQuestSettings Settings => settings;
        public XLeRobotQuestStatus LastStatus => lastStatus;

        private void OnEnable()
        {
            pollRoutine = StartCoroutine(PollLoop());
        }

        private void OnDisable()
        {
            if (pollRoutine != null)
            {
                StopCoroutine(pollRoutine);
                pollRoutine = null;
            }
        }

        public void ConfigureEndpoint(string host, int port, bool tls, string token)
        {
            settings.workstationHost = host;
            settings.workstationPort = port;
            settings.useTls = tls;
            settings.pairingToken = token ?? "";
        }

        private IEnumerator PollLoop()
        {
            WaitForSeconds wait = new WaitForSeconds(1.0f / Mathf.Max(1.0f, pollHz));
            while (enabled)
            {
                yield return FetchStatus();
                yield return wait;
            }
        }

        private IEnumerator FetchStatus()
        {
            using UnityWebRequest req = UnityWebRequest.Get($"{settings.HttpBaseUrl}/api/vr/quest/operator");
            req.SetRequestHeader("x-quest-pairing-token", settings.pairingToken);
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                onError.Invoke(req.error);
                yield break;
            }

            ApplyStatusJson(req.downloadHandler.text);
        }

        public void ApplyStatusJson(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
            {
                return;
            }
            onRawStatusJson.Invoke(json);
            lastStatus = XLeRobotQuestStatus.FromJson(json);
            onStatus.Invoke(lastStatus);
        }

        public void ApplyServerMessage(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
            {
                return;
            }

            if (json.Contains("\"ok\":false"))
            {
                string error = ExtractStringValue(json, "error");
                onError.Invoke(string.IsNullOrEmpty(error) ? json : error);
                return;
            }

            string operatorJson = ExtractObjectValue(json, "operator");
            if (!string.IsNullOrEmpty(operatorJson))
            {
                ApplyStatusJson(operatorJson);
            }
        }

        private static string ExtractObjectValue(string json, string key)
        {
            string quotedKey = "\"" + key + "\"";
            int keyIndex = json.IndexOf(quotedKey, System.StringComparison.Ordinal);
            if (keyIndex < 0)
            {
                return "";
            }

            int colonIndex = json.IndexOf(':', keyIndex + quotedKey.Length);
            if (colonIndex < 0)
            {
                return "";
            }

            int objectStart = colonIndex + 1;
            while (objectStart < json.Length && char.IsWhiteSpace(json[objectStart]))
            {
                objectStart++;
            }
            if (objectStart >= json.Length || json[objectStart] != '{')
            {
                return "";
            }

            bool inString = false;
            bool escaped = false;
            int depth = 0;
            for (int i = objectStart; i < json.Length; i++)
            {
                char ch = json[i];
                if (escaped)
                {
                    escaped = false;
                    continue;
                }
                if (ch == '\\' && inString)
                {
                    escaped = true;
                    continue;
                }
                if (ch == '"')
                {
                    inString = !inString;
                    continue;
                }
                if (inString)
                {
                    continue;
                }
                if (ch == '{')
                {
                    depth++;
                }
                else if (ch == '}')
                {
                    depth--;
                    if (depth == 0)
                    {
                        return json.Substring(objectStart, i - objectStart + 1);
                    }
                }
            }
            return "";
        }

        private static string ExtractStringValue(string json, string key)
        {
            string quotedKey = "\"" + key + "\"";
            int keyIndex = json.IndexOf(quotedKey, System.StringComparison.Ordinal);
            if (keyIndex < 0)
            {
                return "";
            }

            int colonIndex = json.IndexOf(':', keyIndex + quotedKey.Length);
            if (colonIndex < 0)
            {
                return "";
            }
            int valueStart = colonIndex + 1;
            while (valueStart < json.Length && char.IsWhiteSpace(json[valueStart]))
            {
                valueStart++;
            }
            if (valueStart >= json.Length || json[valueStart] != '"')
            {
                return "";
            }

            valueStart++;
            System.Text.StringBuilder value = new();
            bool escaped = false;
            for (int i = valueStart; i < json.Length; i++)
            {
                char ch = json[i];
                if (escaped)
                {
                    value.Append(ch);
                    escaped = false;
                    continue;
                }
                if (ch == '\\')
                {
                    escaped = true;
                    continue;
                }
                if (ch == '"')
                {
                    return value.ToString();
                }
                value.Append(ch);
            }
            return "";
        }
    }
}
