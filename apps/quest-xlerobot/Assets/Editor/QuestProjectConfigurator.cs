using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.XR.Management;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR;
using UnityEngine.XR.OpenXR.Features;

public static class QuestProjectConfigurator
{
    private const string XRGeneralSettingsPath = "Assets/XR/XRGeneralSettingsPerBuildTarget.asset";
    private const string OpenXRLoaderPath = "Assets/XR/Loaders/OpenXRLoader.asset";
    private const string AndroidPackageName = "com.xlerobot.questteleop";

    private static readonly string[] BuildScenes =
    {
        "Assets/Scenes/BaseScene.unity",
        "Assets/Scenes/MirrorScene.unity",
        "Assets/Scenes/TeleoperationScene.unity",
    };

    private static readonly HashSet<string> QuestFeatureTypes = new()
    {
        "UnityEngine.XR.OpenXR.Features.MetaQuestSupport.MetaQuestFeature",
        "UnityEngine.XR.OpenXR.Features.Meta.ARSessionFeature",
        "UnityEngine.XR.OpenXR.Features.Meta.ARCameraFeature",
        "UnityEngine.XR.OpenXR.Features.Interactions.OculusTouchControllerProfile",
        "UnityEngine.XR.OpenXR.Features.Interactions.MetaQuestTouchPlusControllerProfile",
        "UnityEngine.XR.OpenXR.Features.Interactions.MetaQuestTouchProControllerProfile",
    };

    [MenuItem("XLeRobot/Configure Quest Project")]
    public static void ConfigureQuestProject()
    {
        EnsureAndroidBuildSettings();
        EnsureBuildScenes();
        EnsureXRManagement();
        EnableQuestOpenXRFeatures();

        AssetDatabase.SaveAssets();
        Debug.Log("XLeRobot Quest project configuration complete.");
    }

    public static void ConfigureQuestProjectAndExit()
    {
        try
        {
            ConfigureQuestProject();
            EditorApplication.Exit(0);
        }
        catch (System.Exception ex)
        {
            Debug.LogException(ex);
            EditorApplication.Exit(1);
        }
    }

    public static void BuildQuestApkAndExit()
    {
        try
        {
            ConfigureQuestProject();
            Directory.CreateDirectory("builds");
            var options = new BuildPlayerOptions
            {
                scenes = BuildScenes,
                locationPathName = "builds/xlerobot-quest-teleop.apk",
                target = BuildTarget.Android,
                targetGroup = BuildTargetGroup.Android,
                options = BuildOptions.None,
            };
            BuildReport report = BuildPipeline.BuildPlayer(options);
            if (report.summary.result != BuildResult.Succeeded)
            {
                Debug.LogError($"Quest APK build failed: {report.summary.result}");
                EditorApplication.Exit(1);
                return;
            }

            Debug.Log($"Quest APK build succeeded: {report.summary.outputPath}");
            EditorApplication.Exit(0);
        }
        catch (System.Exception ex)
        {
            Debug.LogException(ex);
            EditorApplication.Exit(1);
        }
    }

    private static void EnsureAndroidBuildSettings()
    {
        if (BuildPipeline.IsBuildTargetSupported(BuildTargetGroup.Android, BuildTarget.Android))
        {
            EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Android, BuildTarget.Android);
        }

        PlayerSettings.companyName = "XLeRobot";
        PlayerSettings.productName = "XLeRobot Quest Teleop";
        PlayerSettings.colorSpace = ColorSpace.Linear;
        PlayerSettings.insecureHttpOption = InsecureHttpOption.AlwaysAllowed;
        PlayerSettings.SetApplicationIdentifier(BuildTargetGroup.Android, AndroidPackageName);
        PlayerSettings.SetScriptingBackend(NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
        PlayerSettings.Android.applicationEntry = AndroidApplicationEntry.Activity;
        PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
        PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel29;
        PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto;
        PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
        PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.Vulkan });
    }

    private static void EnsureBuildScenes()
    {
        EditorBuildSettings.scenes = BuildScenes
            .Where(scene => AssetDatabase.LoadAssetAtPath<SceneAsset>(scene) != null)
            .Select(scene => new EditorBuildSettingsScene(scene, true))
            .ToArray();
    }

    private static void EnsureXRManagement()
    {
        var settingsByTarget = AssetDatabase.LoadAssetAtPath<XRGeneralSettingsPerBuildTarget>(XRGeneralSettingsPath);
        if (settingsByTarget == null)
        {
            settingsByTarget = ScriptableObject.CreateInstance<XRGeneralSettingsPerBuildTarget>();
            AssetDatabase.CreateAsset(settingsByTarget, XRGeneralSettingsPath);
        }

        EditorBuildSettings.AddConfigObject(XRGeneralSettings.k_SettingsKey, settingsByTarget, true);

        if (!settingsByTarget.HasSettingsForBuildTarget(BuildTargetGroup.Android))
        {
            settingsByTarget.CreateDefaultSettingsForBuildTarget(BuildTargetGroup.Android);
        }

        if (!settingsByTarget.HasManagerSettingsForBuildTarget(BuildTargetGroup.Android))
        {
            settingsByTarget.CreateDefaultManagerSettingsForBuildTarget(BuildTargetGroup.Android);
        }

        var manager = settingsByTarget.ManagerSettingsForBuildTarget(BuildTargetGroup.Android);
        manager.automaticLoading = true;
        manager.automaticRunning = true;

        var loader = AssetDatabase.LoadAssetAtPath<OpenXRLoader>(OpenXRLoaderPath);
        if (loader == null)
        {
            loader = ScriptableObject.CreateInstance<OpenXRLoader>();
            AssetDatabase.CreateAsset(loader, OpenXRLoaderPath);
        }

        var loaders = manager.activeLoaders
            .Where(existing => existing != null && existing.GetType() != typeof(OpenXRLoader))
            .ToList();
        loaders.Insert(0, loader);
        manager.TrySetLoaders(loaders);

        EditorUtility.SetDirty(loader);
        EditorUtility.SetDirty(manager);
        EditorUtility.SetDirty(settingsByTarget);
    }

    private static void EnableQuestOpenXRFeatures()
    {
        var settings = OpenXRSettings.GetSettingsForBuildTargetGroup(BuildTargetGroup.Android);
        if (settings == null)
        {
            throw new System.InvalidOperationException("OpenXR settings for Android were not found.");
        }

        settings.renderMode = OpenXRSettings.RenderMode.SinglePassInstanced;

        foreach (OpenXRFeature feature in settings.GetFeatures())
        {
            if (feature != null && QuestFeatureTypes.Contains(feature.GetType().FullName))
            {
                feature.enabled = true;
                EditorUtility.SetDirty(feature);
            }
        }

        EditorUtility.SetDirty(settings);
    }
}
