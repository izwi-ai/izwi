import { AudioApiClient } from "@/shared/api/audio";
import { ChatApiClient } from "@/shared/api/chat";
import { HealthApiClient } from "@/shared/api/health";
import { ApiHttpClient, API_BASE } from "@/shared/api/http";
import { ModelApiClient } from "@/shared/api/models";
import { OnboardingApiClient } from "@/shared/api/onboarding";
import { PreferencesApiClient } from "@/shared/api/preferences";
import { VoiceApiClient } from "@/shared/api/voice";

const modelMethodNames = [
  "listModels",
  "getModelInfo",
  "downloadModel",
  "loadModel",
  "unloadModel",
  "deleteModel",
  "cancelDownload",
] as const;

const audioMethodNames = [
  "generateTTS",
  "generateTTSWithStats",
  "generateTTSStream",
  "listSpeechHistoryRecords",
  "listSpeechHistoryRecordPage",
  "getSpeechHistoryRecord",
  "createSpeechHistoryRecord",
  "createSpeechHistoryRecordStream",
  "speechHistoryRecordAudioUrl",
  "downloadAudioFile",
  "saveAudioFile",
  "deleteSpeechHistoryRecord",
  "listTextToSpeechRecords",
  "listTextToSpeechRecordPage",
  "getTextToSpeechRecord",
  "createTextToSpeechRecord",
  "createTextToSpeechRecordStream",
  "attachTextToSpeechRecordStream",
  "cancelTextToSpeechRecord",
  "textToSpeechRecordAudioUrl",
  "deleteTextToSpeechRecord",
  "listVoiceDesignRecords",
  "listVoiceDesignRecordPage",
  "getVoiceDesignRecord",
  "createVoiceDesignRecord",
  "voiceDesignRecordAudioUrl",
  "deleteVoiceDesignRecord",
  "listVoiceCloningRecords",
  "listVoiceCloningRecordPage",
  "getVoiceCloningRecord",
  "createVoiceCloningRecord",
  "voiceCloningRecordAudioUrl",
  "deleteVoiceCloningRecord",
  "listSavedVoices",
  "listSavedVoicePage",
  "getSavedVoice",
  "createSavedVoice",
  "savedVoiceAudioUrl",
  "deleteSavedVoice",
  "listStudioProjects",
  "listStudioProjectPage",
  "createStudioProject",
  "getStudioProject",
  "updateStudioProject",
  "updateStudioProjectSegment",
  "createStudioProjectSegment",
  "splitStudioProjectSegment",
  "mergeStudioProjectSegmentWithNext",
  "reorderStudioProjectSegments",
  "bulkDeleteStudioProjectSegments",
  "deleteStudioProjectSegment",
  "renderStudioProjectSegment",
  "studioProjectSegmentAudioUrl",
  "studioProjectAudioUrl",
  "deleteStudioProject",
  "listStudioProjectFolders",
  "createStudioProjectFolder",
  "getStudioProjectMeta",
  "updateStudioProjectMeta",
  "listStudioProjectPronunciations",
  "createStudioProjectPronunciation",
  "deleteStudioProjectPronunciation",
  "listStudioProjectSnapshots",
  "createStudioProjectSnapshot",
  "restoreStudioProjectSnapshot",
  "listStudioProjectRenderJobs",
  "createStudioProjectRenderJob",
  "cancelStudioProjectRenderJob",
  "listSpeechTextJobs",
  "listSpeechTextJobPage",
  "getSpeechTextJob",
  "createSpeechTextJob",
  "updateSpeechTextJob",
  "rerunSpeechTextJob",
  "cancelSpeechTextJob",
  "speechTextJobAudioUrl",
  "regenerateSpeechTextJobSummary",
  "deleteSpeechTextJob",
  "listTranscriptionRecords",
  "listTranscriptionRecordPage",
  "getTranscriptionRecord",
  "createTranscriptionRecord",
  "createTranscriptionRecordStream",
  "regenerateTranscriptionSummary",
  "transcriptionRecordAudioUrl",
  "cancelTranscriptionRecord",
  "retryTranscriptionRecord",
  "deleteTranscriptionRecord",
  "listSpeakerAttributedAsrRecords",
  "listSpeakerAttributedAsrRecordPage",
  "getSpeakerAttributedAsrRecord",
  "createSpeakerAttributedAsrRecord",
  "speakerAttributedAsrRecordAudioUrl",
  "regenerateSpeakerAttributedAsrSummary",
  "deleteSpeakerAttributedAsrRecord",
  "asrStatus",
  "asrTranscribe",
  "listDiarizationRecords",
  "listDiarizationRecordPage",
  "getDiarizationRecord",
  "updateDiarizationRecord",
  "rerunDiarizationRecord",
  "cancelDiarizationRecord",
  "regenerateDiarizationSummary",
  "createDiarizationRecord",
  "diarizationRecordAudioUrl",
  "deleteDiarizationRecord",
  "asrTranscribeStream",
  "synthesize",
  "transcribe",
] as const;

const chatMethodNames = [
  "listChatThreads",
  "createChatThread",
  "updateChatThread",
  "generateChatThreadTitle",
  "getChatThread",
  "listChatThreadMessages",
  "deleteChatThread",
  "sendChatThreadMessage",
  "sendChatThreadMessageStream",
  "editChatThreadMessageStream",
  "regenerateChatThreadMessageStream",
  "deleteChatThreadMessage",
  "createAgentSession",
  "getAgentSession",
  "createAgentTurn",
  "chatCompletions",
  "chatCompletionsStream",
  "createResponse",
  "createResponseStream",
  "getResponse",
  "cancelResponse",
  "deleteResponse",
] as const;

const voiceMethodNames = [
  "getVoiceProfile",
  "updateVoiceProfile",
  "listVoiceObservations",
  "deleteVoiceObservation",
  "clearVoiceObservations",
] as const;

const onboardingMethodNames = [
  "getOnboardingState",
  "completeOnboarding",
] as const;

const healthMethodNames = ["getHealth"] as const;

const preferencesMethodNames = [
  "getPreferences",
  "updateAnalyticsPreference",
] as const;

function bindMethods<T extends object, K extends readonly (keyof T)[]>(
  instance: T,
  methodNames: K,
): Pick<T, K[number]> {
  const bound = {} as Pick<T, K[number]>;

  methodNames.forEach((name) => {
    const value = instance[name];
    bound[name] = (
      typeof value === "function" ? value.bind(instance) : value
    ) as T[K[number]];
  });

  return bound;
}

export function createApiClient(baseUrl: string = API_BASE) {
  const http = new ApiHttpClient(baseUrl);
  const modelsApi = new ModelApiClient(http);
  const audioApi = new AudioApiClient(http);
  const chatApi = new ChatApiClient(http);
  const voiceApi = new VoiceApiClient(http);
  const onboardingApi = new OnboardingApiClient(http);
  const preferencesApi = new PreferencesApiClient(http);
  const healthApi = new HealthApiClient(http);

  return {
    baseUrl: http.baseUrl,
    ...bindMethods(modelsApi, modelMethodNames),
    ...bindMethods(audioApi, audioMethodNames),
    ...bindMethods(chatApi, chatMethodNames),
    ...bindMethods(voiceApi, voiceMethodNames),
    ...bindMethods(onboardingApi, onboardingMethodNames),
    ...bindMethods(preferencesApi, preferencesMethodNames),
    ...bindMethods(healthApi, healthMethodNames),
  };
}

export type ApiClient = ReturnType<typeof createApiClient>;
