import { AudioApiClient } from "@/shared/api/audio";
import { ChatApiClient } from "@/shared/api/chat";
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
  "updateStudioProjectRenderJob",
  "listTranscriptionRecords",
  "listTranscriptionRecordPage",
  "getTranscriptionRecord",
  "createTranscriptionRecord",
  "createTranscriptionRecordStream",
  "regenerateTranscriptionSummary",
  "transcriptionRecordAudioUrl",
  "deleteTranscriptionRecord",
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
  "diarize",
  "asrTranscribeStream",
  "synthesize",
  "transcribe",
] as const;

const chatMethodNames = [
  "listChatThreads",
  "createChatThread",
  "updateChatThread",
  "getChatThread",
  "listChatThreadMessages",
  "deleteChatThread",
  "sendChatThreadMessage",
  "sendChatThreadMessageStream",
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

  return {
    baseUrl: http.baseUrl,
    ...bindMethods(modelsApi, modelMethodNames),
    ...bindMethods(audioApi, audioMethodNames),
    ...bindMethods(chatApi, chatMethodNames),
    ...bindMethods(voiceApi, voiceMethodNames),
    ...bindMethods(onboardingApi, onboardingMethodNames),
    ...bindMethods(preferencesApi, preferencesMethodNames),
  };
}

export type ApiClient = ReturnType<typeof createApiClient>;
