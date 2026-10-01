import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  ModelCatalogProvider,
  useModelCatalog,
} from "@/app/providers/ModelCatalogProvider";
import { NotificationProvider } from "@/app/providers/NotificationProvider";
import type { ModelInfo } from "@/api";

const apiMocks = vi.hoisted(() => ({
  listModels: vi.fn(),
  loadModel: vi.fn(),
  unloadModel: vi.fn(),
}));

vi.mock("@/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/api")>();
  return {
    ...original,
    api: {
      baseUrl: "http://localhost:8080/v1",
      listModels: apiMocks.listModels,
      loadModel: apiMocks.loadModel,
      unloadModel: apiMocks.unloadModel,
    },
  };
});

vi.mock("@/app/analytics/events", () => ({
  trackModelDownloadCompleted: vi.fn(),
  trackModelDownloadStarted: vi.fn(),
  trackModelLoaded: vi.fn(),
}));

const model: ModelInfo = {
  variant: "Qwen3.5-4B",
  status: "downloaded",
  local_path: "/models/qwen",
  size_bytes: 42,
  download_progress: null,
  error_message: null,
};

function CatalogProbe() {
  const {
    models,
    error,
    catalogError,
    loading,
    refreshModels,
    loadModel,
    unloadModel,
  } = useModelCatalog();

  return (
    <div>
      <span>{loading ? "loading" : "ready"}</span>
      <span data-testid="model-count">{models.length}</span>
      <span data-testid="catalog-error">{error}</span>
      <span data-testid="catalog-load-error">{catalogError}</span>
      <button type="button" onClick={() => void refreshModels()}>
        Retry catalog
      </button>
      <button type="button" onClick={() => void loadModel(model.variant)}>
        Load
      </button>
      <button type="button" onClick={() => void unloadModel(model.variant)}>
        Unload
      </button>
    </div>
  );
}

function renderCatalog() {
  return render(
    <NotificationProvider>
      <ModelCatalogProvider>
        <CatalogProbe />
      </ModelCatalogProvider>
    </NotificationProvider>,
  );
}

function deferredPromise<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

describe("ModelCatalogProvider model action errors", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMocks.listModels.mockResolvedValue({ models: [model] });
  });

  it("surfaces the exact backend model load error", async () => {
    apiMocks.loadModel.mockRejectedValue(
      new Error("Metal allocation failed: requested 8192 MiB"),
    );
    renderCatalog();
    await screen.findByText("ready");

    fireEvent.click(screen.getByRole("button", { name: "Load" }));

    await screen.findByText("Model load failed");
    await waitFor(() =>
      expect(screen.getByTestId("catalog-error")).toHaveTextContent(
        "Metal allocation failed: requested 8192 MiB",
      ),
    );
    expect(
      screen.getAllByText("Metal allocation failed: requested 8192 MiB"),
    ).toHaveLength(2);
  });

  it("surfaces an initial catalog failure and clears it after retry", async () => {
    apiMocks.listModels
      .mockRejectedValueOnce(new Error("Local model service is offline"))
      .mockResolvedValueOnce({ models: [model] });

    renderCatalog();

    await screen.findByText("ready");
    expect(screen.getByTestId("model-count")).toHaveTextContent("0");
    expect(screen.getByTestId("catalog-load-error")).toHaveTextContent(
      "Local model service is offline",
    );

    fireEvent.click(screen.getByRole("button", { name: "Retry catalog" }));

    await waitFor(() =>
      expect(screen.getByTestId("model-count")).toHaveTextContent("1"),
    );
    expect(screen.getByTestId("catalog-load-error")).toBeEmptyDOMElement();
    expect(apiMocks.listModels).toHaveBeenCalledTimes(2);
  });

  it("treats an empty catalog response as loaded rather than failed", async () => {
    apiMocks.listModels.mockResolvedValueOnce({ models: [] });

    renderCatalog();

    await screen.findByText("ready");
    expect(screen.getByTestId("model-count")).toHaveTextContent("0");
    expect(screen.getByTestId("catalog-load-error")).toBeEmptyDOMElement();
  });

  it("surfaces the exact backend model unload error", async () => {
    apiMocks.unloadModel.mockRejectedValue(
      new Error("Model Qwen3.5-4B is serving active requests"),
    );
    renderCatalog();
    await screen.findByText("ready");

    fireEvent.click(screen.getByRole("button", { name: "Unload" }));

    await screen.findByText("Model unload failed");
    await waitFor(() =>
      expect(screen.getByTestId("catalog-error")).toHaveTextContent(
        "Model Qwen3.5-4B is serving active requests",
      ),
    );
    expect(
      screen.getAllByText("Model Qwen3.5-4B is serving active requests"),
    ).toHaveLength(2);
  });

  it("cancels an in-flight model load without reporting it as loaded", async () => {
    const load = deferredPromise<{ status: string; message: string }>();
    const unload = deferredPromise<{ status: string; message: string }>();
    apiMocks.loadModel.mockReturnValue(load.promise);
    apiMocks.unloadModel.mockReturnValue(unload.promise);
    renderCatalog();
    await screen.findByText("ready");

    fireEvent.click(screen.getByRole("button", { name: "Load" }));
    await waitFor(() => expect(apiMocks.loadModel).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("button", { name: "Unload" }));
    await waitFor(() => expect(apiMocks.unloadModel).toHaveBeenCalled());

    await act(async () => {
      load.resolve({ status: "loaded", message: "loaded" });
      await load.promise;
    });
    expect(screen.queryByText("Model loaded")).not.toBeInTheDocument();

    await act(async () => {
      unload.resolve({ status: "unloaded", message: "unloaded" });
      await unload.promise;
    });
    expect(await screen.findByText("Model load cancelled")).toBeInTheDocument();
  });
});

function ChatSwitchProbe({ targetVariant }: { targetVariant: string }) {
  const { models, loadModel, residencySummary } = useModelCatalog();

  return (
    <div>
      <span data-testid="resident-models">
        {models.filter((model) => model.status === "ready").map((model) => model.variant).join(",")}
      </span>
      <span data-testid="residency-summary">
        {residencySummary
          ? `${residencySummary.resident_count}/${residencySummary.max_loaded_models}`
          : "none"}
      </span>
      <button type="button" onClick={() => void loadModel(targetVariant)}>
        Load target
      </button>
    </div>
  );
}

function renderChatSwitch(targetVariant: string) {
  return render(
    <NotificationProvider>
      <ModelCatalogProvider>
        <ChatSwitchProbe targetVariant={targetVariant} />
      </ModelCatalogProvider>
    </NotificationProvider>,
  );
}

describe("ModelCatalogProvider residency-aware chat eviction", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("keeps the diarization refiner resident across chat model switches", async () => {
    apiMocks.listModels.mockResolvedValue({
      models: [
        { ...model, status: "ready" },
        {
          ...model,
          variant: "Qwen3-8B-GGUF",
          status: "ready",
        },
      ],
    });
    apiMocks.loadModel.mockResolvedValue({ status: "loaded", message: "loaded" });

    renderChatSwitch("Qwen3-8B-GGUF");
    await screen.findByTestId("resident-models");

    fireEvent.click(screen.getByRole("button", { name: "Load target" }));

    await waitFor(() => expect(apiMocks.loadModel).toHaveBeenCalledWith("Qwen3-8B-GGUF"));
    expect(apiMocks.unloadModel).not.toHaveBeenCalled();
  });

  it("still evicts non-pipeline chat models on a chat switch", async () => {
    apiMocks.listModels.mockResolvedValue({
      models: [
        { ...model, variant: "Qwen3-8B-GGUF", status: "ready" },
        { ...model, variant: "Qwen3-4B-GGUF", status: "ready" },
      ],
    });
    apiMocks.loadModel.mockResolvedValue({ status: "loaded", message: "loaded" });
    apiMocks.unloadModel.mockResolvedValue({ status: "unloaded", message: "unloaded" });

    renderChatSwitch("Qwen3-4B-GGUF");
    await screen.findByTestId("resident-models");

    fireEvent.click(screen.getByRole("button", { name: "Load target" }));

    await waitFor(() =>
      expect(apiMocks.unloadModel).toHaveBeenCalledWith("Qwen3-8B-GGUF"),
    );
    expect(apiMocks.loadModel).toHaveBeenCalledWith("Qwen3-4B-GGUF");
  });

  it("never evicts pinned models on a chat switch", async () => {
    apiMocks.listModels.mockResolvedValue({
      models: [
        { ...model, variant: "Qwen3-8B-GGUF", status: "ready", pinned: true },
        { ...model, variant: "Qwen3-4B-GGUF", status: "ready" },
      ],
    });
    apiMocks.loadModel.mockResolvedValue({ status: "loaded", message: "loaded" });

    renderChatSwitch("Qwen3-4B-GGUF");
    await screen.findByTestId("resident-models");

    fireEvent.click(screen.getByRole("button", { name: "Load target" }));

    await waitFor(() => expect(apiMocks.loadModel).toHaveBeenCalledWith("Qwen3-4B-GGUF"));
    expect(apiMocks.unloadModel).not.toHaveBeenCalled();
  });

  it("exposes the server residency summary", async () => {
    apiMocks.listModels.mockResolvedValue({
      models: [{ ...model, status: "ready" }],
      residency: {
        resident_count: 1,
        max_loaded_models: 4,
        model_keep_alive_secs: 600,
      },
    });

    renderChatSwitch("Qwen3-8B-GGUF");

    await waitFor(() =>
      expect(screen.getByTestId("residency-summary")).toHaveTextContent("1/4"),
    );
  });
});
