// ==============================================================================
// Web Worker (tts-worker.js) — Dual-Engine Real Offline AI TTS (Kokoro & VieNeu)
// 100% Client-Side WebAssembly (SIMD Multi-threaded), ZERO Backend API.
// ==============================================================================

/* global ort */
importScripts("./ort.min.js");

let currentThreads = 1;
let activeEngine = "vieneu"; // 'kokoro' | 'vieneu'
let kokoroSession = null;
let currentKokoroModelFile = null;
let currentVieneuModelFile = null;
const cachedKokoroVoices = {};

const vieneu = {
  sText: null,
  sDur: null,
  sVe: null,
  sDec: null,
  cfg: null,
  constants: null,
  voices: null,
  nullCtx: null,
  nullMask: null,
};

self.onmessage = async (e) => {
  const { action, payload, id } = e.data || {};
  try {
    if (action === "INIT_CONFIG") {
      handleInitConfig(payload);
    } else if (action === "LOAD_MODEL") {
      await handleLoadModel(payload, id);
    } else if (action === "SYNTHESIZE_STREAM") {
      await handleSynthesis(payload, id, true);
    } else if (action === "SYNTHESIZE_ALL") {
      await handleSynthesis(payload, id, false);
    }
  } catch (err) {
    postError(action, err.message, id);
  }
};

function handleInitConfig(payload) {
  const canThread = typeof SharedArrayBuffer !== "undefined" && self.crossOriginIsolated;
  currentThreads = canThread ? (payload?.threads || 4) : 1;
  ort.env.wasm.simd = true;
  ort.env.wasm.wasmPaths = "./";
  ort.env.wasm.numThreads = currentThreads;
  ort.env.wasm.proxy = false;
  self.postMessage({
    type: "CONFIG_READY",
    threads: currentThreads,
    crossOriginIsolated: !!self.crossOriginIsolated,
  });
}

const WORKER_IDB_NAME = "tts-models-idb";
const WORKER_IDB_STORE = "blobs";

function openWorkerIdb() {
  return new Promise((resolve) => {
    if (typeof indexedDB === "undefined") return resolve(null);
    try {
      const req = indexedDB.open(WORKER_IDB_NAME, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(WORKER_IDB_STORE)) {
          db.createObjectStore(WORKER_IDB_STORE);
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

async function getIdbItem(key) {
  const db = await openWorkerIdb();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(WORKER_IDB_STORE, "readonly");
      const req = tx.objectStore(WORKER_IDB_STORE).get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

async function putIdbItem(key, value) {
  const db = await openWorkerIdb();
  if (!db) return;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(WORKER_IDB_STORE, "readwrite");
      tx.objectStore(WORKER_IDB_STORE).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    } catch {
      resolve();
    }
  });
}

async function deleteIdbItem(key) {
  const db = await openWorkerIdb();
  if (!db) return;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(WORKER_IDB_STORE, "readwrite");
      tx.objectStore(WORKER_IDB_STORE).delete(key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    } catch {
      resolve();
    }
  });
}

async function fetchWithIdbCache(key, localPath, fallbackUrl, label) {
  const cached = await getIdbItem(key);
  if (cached) {
    notifyProgress(1, 1, `Nạp ${label} từ IndexedDB...`);
    if (cached instanceof ArrayBuffer) return cached;
    if (cached instanceof Blob) return await cached.arrayBuffer();
  }
  notifyProgress(1, 1, `Đang tải & lưu ${label} vào IndexedDB...`);
  let buffer = null;
  try {
    const res = await fetch(localPath);
    const ct = res.headers.get("content-type") || "";
    if (res.ok && !ct.includes("html")) {
      buffer = await res.arrayBuffer();
    }
  } catch (e) {
    console.warn(`[Worker] Fetch failed for ${localPath}`, e);
  }
  if (!buffer && fallbackUrl) {
    const fbRes = await fetch(fallbackUrl);
    if (!fbRes.ok) throw new Error(`Lỗi tải ${label}: ${fbRes.status}`);
    buffer = await fbRes.arrayBuffer();
  }
  if (buffer) {
    await putIdbItem(key, buffer);
  }
  return buffer;
}

async function fetchOrFallbackSession(
  key,
  localPath,
  fallbackHfUrl,
  sessionOptions,
  label,
) {
  const buffer = await fetchWithIdbCache(
    key,
    localPath,
    fallbackHfUrl,
    label || key,
  );
  if (!buffer) {
    throw new Error(`Không thể nạp mô hình: ${localPath}`);
  }
  const uint8View =
    buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  try {
    return await ort.InferenceSession.create(uint8View, sessionOptions);
  } catch (err) {
    await deleteIdbItem(key);
    throw err;
  }
}

async function handleLoadModel(payload, id) {
  const engine =
    payload.engine ||
    (payload.modelFile?.includes("kokoro") ? "kokoro" : "vieneu");
  if (engine === "vieneu") {
    await handleLoadVieneu(payload, id);
  } else {
    await handleLoadKokoro(payload, id);
  }
}

async function fetchKokoroParts() {
  notifyProgress(1, 2, "Tải Kokoro INT8 Part 1/2...");
  const r1 = await fetch("./kokoro_vi_fast_int8.part1");
  const b1 = await r1.arrayBuffer();
  notifyProgress(2, 2, "Tải Kokoro INT8 Part 2/2...");
  const r2 = await fetch("./kokoro_vi_fast_int8.part2");
  const b2 = await r2.arrayBuffer();
  const merged = new Uint8Array(b1.byteLength + b2.byteLength);
  merged.set(new Uint8Array(b1), 0);
  merged.set(new Uint8Array(b2), b1.byteLength);
  return merged.buffer;
}

async function fetchKokoroBuffer(modelFile) {
  const cached = await getIdbItem(modelFile);
  if (cached) {
    notifyProgress(1, 1, `Nạp ${modelFile} từ IndexedDB offline...`);
    if (cached instanceof ArrayBuffer) return cached;
    if (cached instanceof Blob) return await cached.arrayBuffer();
  }

  notifyProgress(1, 2, `Đang tải ${modelFile}...`);
  let buffer = null;
  try {
    const res = await fetch(`./${modelFile}`);
    const ct = res.headers.get("content-type") || "";
    if (res.ok && !ct.includes("html")) buffer = await res.arrayBuffer();
  } catch (e) {
    console.warn("[Worker] Direct fetch failed for", modelFile, e);
  }

  if (!buffer && modelFile.includes("kokoro_vi_fast_int8")) {
    buffer = await fetchKokoroParts();
  }

  if (!buffer) {
    notifyProgress(1, 1, "Tải Kokoro từ CDN Hugging Face (310MB)...");
    const hf =
      "https://huggingface.co/contextboxai/Kokoro-Vietnamese/resolve/main/kokoro_vi.onnx";
    const hfRes = await fetch(hf);
    if (!hfRes.ok) throw new Error(`HF Kokoro tải lỗi: HTTP ${hfRes.status}`);
    buffer = await hfRes.arrayBuffer();
  }

  if (buffer) {
    notifyProgress(2, 2, "Đang lưu mô hình vào IndexedDB...");
    await putIdbItem(modelFile, buffer);
  }
  return buffer;
}

async function handleLoadKokoro(payload, id) {
  activeEngine = "kokoro";
  const modelFile = payload.modelFile || "kokoro_vi_fast_int8.onnx";
  if (
    kokoroSession &&
    currentKokoroModelFile === modelFile &&
    !payload.customModelBuffer
  ) {
    self.postMessage({
      type: "MODEL_LOADED",
      engine: "kokoro",
      modelFile,
      elapsed: 0,
      threads: currentThreads,
      id,
    });
    return;
  }
  ort.env.wasm.numThreads = currentThreads;
  const t0 = performance.now();
  const modelBuffer =
    payload.customModelBuffer || (await fetchKokoroBuffer(modelFile));
  notifyProgress(1, 1, `Khởi tạo mạng nơ-ron Kokoro ONNX (${modelFile})...`);
  const uint8View =
    modelBuffer instanceof Uint8Array
      ? modelBuffer
      : new Uint8Array(modelBuffer);
  try {
    kokoroSession = await ort.InferenceSession.create(uint8View, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      enableCpuMemArena: true,
      enableMemPattern: true,
      intraOpNumThreads: 1,
      interOpNumThreads: 1,
    });
  } catch (err) {
    if (!payload.customModelBuffer) {
      await deleteIdbItem(modelFile);
    }
    throw err;
  }
  currentKokoroModelFile = payload.customModelBuffer ? "custom" : modelFile;
  const elapsed = Math.round(performance.now() - t0);
  self.postMessage({
    type: "MODEL_LOADED",
    engine: "kokoro",
    modelFile: payload.customModelBuffer ? "Mô hình ONNX tùy biến" : modelFile,
    elapsed,
    threads: currentThreads,
    id,
  });
}

async function handleLoadVieneu(payload, id) {
  activeEngine = "vieneu";
  const isFp16 = payload.modelFile?.includes("fp16");
  const folder = isFp16 ? "vieneu_fp16" : "vieneu";
  const prefix = isFp16 ? "vieneu_fp16_v2" : "vieneu_v2";
  const modelFile = isFp16 ? "vieneu_v3_nano_fp16" : "vieneu_v3_nano_fp32";

  if (
    vieneu.sText &&
    vieneu.sDur &&
    vieneu.sVe &&
    vieneu.sDec &&
    currentVieneuModelFile === modelFile &&
    !payload.customModelBuffer
  ) {
    self.postMessage({
      type: "MODEL_LOADED",
      engine: "vieneu",
      modelFile,
      elapsed: 0,
      threads: currentThreads,
      id,
    });
    return;
  }
  const t0 = performance.now();
  ort.env.wasm.numThreads = currentThreads;
  const so = {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
    intraOpNumThreads: 1,
    interOpNumThreads: 1,
  };
  const hf =
    "https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Nano/resolve/main/";

  notifyProgress(
    1,
    4,
    `1/4: Đang nạp Text Encoder (${isFp16 ? "FP16 13MB" : "FP32 26MB"})...`,
  );
  vieneu.sText = await fetchOrFallbackSession(
    `${prefix}_text_encoder`,
    `./${folder}/text_encoder.onnx`,
    hf + "text_encoder.onnx",
    so,
    `VieNeu Text Encoder (${isFp16 ? "FP16" : "FP32"})`,
  );
  notifyProgress(
    2,
    4,
    `2/4: Đang nạp Duration Predictor (${isFp16 ? "FP16 0.3MB" : "FP32 0.7MB"})...`,
  );
  vieneu.sDur = await fetchOrFallbackSession(
    `${prefix}_duration_predictor`,
    `./${folder}/duration_predictor.onnx`,
    hf + "duration_predictor.onnx",
    so,
    `VieNeu Duration Predictor (${isFp16 ? "FP16" : "FP32"})`,
  );
  notifyProgress(
    3,
    4,
    `3/4: Đang nạp Vector Estimator (${isFp16 ? "FP16 74MB" : "FP32 148MB"})...`,
  );
  vieneu.sVe = await fetchOrFallbackSession(
    `${prefix}_vector_estimator`,
    `./${folder}/vector_estimator.onnx`,
    hf + "vector_estimator.onnx",
    so,
    `VieNeu Vector Estimator (${isFp16 ? "74MB" : "148MB"})`,
  );
  notifyProgress(
    4,
    4,
    `4/4: Đang nạp Codec Decoder (${isFp16 ? "FP16 47MB" : "FP32 95MB"})...`,
  );
  vieneu.sDec = await fetchOrFallbackSession(
    `${prefix}_codec_decoder`,
    `./${folder}/codec_decoder.onnx`,
    hf + "codec_decoder.onnx",
    so,
    `VieNeu Codec Decoder (${isFp16 ? "47MB" : "95MB"})`,
  );

  await loadVieneuMetadata();
  await initVieneuNullContext();
  currentVieneuModelFile = modelFile;
  const elapsed = Math.round(performance.now() - t0);

  self.postMessage({
    type: "MODEL_LOADED",
    engine: "vieneu",
    modelFile,
    elapsed,
    threads: currentThreads,
    id,
  });
}

function notifyProgress(current, total, detail) {
  self.postMessage({
    type: "MODEL_LOAD_PROGRESS",
    engine: activeEngine,
    current,
    total,
    detail,
  });
}

async function loadVieneuMetadata() {
  const [resCfg, resConst, resVoices] = await Promise.all([
    fetch("./vieneu/config.json"),
    fetch("./vieneu/constants.json"),
    fetch("./vieneu/voices.json"),
  ]);
  vieneu.cfg = await resCfg.json();
  vieneu.constants = await resConst.json();
  vieneu.voices = await resVoices.json();
}

async function initVieneuNullContext() {
  const nullIds = BigInt64Array.from([
    BigInt(vieneu.cfg.bos_id),
    BigInt(vieneu.cfg.eos_id),
  ]);
  const nullIdsTensor = new ort.Tensor("int64", nullIds, [1, 2]);
  const nullStyle = new Float32Array(vieneu.constants.null_style.flat());
  const nullStyleTensor = new ort.Tensor("float32", nullStyle, [1, 50, 256]);
  const res = await vieneu.sText.run({
    ids: nullIdsTensor,
    style: nullStyleTensor,
  });
  vieneu.nullCtx = res.ctx;
  vieneu.nullMask = new ort.Tensor("bool", new Uint8Array([1, 1]), [1, 2]);
}

async function handleSynthesis(payload, id, isStream) {
  const reqEngine = payload.engine || activeEngine;
  if (reqEngine === "vieneu") {
    if (!vieneu.sText) await handleLoadVieneu(payload, id);
    await handleVieneuSynthesis(payload, id, isStream);
  } else {
    if (!kokoroSession) await handleLoadKokoro(payload, id);
    await handleKokoroSynthesis(payload, id, isStream);
  }
}

async function handleVieneuSynthesis(payload, id, isStream) {
  if (!vieneu.sText) throw new Error("VieNeu-TTS chưa được nạp!");
  const {
    chunks,
    voice,
    speed = 1.0,
    steps = 16,
    cfg = 3.0,
    customVieneuVoice,
  } = payload;
  const activeVoice =
    customVieneuVoice ||
    (vieneu.voices?.presets?.[voice] ? voice : "Minh Quân");
  const parts = [];
  const tGlobal = performance.now();

  for (let i = 0; i < chunks.length; i++) {
    const ch = chunks[i];
    const t0 = performance.now();
    const pcm = await runVieneuSampling(
      ch.phones || ch.text,
      activeVoice,
      speed,
      steps,
      cfg,
    );
    const calcMs = Math.round(performance.now() - t0);
    if (isStream) {
      sendStreamChunk(pcm, i, chunks.length, calcMs, id);
    } else {
      parts.push(pcm);
    }
  }
  finishSynthesis(parts, chunks.length, tGlobal, isStream, id, speed, chunks);
}

function sendStreamChunk(pcm, chunkIndex, totalChunks, calcMs, id) {
  self.postMessage(
    {
      type: "STREAM_CHUNK",
      chunkIndex,
      totalChunks,
      pcm: pcm.buffer,
      calcMs,
      sampleLength: pcm.length,
      id,
    },
    [pcm.buffer],
  );
}

function finishSynthesis(
  parts,
  totalChunks,
  tGlobal,
  isStream,
  id,
  speed = 1.0,
  chunks = [],
) {
  if (isStream) {
    self.postMessage({ type: "STREAM_COMPLETE", totalChunks, id });
  } else {
    const merged = mergePcmArrays(parts, chunks, speed);
    const elapsed = Math.round(performance.now() - tGlobal);
    self.postMessage(
      {
        type: "FULL_AUDIO_READY",
        pcm: merged.buffer,
        sampleLength: merged.length,
        elapsed,
        id,
      },
      [merged.buffer],
    );
  }
}

async function runVieneuSampling(phones, voiceOrName, speed, steps, cfg) {
  const voice =
    typeof voiceOrName === "object" && voiceOrName !== null
      ? voiceOrName
      : vieneu.voices.presets[voiceOrName] ||
        vieneu.voices.presets["Minh Quân"];
  const spkTensor = new ort.Tensor(
    "float32",
    new Float32Array(voice.speaker_emb),
    [1, 192],
  );
  const flatStyle = Array.isArray(voice.style[0])
    ? voice.style.flat()
    : voice.style;
  const styleTensor = new ort.Tensor(
    "float32",
    new Float32Array(flatStyle),
    [1, 50, 256],
  );

  const idsList = buildVieneuIds(phones);
  const idsTensor = new ort.Tensor(
    "int64",
    BigInt64Array.from(idsList.map(BigInt)),
    [1, idsList.length],
  );
  const maskTensor = new ort.Tensor(
    "bool",
    new Uint8Array(idsList.length).fill(1),
    [1, idsList.length],
  );

  const resText = await vieneu.sText.run({
    ids: idsTensor,
    style: styleTensor,
  });
  const ctx = resText.ctx;
  const resDur = await vieneu.sDur.run({
    ctx,
    ctx_mask: maskTensor,
    spk: spkTensor,
  });
  const secs = Math.min(
    Math.exp(resDur.log_seconds.data[0]) / Math.max(speed, 0.1),
    15.0,
  );
  const T = Math.max(2, Math.round(secs * (24000 / 256 / 6)));

  const xData = createGaussianNoise(144 * T);
  await runVieneuEulerLoop(
    xData,
    T,
    ctx,
    maskTensor,
    spkTensor,
    styleTensor,
    steps,
    cfg,
  );

  const resDec = await vieneu.sDec.run({
    x: new ort.Tensor("float32", xData, [1, 144, T]),
  });
  return trimAndFadeVieneu(new Float32Array(resDec.wav.data));
}

function buildVieneuIds(phones) {
  const vocab = vieneu.cfg.vocab;
  const ids = [vieneu.cfg.bos_id];
  for (const ch of phones) {
    if (vocab[ch] !== undefined) ids.push(vocab[ch]);
  }
  ids.push(vieneu.cfg.eos_id);
  return ids;
}

function createGaussianNoise(size) {
  const data = new Float32Array(size);
  for (let i = 0; i < size; i += 2) {
    const u1 = Math.max(Math.random(), 1e-7);
    const u2 = Math.random();
    const r = Math.sqrt(-2.0 * Math.log(u1));
    const th = 2.0 * Math.PI * u2;
    data[i] = r * Math.cos(th);
    if (i + 1 < size) data[i + 1] = r * Math.sin(th);
  }
  return data;
}

async function runVieneuEulerLoop(xData, T, ctx, mask, spk, style, steps, cfg) {
  const dt = 1.0 / steps;
  for (let step = 0; step < steps; step++) {
    const tVal = step / steps;
    await runSingleEulerStep(xData, T, tVal, dt, ctx, mask, spk, style, cfg);
  }
}

async function runSingleEulerStep(
  xData,
  T,
  tVal,
  dt,
  ctx,
  mask,
  spk,
  style,
  cfg,
) {
  const len = 144 * T;
  const tTensor = new ort.Tensor("float32", new Float32Array([tVal]), [1]);
  const xTensor = new ort.Tensor("float32", xData, [1, 144, T]);

  const resVe = await vieneu.sVe.run({
    x: xTensor,
    t: tTensor,
    ctx,
    ctx_mask: mask,
    spk,
    style,
  });
  let v = resVe.v.data;

  if (cfg > 0) {
    v = await applyEulerCfg(xTensor, tTensor, len, v, cfg);
  }
  for (let j = 0; j < len; j++) {
    xData[j] += dt * v[j];
  }
}

async function applyEulerCfg(xTensor, tTensor, len, vCond, cfg) {
  const nullSpk = new Float32Array(vieneu.constants.null_spk);
  const nullSpkTensor = new ort.Tensor("float32", nullSpk, [1, 192]);
  const nullStyle = new Float32Array(vieneu.constants.null_style.flat());
  const nullStyleTensor = new ort.Tensor("float32", nullStyle, [1, 50, 256]);

  const resVu = await vieneu.sVe.run({
    x: xTensor,
    t: tTensor,
    ctx: vieneu.nullCtx,
    ctx_mask: vieneu.nullMask,
    spk: nullSpkTensor,
    style: nullStyleTensor,
  });
  const vu = resVu.v.data;
  const vOut = new Float32Array(len);
  for (let j = 0; j < len; j++) {
    vOut[j] = vu[j] + cfg * (vCond[j] - vu[j]);
  }
  return vOut;
}

function removeDcOffset(pcm) {
  let sum = 0;
  const len = pcm.length;
  for (let i = 0; i < len; i++) sum += pcm[i];
  const mean = sum / len;
  if (Math.abs(mean) > 1e-4) {
    for (let i = 0; i < len; i++) pcm[i] -= mean;
  }
}

function findAcousticBoundaries(pcm, threshold = 0.0015, padMargin = 2400) {
  const len = pcm.length;
  const win = 240;
  if (len <= win) return { start: 0, end: len };
  let start = 0;
  for (let i = 0; i <= len - win; i += 60) {
    let sumSq = 0;
    for (let k = 0; k < win; k++) sumSq += pcm[i + k] * pcm[i + k];
    if (Math.sqrt(sumSq / win) > threshold) {
      const rawStart = i - padMargin;
      // Tránh cắt mất phụ âm xát/âm gió đầu câu nếu điểm phát âm sớm (< 150ms)
      start = rawStart <= 3600 ? 0 : Math.max(0, rawStart);
      break;
    }
  }
  let end = len;
  for (let i = len - win; i >= start; i -= 60) {
    let sumSq = 0;
    for (let k = 0; k < win; k++) sumSq += pcm[i + k] * pcm[i + k];
    if (Math.sqrt(sumSq / win) > threshold) {
      end = Math.min(len, i + win + padMargin);
      break;
    }
  }
  return { start, end };
}

function applyHannFade(pcm, fadeLen = 128) {
  const len = pcm.length;
  const actualFade = Math.min(fadeLen, Math.floor(len / 2));
  for (let i = 0; i < actualFade; i++) {
    const factor = 0.5 * (1 - Math.cos((Math.PI * i) / actualFade));
    pcm[i] *= factor;
    pcm[len - 1 - i] *= factor;
  }
  return pcm;
}

function findVieneuActiveEdges(wav, win, thresh, nWin) {
  let leadWin = 0;
  for (let i = 0; i < nWin; i++) {
    let sum = 0;
    const offset = i * win;
    for (let k = 0; k < win; k++) sum += Math.abs(wav[offset + k]);
    if (sum / win > thresh) {
      leadWin = i;
      break;
    }
  }

  let tailWin = 0;
  for (let i = nWin - 1; i >= leadWin; i--) {
    let sum = 0;
    const offset = i * win;
    for (let k = 0; k < win; k++) sum += Math.abs(wav[offset + k]);
    if (sum / win > thresh) {
      tailWin = nWin - 1 - i;
      break;
    }
  }
  return { leadWin, tailWin };
}

function applyVieneuCosineFade(out, sr) {
  const fadeLen = Math.min(Math.round(0.015 * sr), Math.floor(out.length / 2));
  for (let i = 0; i < fadeLen; i++) {
    const factor = 0.5 - 0.5 * Math.cos((Math.PI * i) / fadeLen);
    out[i] *= factor;
    out[out.length - 1 - i] *= factor;
  }
  return out;
}

function trimAndFadeVieneu(wav, sr = 24000) {
  if (!wav || wav.length < 240) return wav || new Float32Array(0);
  removeDcOffset(wav);
  const win = 240;
  const thresh = 0.0056234;
  const nWin = Math.floor(wav.length / win);
  if (nWin === 0) return wav;

  const { leadWin, tailWin } = findVieneuActiveEdges(wav, win, thresh, nWin);
  const keep = Math.round(0.04 * sr);
  const a = Math.max(0, leadWin * win - keep);
  const b = wav.length - Math.max(0, tailWin * win - keep);
  if (b <= a) return wav;

  const out = wav.slice(a, b);
  return applyVieneuCosineFade(out, sr);
}

function sanitizeTtsWaveform(rawPcm) {
  if (!rawPcm || rawPcm.length < 128) return rawPcm || new Float32Array(0);
  removeDcOffset(rawPcm);
  return applyHannFade(rawPcm, 128);
}

async function handleKokoroSynthesis(payload, id, isStream) {
  if (!kokoroSession) throw new Error("Kokoro Model chưa được nạp!");
  const {
    chunks,
    voice,
    speed = 1.0,
    noiseScale = 0.667,
    customVoiceBuffer,
  } = payload;
  const knownKokoro = [
    "ngoc_huyen",
    "hung_thinh",
    "diem_trinh",
    "huuduc",
    "mai_linh",
    "manh_dung",
    "anhtu_pro",
    "giahuy",
    "thientam",
    "thuc_trinh",
    "tuan_ngoc",
    "duc_an",
    "duc_duy",
    "mai_loan",
    "my_yen",
    "phat_tai",
    "thanh_dat",
    "storyvert",
    "giong_lai_ngoc_huyen_huuduc",
    "giong_lai_thientam",
    "anhtu_pro_kokoro",
  ];
  let safeVoice = voice;
  if (!customVoiceBuffer && (!safeVoice || !knownKokoro.includes(safeVoice))) {
    console.warn(
      `[Worker] Giọng '${voice}' không thuộc Kokoro, tự động chuẩn hóa về 'ngoc_huyen'`,
    );
    safeVoice = "ngoc_huyen";
  }
  let vpArray;
  if (customVoiceBuffer) {
    vpArray = new Float32Array(customVoiceBuffer);
  } else {
    vpArray = await fetchKokoroVoice(safeVoice);
  }
  const parts = [];
  const tGlobal = performance.now();

  for (let i = 0; i < chunks.length; i++) {
    const ch = chunks[i];
    const t0 = performance.now();
    const pcm = await executeKokoroForward(ch.tokens, vpArray, speed);
    const calcMs = Math.round(performance.now() - t0);
    if (isStream) {
      sendStreamChunk(pcm, i, chunks.length, calcMs, id);
    } else {
      parts.push(pcm);
    }
  }
  finishSynthesis(parts, chunks.length, tGlobal, isStream, id, speed, chunks);
}

async function fetchKokoroVoice(voiceName) {
  if (cachedKokoroVoices[voiceName]) return cachedKokoroVoices[voiceName];
  const buf = await fetchWithIdbCache(
    `kokoro_voice_${voiceName}`,
    `./kokoro_voices/${voiceName}.bin`,
    null,
    `Voicepack ${voiceName}`,
  );
  if (!buf) throw new Error(`Không thể nạp voicepack Kokoro: ${voiceName}`);
  const arr = new Float32Array(buf);
  cachedKokoroVoices[voiceName] = arr;
  return arr;
}

function buildKokoroFeeds(tokens, vpArray, speed) {
  const tokenLen = tokens.length - 2;
  const index = Math.min(Math.max(tokenLen - 1, 0), 509);
  const refStart = index * 256;
  const ref_s = vpArray.subarray(refStart, refStart + 256);

  const inputIdsTensor = new ort.Tensor(
    "int64",
    BigInt64Array.from(tokens.map(BigInt)),
    [1, tokens.length],
  );
  const refSTensor = new ort.Tensor("float32", ref_s, [1, 256]);
  const speedTensor = new ort.Tensor("float32", new Float32Array([speed]), []);

  return {
    feeds: { input_ids: inputIdsTensor, ref_s: refSTensor, speed: speedTensor },
    tensors: [inputIdsTensor, refSTensor, speedTensor],
  };
}

async function executeKokoroForward(tokens, vpArray, speed) {
  const { feeds, tensors } = buildKokoroFeeds(tokens, vpArray, speed);
  const results = await kokoroSession.run(feeds);
  const rawPcm = new Float32Array(results.waveform.data);
  tensors.forEach((t) => t.dispose());
  if (results.waveform?.dispose) results.waveform.dispose();
  return sanitizeTtsWaveform(rawPcm);
}

function mergePcmArrays(parts, chunks = [], speed = 1.0) {
  const safeSpeed = Math.max(0.25, Math.min(3.0, speed || 1.0));
  const pauses = [];
  let totalLen = 0;
  for (let i = 0; i < parts.length; i++) {
    totalLen += parts[i].length;
    if (i < parts.length - 1) {
      const basePause = chunks[i]?.pauseSec ?? 0.15;
      const pauseSec = Math.max(0.02, basePause / safeSpeed);
      const pauseSamples = Math.round(24000 * pauseSec);
      pauses.push(pauseSamples);
      totalLen += pauseSamples;
    }
  }
  const result = new Float32Array(totalLen);
  let offset = 0;
  for (let i = 0; i < parts.length; i++) {
    result.set(parts[i], offset);
    offset += parts[i].length;
    if (i < parts.length - 1) offset += pauses[i];
  }
  return result;
}

function postError(action, message, id) {
  self.postMessage({ type: "WORKER_ERROR", action, message, id });
}
