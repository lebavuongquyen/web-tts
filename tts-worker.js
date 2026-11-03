// ==============================================================================
// Web Worker (tts-worker.js) — Dual-Engine Real Offline AI TTS (Kokoro & VieNeu)
// 100% Client-Side WebAssembly (SIMD Multi-threaded), ZERO Backend API.
// ==============================================================================

/* global ort */
importScripts('./ort.min.js');

let currentThreads = 8;
let activeEngine = 'vieneu'; // 'kokoro' | 'vieneu'
let kokoroSession = null;
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
  nullMask: null
};

self.onmessage = async (e) => {
  const { action, payload, id } = e.data || {};
  try {
    if (action === 'INIT_CONFIG') {
      handleInitConfig(payload);
    } else if (action === 'LOAD_MODEL') {
      await handleLoadModel(payload, id);
    } else if (action === 'SYNTHESIZE_STREAM') {
      await handleSynthesis(payload, id, true);
    } else if (action === 'SYNTHESIZE_ALL') {
      await handleSynthesis(payload, id, false);
    }
  } catch (err) {
    postError(action, err.message, id);
  }
};

function handleInitConfig(payload) {
  const canThread = typeof SharedArrayBuffer !== 'undefined' && self.crossOriginIsolated;
  currentThreads = canThread ? (payload.threads || 8) : 1;
  ort.env.wasm.wasmPaths = './';
  ort.env.wasm.numThreads = currentThreads;
  ort.env.wasm.proxy = false;
  self.postMessage({ type: 'CONFIG_READY', threads: currentThreads });
}

async function fetchOrFallbackSession(localPath, fallbackHfUrl, sessionOptions) {
  try {
    return await ort.InferenceSession.create(localPath, sessionOptions);
  } catch (err) {
    if (fallbackHfUrl) {
      return await ort.InferenceSession.create(fallbackHfUrl, sessionOptions);
    }
    throw err;
  }
}

async function handleLoadModel(payload, id) {
  const engine = payload.engine || (payload.modelFile?.includes('kokoro') ? 'kokoro' : 'vieneu');
  if (engine === 'vieneu') {
    await handleLoadVieneu(payload, id);
  } else {
    await handleLoadKokoro(payload, id);
  }
}

async function handleLoadKokoro(payload, id) {
  const modelFile = payload.modelFile || 'kokoro_vi_fast_int8.onnx';
  ort.env.wasm.numThreads = currentThreads;
  const t0 = performance.now();
  kokoroSession = await ort.InferenceSession.create(`./${modelFile}`, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    enableCpuMemArena: true,
    enableMemPattern: true
  });
  activeEngine = 'kokoro';
  const elapsed = Math.round(performance.now() - t0);
  self.postMessage({
    type: 'MODEL_LOADED',
    engine: 'kokoro',
    modelFile,
    elapsed,
    threads: currentThreads,
    id
  });
}

async function handleLoadVieneu(payload, id) {
  const t0 = performance.now();
  ort.env.wasm.numThreads = currentThreads;
  const so = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  const hf = 'https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Nano/resolve/main/';

  notifyProgress(1, 4, '1/4: Đang nạp Text Encoder...');
  vieneu.sText = await fetchOrFallbackSession('./vieneu/text_encoder.onnx', hf + 'text_encoder.onnx', so);
  notifyProgress(2, 4, '2/4: Đang nạp Duration Predictor...');
  vieneu.sDur = await fetchOrFallbackSession('./vieneu/duration_predictor.onnx', hf + 'duration_predictor.onnx', so);
  notifyProgress(3, 4, '3/4: Đang nạp Vector Estimator (148MB)...');
  vieneu.sVe = await fetchOrFallbackSession('./vieneu/vector_estimator.onnx', hf + 'vector_estimator.onnx', so);
  notifyProgress(4, 4, '4/4: Đang nạp Codec Decoder (95MB)...');
  vieneu.sDec = await fetchOrFallbackSession('./vieneu/codec_decoder.onnx', hf + 'codec_decoder.onnx', so);

  await loadVieneuMetadata();
  await initVieneuNullContext();
  activeEngine = 'vieneu';
  const elapsed = Math.round(performance.now() - t0);

  self.postMessage({
    type: 'MODEL_LOADED',
    engine: 'vieneu',
    modelFile: 'VieNeu-TTS v3 Nano (4 Graphs)',
    elapsed,
    threads: currentThreads,
    id
  });
}

function notifyProgress(current, total, detail) {
  self.postMessage({ type: 'MODEL_LOAD_PROGRESS', current, total, detail });
}

async function loadVieneuMetadata() {
  const [resCfg, resConst, resVoices] = await Promise.all([
    fetch('./vieneu/config.json'),
    fetch('./vieneu/constants.json'),
    fetch('./vieneu/voices.json')
  ]);
  vieneu.cfg = await resCfg.json();
  vieneu.constants = await resConst.json();
  vieneu.voices = await resVoices.json();
}

async function initVieneuNullContext() {
  const nullIds = BigInt64Array.from([BigInt(vieneu.cfg.bos_id), BigInt(vieneu.cfg.eos_id)]);
  const nullIdsTensor = new ort.Tensor('int64', nullIds, [1, 2]);
  const nullStyle = new Float32Array(vieneu.constants.null_style.flat());
  const nullStyleTensor = new ort.Tensor('float32', nullStyle, [1, 50, 256]);
  const res = await vieneu.sText.run({ ids: nullIdsTensor, style: nullStyleTensor });
  vieneu.nullCtx = res.ctx;
  vieneu.nullMask = new ort.Tensor('bool', new Uint8Array([1, 1]), [1, 2]);
}

async function handleSynthesis(payload, id, isStream) {
  if (activeEngine === 'vieneu') {
    await handleVieneuSynthesis(payload, id, isStream);
  } else {
    await handleKokoroSynthesis(payload, id, isStream);
  }
}

async function handleVieneuSynthesis(payload, id, isStream) {
  if (!vieneu.sText) throw new Error('VieNeu-TTS chưa được nạp!');
  const { chunks, voice, speed = 1.0, steps = 6, cfg = 2.5 } = payload;
  const parts = [];
  const tGlobal = performance.now();

  for (let i = 0; i < chunks.length; i++) {
    const ch = chunks[i];
    const t0 = performance.now();
    const pcm = await runVieneuSampling(ch.phones || ch.text, voice, speed, steps, cfg);
    const calcMs = Math.round(performance.now() - t0);
    if (isStream) {
      sendStreamChunk(pcm, i, chunks.length, calcMs, id);
    } else {
      parts.push(pcm);
    }
  }
  finishSynthesis(parts, chunks.length, tGlobal, isStream, id);
}

function sendStreamChunk(pcm, chunkIndex, totalChunks, calcMs, id) {
  self.postMessage(
    {
      type: 'STREAM_CHUNK',
      chunkIndex,
      totalChunks,
      pcm: pcm.buffer,
      calcMs,
      sampleLength: pcm.length,
      id
    },
    [pcm.buffer]
  );
}

function finishSynthesis(parts, totalChunks, tGlobal, isStream, id) {
  if (isStream) {
    self.postMessage({ type: 'STREAM_COMPLETE', totalChunks, id });
  } else {
    const merged = mergePcmArrays(parts);
    const elapsed = Math.round(performance.now() - tGlobal);
    self.postMessage(
      {
        type: 'FULL_AUDIO_READY',
        pcm: merged.buffer,
        sampleLength: merged.length,
        elapsed,
        id
      },
      [merged.buffer]
    );
  }
}

async function runVieneuSampling(phones, voiceName, speed, steps, cfg) {
  const voice = vieneu.voices.presets[voiceName] || vieneu.voices.presets['Minh Quân'];
  const spkTensor = new ort.Tensor('float32', new Float32Array(voice.speaker_emb), [1, 192]);
  const styleTensor = new ort.Tensor('float32', new Float32Array(voice.style.flat()), [1, 50, 256]);

  const idsList = buildVieneuIds(phones);
  const idsTensor = new ort.Tensor('int64', BigInt64Array.from(idsList.map(BigInt)), [1, idsList.length]);
  const maskTensor = new ort.Tensor('bool', new Uint8Array(idsList.length).fill(1), [1, idsList.length]);

  const resText = await vieneu.sText.run({ ids: idsTensor, style: styleTensor });
  const ctx = resText.ctx;
  const resDur = await vieneu.sDur.run({ ctx, ctx_mask: maskTensor, spk: spkTensor });
  const secs = Math.min(Math.exp(resDur.log_seconds.data[0]) / Math.max(speed, 0.1), 15.0);
  const T = Math.max(2, Math.round(secs * (24000 / 256 / 6)));

  const xData = createGaussianNoise(144 * T);
  await runVieneuEulerLoop(xData, T, ctx, maskTensor, spkTensor, styleTensor, steps, cfg);

  const resDec = await vieneu.sDec.run({ x: new ort.Tensor('float32', xData, [1, 144, T]) });
  return trimAndFadePcm(new Float32Array(resDec.wav.data));
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

async function runSingleEulerStep(xData, T, tVal, dt, ctx, mask, spk, style, cfg) {
  const len = 144 * T;
  const tTensor = new ort.Tensor('float32', new Float32Array([tVal]), [1]);
  const xTensor = new ort.Tensor('float32', xData, [1, 144, T]);

  const resVe = await vieneu.sVe.run({ x: xTensor, t: tTensor, ctx, ctx_mask: mask, spk, style });
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
  const nullSpkTensor = new ort.Tensor('float32', nullSpk, [1, 192]);
  const nullStyle = new Float32Array(vieneu.constants.null_style.flat());
  const nullStyleTensor = new ort.Tensor('float32', nullStyle, [1, 50, 256]);

  const resVu = await vieneu.sVe.run({
    x: xTensor,
    t: tTensor,
    ctx: vieneu.nullCtx,
    ctx_mask: vieneu.nullMask,
    spk: nullSpkTensor,
    style: nullStyleTensor
  });
  const vu = resVu.v.data;
  const vOut = new Float32Array(len);
  for (let j = 0; j < len; j++) {
    vOut[j] = vu[j] + cfg * (vCond[j] - vu[j]);
  }
  return vOut;
}

function trimAndFadePcm(pcm, fadeLen = 240) {
  const len = pcm.length;
  for (let i = 0; i < Math.min(fadeLen, len); i++) {
    const f = i / fadeLen;
    pcm[i] *= f;
    pcm[len - 1 - i] *= f;
  }
  return pcm;
}

async function handleKokoroSynthesis(payload, id, isStream) {
  if (!kokoroSession) throw new Error('Kokoro Model chưa được nạp!');
  const { chunks, voice, speed = 1.0 } = payload;
  const vpArray = await fetchKokoroVoice(voice);
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
  finishSynthesis(parts, chunks.length, tGlobal, isStream, id);
}

async function fetchKokoroVoice(voiceName) {
  if (cachedKokoroVoices[voiceName]) return cachedKokoroVoices[voiceName];
  const res = await fetch(`./kokoro_voices/${voiceName}.bin`);
  if (!res.ok) throw new Error(`Không thể nạp voicepack Kokoro: ${voiceName}`);
  const buf = await res.arrayBuffer();
  const arr = new Float32Array(buf);
  cachedKokoroVoices[voiceName] = arr;
  return arr;
}

function buildKokoroFeeds(tokens, vpArray, speed) {
  const tokenLen = tokens.length - 2;
  const index = Math.min(Math.max(tokenLen - 1, 0), 509);
  const refStart = index * 256;
  const ref_s = vpArray.subarray(refStart, refStart + 256);

  const inputIdsTensor = new ort.Tensor('int64', BigInt64Array.from(tokens.map(BigInt)), [1, tokens.length]);
  const refSTensor = new ort.Tensor('float32', ref_s, [1, 256]);
  const speedTensor = new ort.Tensor('float32', new Float32Array([speed]), []);

  return { feeds: { input_ids: inputIdsTensor, ref_s: refSTensor, speed: speedTensor }, tensors: [inputIdsTensor, refSTensor, speedTensor] };
}

async function executeKokoroForward(tokens, vpArray, speed) {
  const { feeds, tensors } = buildKokoroFeeds(tokens, vpArray, speed);
  const results = await kokoroSession.run(feeds);
  const pcm = new Float32Array(results.waveform.data);
  tensors.forEach((t) => t.dispose());
  if (results.waveform?.dispose) results.waveform.dispose();
  return pcm;
}

function mergePcmArrays(parts) {
  const pauseSamples = Math.round(24000 * 0.15);
  let totalLen = 0;
  for (let i = 0; i < parts.length; i++) {
    totalLen += parts[i].length;
    if (i < parts.length - 1) totalLen += pauseSamples;
  }
  const result = new Float32Array(totalLen);
  let offset = 0;
  for (let i = 0; i < parts.length; i++) {
    result.set(parts[i], offset);
    offset += parts[i].length;
    if (i < parts.length - 1) offset += pauseSamples;
  }
  return result;
}

function postError(action, message, id) {
  self.postMessage({ type: 'WORKER_ERROR', action, message, id });
}
