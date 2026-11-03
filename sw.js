// ==============================================================================
// Service Worker (sw.js) — Kokoro & VieNeu Offline Cache & PWA Layer (v4)
// Đảm bảo Web Client hoạt động 100% OFFLINE, 0 CẦN GỌI BẤT KỲ API NÀO.
// ==============================================================================

const CACHE_NAME = 'tts-offline-v5';

const CORE_ASSETS = [
  './',
  './index.html',
  './demo_worker_tts.html',
  './test_kokoro_wasm.html',
  './tts-worker.js',
  './vi_phoneme_dict.json',
  './ort.min.js',
  './ort-wasm-simd-threaded.mjs',
  './ort-wasm-simd-threaded.wasm',
  './vieneu/dict.json',
  './vieneu/config.json',
  './vieneu/constants.json'
];

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(cacheCoreAssets());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(cleanupOldCaches());
});

self.addEventListener('fetch', (event) => {
  event.respondWith(handleFetchRequest(event.request));
});

self.addEventListener('message', (event) => {
  handleClientMessage(event);
});

async function cacheCoreAssets() {
  const cache = await caches.open(CACHE_NAME);
  for (const url of CORE_ASSETS) {
    try {
      const res = await fetch(url);
      if (res && res.ok) await cache.put(url, res);
    } catch (err) {
      console.warn('[SW] Caching core asset notice:', url, err);
    }
  }
}

async function cleanupOldCaches() {
  await self.clients.claim();
  const keys = await caches.keys();
  const deletePromises = keys
    .filter((k) => k !== CACHE_NAME)
    .map((k) => caches.delete(k));
  await Promise.all(deletePromises);
  await reportCacheStatus();
}

function addCoopCoepHeaders(response) {
  if (!response || response.status === 0 || response.type === 'opaque') {
    return response;
  }
  const headers = new Headers(response.headers);
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: headers
  });
}

function resolveHfFallback(reqUrl) {
  const urlStr = reqUrl ? reqUrl.toString() : '';
  if (urlStr.includes('vector_estimator.onnx')) {
    return 'https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Nano/resolve/main/vector_estimator.onnx';
  }
  if (urlStr.includes('codec_decoder.onnx')) {
    return 'https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Nano/resolve/main/codec_decoder.onnx';
  }
  if (urlStr.includes('text_encoder.onnx')) {
    return 'https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Nano/resolve/main/text_encoder.onnx';
  }
  if (urlStr.includes('duration_predictor.onnx')) {
    return 'https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Nano/resolve/main/duration_predictor.onnx';
  }
  return null;
}

async function handleFetchRequest(request) {
  const cache = await caches.open(CACHE_NAME);
  const url = new URL(request.url);

  if (request.mode === 'navigate' || url.pathname.endsWith('.html') || url.pathname === '/' || url.pathname.endsWith('/web-tts/')) {
    try {
      const netRes = await fetch(request);
      if (netRes && netRes.status === 200) {
        cache.put(request, netRes.clone());
        return addCoopCoepHeaders(netRes);
      }
      return netRes;
    } catch (err) {
      const cached = await cache.match(request);
      if (cached) return addCoopCoepHeaders(cached);
      throw err;
    }
  }

  const cachedResponse = await cache.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }
  return fetchAndStore(cache, request);
}

async function fetchAndStore(cache, request) {
  try {
    let networkResponse = await fetch(request);
    if (!networkResponse || networkResponse.status === 404) {
      const fallbackUrl = resolveHfFallback(request.url);
      if (fallbackUrl) networkResponse = await fetch(fallbackUrl);
    }
    if (networkResponse && networkResponse.status === 200) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (err) {
    const fallback = await cache.match(request);
    if (fallback) return fallback;
    throw err;
  }
}

const VIENEU_ASSETS = [
  './vieneu/text_encoder.onnx',
  './vieneu/duration_predictor.onnx',
  './vieneu/vector_estimator.onnx',
  './vieneu/codec_decoder.onnx',
  './vieneu/constants.json',
  './vieneu/voices.json',
  './vieneu/dict.json',
  './vieneu/config.json'
];

async function broadcastToClients(msg) {
  try {
    const allClients = await self.clients.matchAll({
      includeUncontrolled: true,
      type: 'window'
    });
    for (const c of allClients) {
      c.postMessage(msg);
    }
  } catch (e) {
    console.warn('[SW] Broadcast failed:', e);
  }
}

function handleClientMessage(event) {
  const { action, payload } = event.data || {};
  if (action === 'QUERY_STATUS') {
    reportCacheStatus(event.source);
  } else if (action === 'PRECACHE_MODEL') {
    precacheModelAndVoices(event.source, payload);
  } else if (action === 'CLEAR_CACHE') {
    purgeCache(event.source);
  }
}

async function reportCacheStatus(client) {
  const cache = await caches.open(CACHE_NAME);
  const keys = await cache.keys();
  let totalBytes = 0;
  const cachedUrls = [];

  for (const req of keys) {
    cachedUrls.push(req.url);
    const res = await cache.match(req);
    if (res) {
      const cl = res.headers.get('content-length');
      totalBytes += cl ? parseInt(cl, 10) : 0;
    }
  }

  const msg = {
    type: 'SW_CACHE_STATUS',
    itemCount: keys.length,
    totalBytes: totalBytes,
    urls: cachedUrls
  };
  if (client) client.postMessage(msg);
  await broadcastToClients(msg);
}

function resolvePrecacheList(payload) {
  if (payload?.engine === 'vieneu') {
    return VIENEU_ASSETS;
  }
  const modelUrl = payload?.modelUrl || './kokoro_vi_fast_int8.onnx';
  const voices = payload?.voices || ['ngoc_huyen', 'hung_thinh', 'diem_trinh', 'huuduc'];
  const list = [modelUrl];
  voices.forEach((v) => list.push(`./kokoro_voices/${v}.bin`));
  return list;
}

async function precacheModelAndVoices(client, payload) {
  const cache = await caches.open(CACHE_NAME);
  const listToFetch = resolvePrecacheList(payload);

  for (let i = 0; i < listToFetch.length; i++) {
    const url = listToFetch[i];
    await cacheSingleUrl(cache, url, i + 1, listToFetch.length, client);
  }

  const doneMsg = { type: 'SW_PRECACHE_COMPLETE' };
  if (client) client.postMessage(doneMsg);
  await broadcastToClients(doneMsg);
  await reportCacheStatus(client);
}

async function cacheSingleUrl(cache, url, current, total, client) {
  try {
    let res = await fetch(url);
    if (!res || !res.ok) {
      const fb = resolveHfFallback(url);
      if (fb) res = await fetch(fb);
    }
    if (res && res.ok) {
      await cache.put(url, res.clone());
      const msg = { type: 'SW_PRECACHE_PROGRESS', url, current, total };
      if (client) client.postMessage(msg);
      await broadcastToClients(msg);
    }
  } catch (e) {
    console.warn('[SW] Failed to precache:', url, e);
  }
}

async function purgeCache(client) {
  await caches.delete(CACHE_NAME);
  const msg = { type: 'SW_CACHE_CLEARED' };
  if (client) client.postMessage(msg);
  await broadcastToClients(msg);
  await reportCacheStatus(client);
}
