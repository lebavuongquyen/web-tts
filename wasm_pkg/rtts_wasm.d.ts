/* tslint:disable */
/* eslint-disable */
/**
* @returns {string}
*/
export function rtts_version(): string;
/**
* 1. Chuẩn hóa văn bản tiếng Việt sang âm đọc thuần túy (WASM)
* @param {string} text
* @returns {string}
*/
export function normalize_vietnamese(text: string): string;
/**
* 2. Tách câu thông minh (Smart Chunking: 10 - 25 từ/câu)
* @param {string} text
* @param {number} max_words
* @returns {string}
*/
export function split_sentences_json(text: string, max_words: number): string;
/**
* 3. Pha trộn 2 vector giọng nói nhị phân (Float32Array) trong WASM
* @param {Float32Array} v1
* @param {Float32Array} v2
* @param {number} w1
* @param {number} w2
* @returns {Float32Array}
*/
export function blend_voice_vectors(v1: Float32Array, v2: Float32Array, w1: number, w2: number): Float32Array;
/**
* 4. Thuật toán DSP: Trích xuất Pitch F0, RMS, ZCR và sinh ma trận 30KB Float32 (10x768)
* @param {Float32Array} pcm_samples
* @param {number} sample_rate
* @returns {Float32Array}
*/
export function extract_dsp_voice_wasm(pcm_samples: Float32Array, sample_rate: number): Float32Array;
/**
* 5. Tạo nội dung phụ đề đồng bộ SRT
* @param {(string)[]} sentences
* @param {Float32Array} durations
* @returns {string}
*/
export function generate_srt_wasm(sentences: (string)[], durations: Float32Array): string;
/**
* 6. BỘ TẠO SÓNG ÂM THẬT (REAL WASM PCM SYNTHESIZER)
* Sinh mảng Float32 PCM 24.000 Hz trực tiếp trong RAM từ văn bản bất kỳ mà không cần file MP3 hay server!
* @param {string} text
* @param {number} pitch_hz
* @returns {Float32Array}
*/
export function synthesize_pcm_wasm(text: string, pitch_hz: number): Float32Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly blend_voice_vectors: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
  readonly extract_dsp_voice_wasm: (a: number, b: number, c: number, d: number) => void;
  readonly generate_srt_wasm: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly normalize_vietnamese: (a: number, b: number, c: number) => void;
  readonly rtts_version: (a: number) => void;
  readonly split_sentences_json: (a: number, b: number, c: number, d: number) => void;
  readonly synthesize_pcm_wasm: (a: number, b: number, c: number, d: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {SyncInitInput} module
*
* @returns {InitOutput}
*/
export function initSync(module: SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
