/*
 * VoiceStudio Reader — live PDF narration in the browser.
 *
 * Flow (nothing waits for the whole document):
 *   1. pdf.js opens the PDF locally in the browser and extracts the text of a
 *      page only when narration (or the viewer) needs it — milliseconds per page.
 *   2. The page text is split into short chunks (a few sentences each). Each
 *      chunk is one request on the backend's streaming socket, /ws/tts.
 *   3. /ws/tts synthesizes sentence by sentence and streams PCM16 audio back;
 *      the Web Audio API schedules it gaplessly as it arrives.
 *   4. While one chunk plays, the next chunks are already queued on the socket,
 *      so generation stays ahead of playback. Requests stop once LOOKAHEAD_S
 *      seconds of audio are buffered, so stopping never wastes much GPU time.
 *   5. At the end of a page it rolls straight on to the next, until Stop or
 *      the end of the document.
 */
(() => {
  'use strict';

  const LOOKAHEAD_S = 25;          // stop requesting once this much audio is queued
  const MAX_INFLIGHT = 2;          // chunk requests queued on the socket at once
  const CHUNK_TARGET = 320;        // characters per TTS request (soft)
  const CHUNK_MAX = 520;           // hard ceiling for one request
  const MAX_CONSECUTIVE_ERRORS = 3;
  // Every backend URL is resolved against where this page was served from, so
  // the reader works at the backend root (/reader/) and behind a path prefix
  // such as the web build's dev proxy (/api/reader/), on whatever port.
  const READER_BASE = new URL('./', document.baseURI);           // …/reader/
  const API_BASE = new URL('../', READER_BASE);                  // backend root (or prefix)
  const apiUrl = (path) => new URL(path.replace(/^\//, ''), API_BASE).toString();
  const PDFJS = new URL('vendor/pdfjs/', READER_BASE).toString();

  pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS + 'pdf.worker.min.js';

  const $ = (id) => document.getElementById(id);
  const ui = {
    file: $('file-input'), voice: $('voice-select'), speed: $('speed'), speedOut: $('speed-out'),
    play: $('play-btn'), pause: $('pause-btn'), stop: $('stop-btn'),
    prev: $('prev-btn'), next: $('next-btn'), pageInput: $('page-input'), pageCount: $('page-count'),
    follow: $('follow'), wrap: $('canvas-wrap'), canvas: $('pdf-canvas'), empty: $('empty'),
    textTitle: $('text-title'), pageText: $('page-text'), status: $('status'), metrics: $('metrics'),
  };

  const doc = { pdf: null, key: null, name: '', numPages: 0, viewPage: 1, pages: new Map() };
  let renderTask = null;
  let showToken = 0;
  let audioCtx = null;
  let session = null;            // the active narration, or null
  let highlight = null;          // { page, idx } of the chunk being heard
  let backendReady = false;

  // ── small helpers ─────────────────────────────────────────────────────────
  function setStatus(text, isError = false) {
    ui.status.textContent = text;
    ui.status.classList.toggle('error', isError);
  }
  function store(key, value) { try { localStorage.setItem(key, value); } catch { /* storage off */ } }
  function recall(key) { try { return localStorage.getItem(key); } catch { return null; } }
  const clampPage = (n) => Math.min(Math.max(1, n | 0), doc.numPages || 1);

  // ── text extraction (one page at a time, cached) ─────────────────────────
  const PAGE_NUMBER_LINE = /^(?:page\s*)?[-–—\s]*\d{1,4}(?:\s*(?:of|\/)\s*\d{1,4})?[-–—\s]*$/i;
  const ENDS_WITH_PUNCT = /[.!?…:;,。！？"'”’)\]]$/;

  function cleanText(raw) {
    const text = raw
      .replace(/­/g, '')                       // soft hyphens
      .replace(/(\p{L})-\n(\p{Ll})/gu, '$1$2')      // re-join words hyphenated across lines
      .replace(/[ \t ]+/g, ' ');
    const lines = text.split('\n').map((l) => l.trim()).filter(Boolean)
      .filter((l) => !PAGE_NUMBER_LINE.test(l));    // running page numbers ("12", "Page 3", "3 of 40")
    // A short line with no closing punctuation followed by a capitalised line is a
    // heading ("Chapter 2", "The Letter"): end it with a full stop so the voice
    // pauses there instead of running it into the first sentence.
    return lines.map((l, i) => {
      const next = lines[i + 1];
      const heading = l.length <= 60 && !ENDS_WITH_PUNCT.test(l) && next && /^[\p{Lu}\d]/u.test(next);
      return heading ? l + '.' : l;
    }).join(' ').trim();
  }

  // Sentence terminators: . ! ? … plus the CJK full stop / exclamation / question marks.
  const SENTENCE_RE = /[^.!?…。！？]+(?:[.!?…。！？]+["'”’»)\]]*\s*|$)/g;

  function hardSplit(sentence) {
    const parts = [];
    let s = sentence;
    while (s.length > CHUNK_MAX) {
      let cut = Math.max(s.lastIndexOf(', ', CHUNK_MAX), s.lastIndexOf('; ', CHUNK_MAX));
      if (cut < CHUNK_TARGET / 2) cut = s.lastIndexOf(' ', CHUNK_MAX);
      if (cut <= 0) cut = CHUNK_MAX;
      parts.push(s.slice(0, cut + 1).trim());
      s = s.slice(cut + 1);
    }
    if (s.trim()) parts.push(s.trim());
    return parts;
  }

  function splitChunks(text) {
    const sentences = (text.match(SENTENCE_RE) || []).map((x) => x.trim()).filter(Boolean);
    const chunks = [];
    let cur = '';
    for (const sentence of sentences) {
      for (const piece of sentence.length > CHUNK_MAX ? hardSplit(sentence) : [sentence]) {
        if (cur && cur.length + piece.length + 1 > CHUNK_TARGET) { chunks.push(cur); cur = ''; }
        cur = cur ? cur + ' ' + piece : piece;
      }
    }
    if (cur) chunks.push(cur);
    // Only keep chunks that contain something speakable (not bare page numbers/punctuation).
    return chunks.filter((c) => /\p{L}/u.test(c));
  }

  function pageChunks(n) {
    if (!doc.pages.has(n)) {
      const pdf = doc.pdf;
      const p = pdf.getPage(n)
        .then((page) => page.getTextContent())
        .then((tc) => {
          let raw = '';
          for (const it of tc.items) {
            if (typeof it.str === 'string') { raw += it.str; if (it.hasEOL) raw += '\n'; }
          }
          return splitChunks(cleanText(raw));
        });
      p.catch(() => { if (doc.pdf === pdf) doc.pages.delete(n); });
      doc.pages.set(n, p);
    }
    return doc.pages.get(n);
  }

  // ── PDF view ─────────────────────────────────────────────────────────────
  async function openFile(file) {
    stopNarration('');
    setStatus(`Opening ${file.name}…`);
    try {
      const data = new Uint8Array(await file.arrayBuffer());
      const pdf = await pdfjsLib.getDocument({
        data,
        cMapUrl: PDFJS + 'cmaps/',
        cMapPacked: true,
        standardFontDataUrl: PDFJS + 'standard_fonts/',
      }).promise;
      if (doc.pdf) doc.pdf.destroy();
      Object.assign(doc, { pdf, name: file.name, numPages: pdf.numPages, key: `reader:${file.name}:${file.size}` });
      doc.pages = new Map();
      ui.pageCount.textContent = String(pdf.numPages);
      ui.pageInput.max = String(pdf.numPages);
      ui.pageInput.disabled = false;
      ui.empty.hidden = true;
      ui.canvas.hidden = false;
      const saved = parseInt(recall(doc.key) || '', 10);
      const start = saved > 1 && saved <= pdf.numPages ? saved : 1;
      await showPage(start);
      setStatus(start > 1
        ? `Opened ${file.name} (${pdf.numPages} pages) — back at page ${start}, where you left off.`
        : `Opened ${file.name} (${pdf.numPages} pages). Pick a page and press Read this page.`);
    } catch (e) {
      setStatus(`Couldn't open this PDF: ${e && e.message ? e.message : e}`, true);
    }
    updateButtons();
  }

  // Which page of which document the viewer currently shows (or is drawing).
  let shown = { pdf: null, page: 0 };

  async function showPage(n) {
    if (!doc.pdf) return;
    n = clampPage(n);
    // Asking for the page already on screen must not redraw it: the page box
    // fires `change` when it loses focus, i.e. on the mousedown of a click on a
    // sentence; re-creating the sentences there swallowed that click.
    if (shown.pdf === doc.pdf && shown.page === n) {
      ui.pageInput.value = String(n);
      return;
    }
    shown = { pdf: doc.pdf, page: n };
    const token = ++showToken;
    doc.viewPage = n;
    ui.pageInput.value = String(n);
    ui.textTitle.textContent = `Page ${n} text`;
    updateButtons();
    await Promise.all([renderCanvas(n, token), renderText(n, token)]);
  }

  async function renderCanvas(n, token) {
    const page = await doc.pdf.getPage(n);
    if (token !== showToken) return;
    const dpr = window.devicePixelRatio || 1;
    const avail = Math.max(200, ui.wrap.clientWidth - 32);
    const base = page.getViewport({ scale: 1 });
    const scale = Math.min(2.5, avail / base.width);
    const vp = page.getViewport({ scale: scale * dpr });
    if (renderTask) renderTask.cancel();
    const canvas = ui.canvas;
    canvas.width = Math.floor(vp.width);
    canvas.height = Math.floor(vp.height);
    canvas.style.width = `${Math.floor(vp.width / dpr)}px`;
    canvas.style.height = `${Math.floor(vp.height / dpr)}px`;
    const task = page.render({ canvasContext: canvas.getContext('2d'), viewport: vp });
    renderTask = task;
    try {
      await task.promise;
    } catch (e) {
      if (!e || e.name !== 'RenderingCancelledException') throw e;
    } finally {
      if (renderTask === task) renderTask = null;
    }
  }

  async function renderText(n, token) {
    let chunks;
    try {
      chunks = await pageChunks(n);
    } catch (e) {
      chunks = null;
    }
    if (token !== showToken) return;
    const box = ui.pageText;
    box.replaceChildren();
    if (!chunks || !chunks.length) {
      const p = document.createElement('p');
      p.className = 'note';
      p.textContent = chunks
        ? 'No readable text on this page (it may be a scanned image). Narration skips it.'
        : 'Could not read the text of this page.';
      box.append(p);
      return;
    }
    chunks.forEach((c, i) => {
      const span = document.createElement('span');
      span.className = 'chunk';
      span.dataset.idx = String(i);
      span.textContent = c + ' ';
      box.append(span);
    });
    if (session) {
      for (const it of session.failed) {
        if (it.page === n) box.querySelector(`.chunk[data-idx="${it.idx}"]`)?.classList.add('failed');
      }
    }
    box.scrollTop = 0;
    applyHighlight(true);
  }

  function applyHighlight(scroll) {
    for (const el of ui.pageText.querySelectorAll('.chunk.playing')) el.classList.remove('playing');
    if (!highlight || highlight.page !== doc.viewPage) return;
    const el = ui.pageText.querySelector(`.chunk[data-idx="${highlight.idx}"]`);
    if (!el) return;
    el.classList.add('playing');
    if (scroll) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  // ── voices ───────────────────────────────────────────────────────────────
  async function loadVoices() {
    try {
      const res = await fetch(apiUrl('/profiles'));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const profiles = await res.json();
      ui.voice.replaceChildren();
      for (const p of profiles) {
        const o = document.createElement('option');
        o.value = p.id;
        o.textContent = p.language ? `${p.name} (${p.language})` : p.name;
        ui.voice.append(o);
      }
      const def = document.createElement('option');
      def.value = '';
      def.textContent = 'Engine default (may vary)';
      ui.voice.append(def);
      const saved = recall('reader:voice');
      const hasSaved = saved !== null && [...ui.voice.options].some((o) => o.value === saved);
      // A saved voice wins; otherwise prefer a cloned voice (consistent timbre across chunks).
      const firstClone = profiles.find((p) => p.ref_audio_path || p.locked_audio_path);
      ui.voice.value = hasSaved ? saved : firstClone ? firstClone.id : (profiles[0]?.id ?? '');
      return true;
    } catch (e) {
      ui.voice.replaceChildren(new Option('Engine default', ''));
      setStatus(`Couldn't load voices (${e.message}); using the engine default voice.`, true);
      return false;
    }
  }

  // ── narration ────────────────────────────────────────────────────────────
  async function startNarration(page, idx = 0) {
    if (!doc.pdf || !backendReady) return;
    stopNarration('');
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    await audioCtx.resume();

    const wsUrl = new URL(apiUrl('/ws/tts'));
    wsUrl.protocol = wsUrl.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(wsUrl.toString());
    ws.binaryType = 'arraybuffer';
    const s = {
      ws,
      cursor: { page: clampPage(page), idx },
      pending: [],        // requests sent, waiting for their "start" frame
      current: null,      // request whose audio is arriving now
      items: [],          // requests with scheduled audio (for highlighting)
      failed: [],
      sources: new Set(),
      nextTime: 0,
      sampleRate: 24000,
      exhausted: false,   // no more text to request
      pumping: false,
      errors: 0,
      startedAt: performance.now(),
      firstAudioAt: null,
      lastRtf: null,
      engine: '',
      underruns: 0,
      playingPage: null,
      startPage: clampPage(page),
    };
    session = s;
    highlight = null;
    ws.onopen = () => pump(s);
    ws.onmessage = (ev) => onMessage(s, ev);
    ws.onerror = () => {};
    ws.onclose = () => {
      if (session === s) stopNarration('Lost the connection to the VoiceStudio backend.', true);
    };
    setStatus(`Starting narration at page ${s.startPage}… (the first run loads the voice model, which can take a minute)`);
    ui.metrics.textContent = '';
    updateButtons();
  }

  function aheadSeconds(s) {
    return Math.max(0, s.nextTime - audioCtx.currentTime);
  }

  async function nextChunk(s) {
    while (s.cursor.page <= doc.numPages) {
      const { page } = s.cursor;
      let chunks;
      try {
        chunks = await pageChunks(page);
      } catch {
        chunks = [];
      }
      if (session !== s) return null;
      if (s.cursor.idx < chunks.length) {
        const idx = s.cursor.idx++;
        if (s.cursor.idx >= chunks.length && page < doc.numPages) pageChunks(page + 1).catch(() => {}); // warm next page
        return { page, idx, text: chunks[idx], start: null, end: null };
      }
      s.cursor.page += 1;
      s.cursor.idx = 0;
    }
    return null;
  }

  async function pump(s) {
    if (s.pumping) return;
    s.pumping = true;
    try {
      while (
        session === s && !s.exhausted && s.ws.readyState === WebSocket.OPEN
        && s.pending.length + (s.current ? 1 : 0) < MAX_INFLIGHT
        && aheadSeconds(s) < LOOKAHEAD_S
      ) {
        const item = await nextChunk(s);
        if (session !== s) return;
        if (!item) { s.exhausted = true; break; }
        s.pending.push(item);
        const req = { text: item.text, speed: Number(ui.speed.value) };
        if (ui.voice.value) req.voice = ui.voice.value;
        s.ws.send(JSON.stringify(req));
      }
    } finally {
      s.pumping = false;
    }
  }

  function onMessage(s, ev) {
    if (session !== s) return;
    if (typeof ev.data !== 'string') { scheduleAudio(s, ev.data); return; }
    let m;
    try { m = JSON.parse(ev.data); } catch { return; }
    switch (m.type) {
      case 'start':
        s.current = s.pending.shift() || null;
        if (m.sample_rate) s.sampleRate = m.sample_rate;
        if (m.engine) s.engine = m.engine;
        if (s.current) s.items.push(s.current);
        break;
      case 'routing':
        setStatus(`Note: ${m.reason || m.status}`);
        break;
      case 'done':
        s.current = null;
        s.errors = 0;
        if (typeof m.rtf === 'number') s.lastRtf = m.rtf;
        pump(s);
        break;
      case 'error': {
        const item = s.current || s.pending.shift();
        s.current = null;
        s.errors += 1;
        let detail = m.detail || 'speech generation failed';
        if (m.code) detail += ' — accept the model licence in the main VoiceStudio app, then try again';
        if (item) {
          s.failed.push(item);
          if (item.page === doc.viewPage) {
            ui.pageText.querySelector(`.chunk[data-idx="${item.idx}"]`)?.classList.add('failed');
          }
        }
        if (s.errors >= MAX_CONSECUTIVE_ERRORS || m.code) {
          stopNarration(`Narration stopped: ${detail}`, true);
        } else {
          setStatus(`Skipped a passage on page ${item ? item.page : '?'}: ${detail}`, true);
          pump(s);
        }
        break;
      }
      default:
        break;
    }
  }

  function scheduleAudio(s, buf) {
    const item = s.current;
    if (!item || !buf.byteLength) return;
    const pcm = new Int16Array(buf, 0, buf.byteLength >> 1);
    const f32 = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 32768;
    const ab = audioCtx.createBuffer(1, f32.length, s.sampleRate);
    ab.copyToChannel(f32, 0);
    const src = audioCtx.createBufferSource();
    src.buffer = ab;
    src.connect(audioCtx.destination);

    const now = audioCtx.currentTime;
    if (s.nextTime < now + 0.03) {
      if (s.firstAudioAt !== null) s.underruns += 1;
      s.nextTime = now + 0.08;
    }
    src.start(s.nextTime);
    if (item.start === null) item.start = s.nextTime;
    item.end = s.nextTime + ab.duration;
    s.nextTime += ab.duration;
    s.sources.add(src);
    src.onended = () => s.sources.delete(src);
    if (s.firstAudioAt === null) s.firstAudioAt = performance.now();
  }

  function stopNarration(message, isError = false) {
    const s = session;
    if (!s) { if (message) setStatus(message, isError); return; }
    session = null;
    s.ws.onmessage = s.ws.onclose = s.ws.onopen = null;
    try { s.ws.close(); } catch { /* already closed */ }
    for (const src of s.sources) { try { src.stop(); } catch { /* not started */ } }
    s.sources.clear();
    if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
    highlight = null;
    applyHighlight(false);
    if (message !== '') setStatus(message === undefined ? 'Stopped.' : message, isError);
    updateButtons();
  }

  function togglePause() {
    if (!session || !audioCtx) return;
    if (audioCtx.state === 'running') audioCtx.suspend().then(updateButtons);
    else audioCtx.resume().then(updateButtons);
  }

  // Runs ~7×/s: highlight, follow pages, refill the request queue, detect the end.
  function tick() {
    const s = session;
    if (!s || !audioCtx) return;
    const t = audioCtx.currentTime;
    const paused = audioCtx.state !== 'running';

    const playing = s.items.find((it) => it.start !== null && t >= it.start && t < it.end);
    if (playing) {
      if (!highlight || highlight.page !== playing.page || highlight.idx !== playing.idx) {
        highlight = { page: playing.page, idx: playing.idx };
        if (playing.page !== s.playingPage) {
          s.playingPage = playing.page;
          store(doc.key, String(playing.page));
          if (ui.follow.checked && playing.page !== doc.viewPage) showPage(playing.page);
        }
        applyHighlight(true);
      }
    }
    // Drop finished items so the list stays small over a 200-page book.
    if (s.items.length > 50) s.items = s.items.filter((it) => it.end === null || it.end > t - 1);

    if (!paused) pump(s);

    const generating = s.pending.length > 0 || s.current !== null;
    if (s.exhausted && !generating && t >= s.nextTime) {
      stopNarration('Finished — reached the end of the document.');
      return;
    }

    if (paused) {
      setStatus(`Paused on page ${s.playingPage ?? s.startPage}.`);
    } else if (s.firstAudioAt === null) {
      const secs = Math.round((performance.now() - s.startedAt) / 1000);
      setStatus(`Preparing voice… ${secs}s (the first run loads the model; later starts are quick)`);
    } else if (!playing && generating) {
      setStatus(`Buffering page ${s.cursor.page}… (this GPU is generating slower than real time)`);
    } else if (playing) {
      setStatus(`Reading page ${playing.page} of ${doc.numPages}`);
    }

    const parts = [];
    if (s.engine) parts.push(`engine ${s.engine}`);
    if (s.firstAudioAt !== null) parts.push(`first audio ${((s.firstAudioAt - s.startedAt) / 1000).toFixed(1)}s`);
    if (s.lastRtf !== null) parts.push(`speed ${s.lastRtf > 0 ? (1 / s.lastRtf).toFixed(1) : '–'}× real time`);
    parts.push(`buffered ${aheadSeconds(s).toFixed(0)}s`);
    if (s.underruns) parts.push(`gaps ${s.underruns}`);
    ui.metrics.textContent = parts.join(' · ');
  }
  setInterval(tick, 150);

  // ── controls ─────────────────────────────────────────────────────────────
  function updateButtons() {
    const loaded = !!doc.pdf;
    ui.play.disabled = !loaded || !backendReady;
    ui.pause.disabled = !session;
    ui.stop.disabled = !session;
    ui.pause.textContent = session && audioCtx && audioCtx.state !== 'running' ? '▶ Resume' : '❚❚ Pause';
    ui.prev.disabled = !loaded || doc.viewPage <= 1;
    ui.next.disabled = !loaded || doc.viewPage >= doc.numPages;
  }

  // Restart from what is being heard, so a voice/speed change applies immediately
  // instead of after the ~25 s already buffered.
  function restartFromHere() {
    if (!session) return;
    const from = highlight || { page: session.startPage, idx: 0 };
    startNarration(from.page, from.idx);
  }

  ui.file.addEventListener('change', () => { if (ui.file.files[0]) openFile(ui.file.files[0]); ui.file.value = ''; });
  ui.play.addEventListener('click', () => startNarration(doc.viewPage, 0));
  ui.pause.addEventListener('click', togglePause);
  ui.stop.addEventListener('click', () => stopNarration());
  ui.prev.addEventListener('click', () => showPage(doc.viewPage - 1));
  ui.next.addEventListener('click', () => showPage(doc.viewPage + 1));
  ui.pageInput.addEventListener('change', () => showPage(parseInt(ui.pageInput.value, 10) || 1));
  ui.pageText.addEventListener('click', (e) => {
    const el = e.target.closest('.chunk');
    if (el) startNarration(doc.viewPage, Number(el.dataset.idx));
  });
  ui.voice.addEventListener('change', () => { store('reader:voice', ui.voice.value); restartFromHere(); });
  ui.speed.addEventListener('input', () => { ui.speedOut.textContent = `${Number(ui.speed.value).toFixed(2)}×`; });
  ui.speed.addEventListener('change', restartFromHere);

  document.addEventListener('keydown', (e) => {
    if (e.target.closest && e.target.closest('input, select, textarea')) return;
    if (e.key === ' ') {
      e.preventDefault();
      if (session) togglePause(); else if (!ui.play.disabled) startNarration(doc.viewPage, 0);
    } else if (e.key === 'ArrowLeft') showPage(doc.viewPage - 1);
    else if (e.key === 'ArrowRight') showPage(doc.viewPage + 1);
    else if (e.key === 'Escape') stopNarration();
  });

  // Drag-and-drop a PDF anywhere on the page.
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('drop', (e) => {
    e.preventDefault();
    const f = e.dataTransfer && e.dataTransfer.files[0];
    if (f) openFile(f);
  });

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (doc.pdf) renderCanvas(doc.viewPage, showToken); }, 200);
  });

  // ── backend readiness ────────────────────────────────────────────────────
  async function waitForBackend() {
    for (;;) {
      try {
        const res = await fetch(apiUrl('/health'), { cache: 'no-store' });
        if (res.ok) break;
        setStatus('VoiceStudio backend is starting…');
      } catch {
        setStatus('Waiting for the VoiceStudio backend…', true);
      }
      await new Promise((r) => setTimeout(r, 2000));
    }
    backendReady = true;
    const voicesOk = await loadVoices();
    if (voicesOk && !doc.pdf) setStatus('Ready — open a PDF (or drop one on this page).');
    updateButtons();
  }

  updateButtons();
  waitForBackend();
})();
