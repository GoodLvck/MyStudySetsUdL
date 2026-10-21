/* ── Constants ── */
const SET_ID = document.body.dataset.set;
const MASTERY_KEY = `fswt_midterm_${SET_ID}_mastery`;
const THEME_KEY = 'fswt_midterm_theme';
const SESSION_KEY = `fswt_midterm_${SET_ID}_session`;

const LEVELS = [
  { label: 'Unknown', icon: '❓', cls: 'lvl0' },
  { label: 'Learning', icon: '📖', cls: 'lvl1' },
  { label: 'Familiar',    icon: '💡', cls: 'lvl2' },
  { label: 'Mastered',    icon: '✅', cls: 'lvl3' },
];

/* ── State ── */
let quizData, fillData, apuntesText = '';
let mode         = 'apuntes';
let isRandom     = false;
let currentIdx   = 0;
let questionOrder = [];
let mastery      = {};
let sessionAnswered = new Set();
let sessionStates   = {};

/* ── Init ── */
async function loadData(path, asText = false) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return asText ? response.text() : response.json();
}
function syncTabs() {
  document.querySelectorAll('.tab').forEach(tab => {
    const selected = tab.dataset.mode === mode;
    tab.classList.toggle('active', selected);
    tab.setAttribute('aria-selected', String(selected));
  });
}
function sourceRef(q) {
  if (!q.source) return '';
  const unit = q.source.file.endsWith('.pptx') ? 'slides' : 'pp.';
  return `<p class="source-ref">Source: ${esc(q.source.file.split('/').pop())} · ${unit} ${esc(q.source.pages)}</p>`;
}

async function init() {
  loadTheme();
  try {
    [quizData, fillData, apuntesText] = await Promise.all([
      loadData(`studysets/${SET_ID}_quiz.json`),
      loadData(`studysets/${SET_ID}_fillblanks.json`),
      loadData(`studysets/resumen_${SET_ID}.md`, true),
    ]);
  } catch (e) {
    document.getElementById('question-card').innerHTML =
      `<div class="error-msg">Error loading study material: ${esc(e.message)}</div>`;
    return;
  }
  loadMastery();
  restoreSession();
  syncTabs();
  toggleExerciseUI(mode !== 'apuntes');
  renderQuestion();
  renderMasteryStats();
  setupListeners();
}

/* ── Persistence ── */
function loadMastery() {
  try { mastery = JSON.parse(localStorage.getItem(MASTERY_KEY) || '{}'); }
  catch { mastery = {}; }
}
function saveMastery() {
  localStorage.setItem(MASTERY_KEY, JSON.stringify(mastery));
}
function getMastery(key)          { return mastery[key] ?? 0; }
function setMastery(key, correct) {
  const c = getMastery(key);
  mastery[key] = correct ? Math.min(3, c + 1) : Math.max(0, c - 1);
  saveMastery();
}

function saveSession() {
  const data = {
    mode, isRandom, currentIdx,
    questionOrder,
    sessionAnswered: [...sessionAnswered],
    sessionStates,
  };
  localStorage.setItem(SESSION_KEY, JSON.stringify(data));
}
function restoreSession() {
  try {
    const saved = JSON.parse(localStorage.getItem(SESSION_KEY) || 'null');
    if (!saved) { setupOrder(); return; }
    mode = ['quiz', 'fill', 'apuntes'].includes(saved.mode) ? saved.mode : 'apuntes';
    isRandom = saved.isRandom === true;
    const count = getQuestions().length;
    const order = saved.questionOrder;
    if (!Array.isArray(order) || order.length !== count || new Set(order).size !== count ||
        order.some(i => !Number.isInteger(i) || i < 0 || i >= count)) {
      setupOrder();
    } else {
      questionOrder = order;
      currentIdx = Math.max(0, Math.min(Number.isInteger(saved.currentIdx) ? saved.currentIdx : 0, count - 1));
      sessionStates = saved.sessionStates && typeof saved.sessionStates === 'object' ? saved.sessionStates : {};
      const validKeys = new Set(getQuestions().map(stateKey));
      sessionAnswered = new Set((Array.isArray(saved.sessionAnswered) ? saved.sessionAnswered : []).filter(k => validKeys.has(k)));
    }
    document.getElementById('random-toggle').classList.toggle('active', isRandom);
    updateProgress();
  } catch { mode = 'quiz'; setupOrder(); }
}

function loadTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  const theme = saved || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  applyTheme(theme);
}
function applyTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  document.getElementById('theme-toggle').textContent = t === 'dark' ? '☀️' : '🌙';
  localStorage.setItem(THEME_KEY, t);
}

/* ── Order ── */
function getQuestions() {
  if (mode === 'quiz')      return quizData.questions;
  return fillData.questions;
}
function setupOrder() {
  const qs = getQuestions();
  questionOrder = qs.map((_, i) => i);
  if (isRandom) shuffle(questionOrder);
  currentIdx = 0;
  sessionAnswered.clear();
  sessionStates = {};
  updateProgress();
}
function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
}
function getCurrentQuestion() {
  return getQuestions()[questionOrder[currentIdx]];
}
function modePrefix() {
  if (mode === 'quiz')      return 'q_';
  return 'f_';
}
function masteryKey(q) { return modePrefix() + q.id; }
function stateKey(q)   { return modePrefix() + q.id; }

/* ── Apuntes ── */
function toggleExerciseUI(show) {
  const d = show ? '' : 'none';
  document.querySelector('.controls').style.display = d;
  document.querySelector('.progress-section').style.display = d;
  document.querySelector('.navigation').style.display = d;
  document.getElementById('question-card').classList.toggle('apuntes-card', !show);
}

function renderApuntes() {
  const card = document.getElementById('question-card');
  card.innerHTML = window.marked
    ? `<div class="apuntes-view">${marked.parse(apuntesText)}</div>`
    : `<pre class="apuntes-view">${esc(apuntesText)}</pre>`;
}

/* ── Render ── */
function renderQuestion() {
  if (mode === 'apuntes') { renderApuntes(); saveSession(); return; }
  const q     = getCurrentQuestion();
  const total = questionOrder.length;
  document.getElementById('question-counter').textContent = `${currentIdx + 1} / ${total}`;
  document.getElementById('prev-btn').disabled = currentIdx === 0;
  document.getElementById('next-btn').disabled = currentIdx === total - 1;
  const card = document.getElementById('question-card');
  if (mode === 'quiz')          renderQuiz(q, card);
  else                          renderFill(q, card);
  saveSession();
}

function buildHeader(q) {
  const ml  = LEVELS[getMastery(masteryKey(q))];
  const dif = { easy: 'Easy', medium: 'Medium', hard: 'Hard' }[q.difficulty] || q.difficulty;
  return `
    <div class="question-header">
      <span class="topic-badge">${esc(q.topic)}</span>
      <span class="diff-badge ${q.difficulty}">${dif}</span>
      <span class="mastery-badge ${ml.cls}" id="mastery-badge">${ml.icon} ${ml.label}</span>
    </div>`;
}

/* Quiz */
function renderQuiz(q, card) {
  const prev = sessionStates[stateKey(q)];
  let optHtml = q.options.map(opt => {
    const k = opt[0];
    let cls = 'option';
    if (prev) {
      if (k === q.correctAnswer) cls += ' correct';
      else if (k === prev.selected) cls += ' wrong';
    }
    return `<button class="${cls}" data-key="${k}"${prev ? ' disabled' : ''}>${esc(opt)}</button>`;
  }).join('');
  const explHdr = prev ? (prev.correct ? '✅ Correct!' : '❌ Incorrect') : '';
  card.innerHTML = `
    ${buildHeader(q)}
    <p class="question-text">${esc(q.question)}</p>
    <div class="options" id="opts">${optHtml}</div>
    <div class="explanation${prev ? '' : ' hidden'}" id="expl">
      <span class="expl-header">${explHdr}</span>
      ${esc(q.explanation)}
      ${sourceRef(q)}
    </div>`;
  if (!prev) {
    card.querySelectorAll('.option').forEach(btn =>
      btn.addEventListener('click', () => onQuizAnswer(q, btn.dataset.key))
    );
  }
}

function onQuizAnswer(q, sel) {
  const correct = sel === q.correctAnswer;
  sessionStates[stateKey(q)] = { correct, selected: sel };
  sessionAnswered.add(stateKey(q));
  setMastery(masteryKey(q), correct);
  document.querySelectorAll('.option').forEach(b => {
    b.disabled = true;
    if (b.dataset.key === q.correctAnswer) b.classList.add('correct');
    else if (b.dataset.key === sel)         b.classList.add('wrong');
  });
  const expl = document.getElementById('expl');
  expl.querySelector('.expl-header').textContent = correct ? '✅ Correct!' : '❌ Incorrect';
  expl.classList.remove('hidden');
  refreshMasteryBadge(q);
  updateProgress();
  renderMasteryStats();
  saveSession();
}

/* Fill */
function renderFill(q, card) {
  const prev  = sessionStates[stateKey(q)];
  const parts = q.sentence.split('___');
  let sentHtml = '';
  parts.forEach((p, i) => {
    sentHtml += esc(p);
    if (i < parts.length - 1) {
      const val     = prev ? esc(prev.inputs[i]) : '';
      const cls     = prev ? (' ' + (prev.correct[i] ? 'correct' : 'wrong')) : '';
      const dis     = prev ? ' disabled' : '';
      const sz      = Math.max(8, (prev ? prev.inputs[i] : q.blanks[i]).length + 2);
      sentHtml += `<input type="text" class="blank-input${cls}" data-idx="${i}" value="${val}" aria-label="Blank ${i + 1}"
        size="${sz}" autocomplete="off" spellcheck="false"${dis}>`;
    }
  });
  const hintsHtml = q.hints.map((h, i) =>
    `<p><strong>Hint ${i + 1}:</strong> ${esc(h)}</p>`).join('');
  let revealHtml = '';
  if (prev) {
    revealHtml = `<div class="answer-reveal">` +
      q.blanks.map((b, i) => {
        const ok = prev.correct[i];
        const icon = ok ? '✓' : '✗';
        return `<span class="${ok ? 'ok' : 'bad'}">${icon} Blank ${i + 1}: <strong>${esc(b)}</strong>` +
          (!ok ? ` <em>(you wrote: "${esc(prev.inputs[i])}")</em>` : '') + `</span>`;
      }).join('<br>') + `</div>`;
  }
  const allOk   = prev && prev.correct.every(Boolean);
  const explHdr = prev ? (allOk ? '✅ All correct!' :
    `⚠️ ${prev.correct.filter(Boolean).length} / ${q.blanks.length} correct`) : '';
  card.innerHTML = `
    ${buildHeader(q)}
    <div class="sentence" id="sent">${sentHtml}</div>
    <div class="fill-controls">
      <button class="btn-secondary" id="hint-btn">💡 Hints</button>
      ${!prev ? '<button class="btn-primary" id="val-btn">✓ Check answers</button>' : ''}
    </div>
    <div class="hints-panel hidden" id="hints">${hintsHtml}</div>
    ${revealHtml}
    <div class="explanation${prev ? '' : ' hidden'}" id="expl">
      <span class="expl-header">${explHdr}</span>
      ${esc(q.explanation)}
      ${sourceRef(q)}
    </div>`;
  document.getElementById('hint-btn').addEventListener('click', () =>
    document.getElementById('hints').classList.toggle('hidden')
  );
  if (!prev) {
    document.getElementById('val-btn').addEventListener('click', () => onFillValidate(q));
    // Enter key on last input submits
    card.querySelectorAll('.blank-input').forEach(inp => {
      inp.addEventListener('keydown', e => {
        if (e.key === 'Enter') onFillValidate(q);
      });
    });
  }
}

function onFillValidate(q) {
  const inputs = [...document.querySelectorAll('.blank-input')];
  const vals   = inputs.map(i => i.value);
  const correct = vals.map((v, i) => [q.blanks[i], ...(q.acceptedAnswers?.[i] || [])].some(answer => norm(v) === norm(answer)));
  const allOk   = correct.every(Boolean);
  sessionStates[stateKey(q)] = { correct, inputs: vals };
  sessionAnswered.add(stateKey(q));
  setMastery(masteryKey(q), allOk);
  // Re-render so reveal section appears
  const card = document.getElementById('question-card');
  renderFill(q, card);
  updateProgress();
  renderMasteryStats();
  saveSession();
}

/* ── Helpers ── */
function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function norm(s) {
  return s.toLowerCase().trim()
    .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
    .replace(/[‐‑–—]/g, '-').replace(/\s+/g, ' ');
}
function updateProgress() {
  const total = questionOrder.length;
  const done  = sessionAnswered.size;
  const pct   = total ? (done / total) * 100 : 0;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-label').textContent = `${done} / ${total} answered`;
  if (mode !== 'apuntes' && done === total && total > 0) {
    showCompletion();
  }
}
function refreshMasteryBadge(q) {
  const ml = LEVELS[getMastery(masteryKey(q))];
  const badge = document.getElementById('mastery-badge');
  if (badge) { badge.textContent = `${ml.icon} ${ml.label}`; badge.className = `mastery-badge ${ml.cls}`; }
}
function renderMasteryStats() {
  const qs     = getQuestions();
  const prefix = modePrefix();
  const counts = [0, 0, 0, 0];
  qs.forEach(q => counts[getMastery(prefix + q.id)]++);
  const titles = ['Unknown', 'Learning', 'Familiar', 'Mastered'];
  document.getElementById('mastery-stats').innerHTML =
    LEVELS.map((l, i) =>
      `<span class="mastery-stat ${l.cls}" title="${titles[i]}: ${counts[i]}">${l.icon} ${counts[i]}</span>`
    ).join('');
}
function showCompletion() {
  const qs      = getQuestions();
  const prefix  = modePrefix();
  const counts  = [0, 0, 0, 0];
  qs.forEach(q => counts[getMastery(prefix + q.id)]++);
  let correct = 0;
  Object.values(sessionStates).forEach(s => {
    if (s.perfect !== undefined) { if (s.perfect) correct++; }
    else if (Array.isArray(s.correct)) { if (s.correct.every(Boolean)) correct++; }
    else if (s.correct) correct++;
  });
  document.getElementById('completion-stats').innerHTML =
    `<div class="completion-stat-row"><span>✅ Correct answers</span><strong>${correct} / ${qs.length}</strong></div>` +
    LEVELS.map((l, i) =>
      `<div class="completion-stat-row"><span>${l.icon} ${l.label}</span><strong>${counts[i]}</strong></div>`
    ).join('');
  document.getElementById('completion-overlay').classList.remove('hidden');
}

/* ── Event Listeners ── */
function setupListeners() {
  document.getElementById('close-completion').addEventListener('click', () => {
    document.getElementById('completion-overlay').classList.add('hidden');
  });
  document.getElementById('theme-toggle').addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    applyTheme(cur === 'dark' ? 'light' : 'dark');
  });

  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      if (tab.dataset.mode === mode) return;
      mode = tab.dataset.mode;
      syncTabs();
      document.getElementById('completion-overlay').classList.add('hidden');
      if (mode === 'apuntes') {
        toggleExerciseUI(false);
        setupOrder();
        renderApuntes();
        saveSession();
      } else {
        toggleExerciseUI(true);
        setupOrder();
        renderQuestion();
        renderMasteryStats();
      }
    });
  });

  document.getElementById('random-toggle').addEventListener('click', () => {
    isRandom = !isRandom;
    document.getElementById('random-toggle').classList.toggle('active', isRandom);
    setupOrder();
    renderQuestion();
  });

  document.getElementById('reset-session').addEventListener('click', () => {
    if (!confirm('Restart this session? Your session progress will be lost, but mastery levels will remain.')) return;
    setupOrder();
    renderQuestion();
    renderMasteryStats();
    document.getElementById('completion-overlay').classList.add('hidden');
  });

  document.getElementById('prev-btn').addEventListener('click', () => {
    if (currentIdx > 0) { currentIdx--; renderQuestion(); }
  });
  document.getElementById('next-btn').addEventListener('click', () => {
    if (currentIdx < questionOrder.length - 1) { currentIdx++; renderQuestion(); }
  });

  document.getElementById('restart-btn').addEventListener('click', () => {
    document.getElementById('completion-overlay').classList.add('hidden');
    setupOrder();
    renderQuestion();
    renderMasteryStats();
  });

  document.addEventListener('keydown', e => {
    if (mode === 'apuntes' || !document.getElementById('completion-overlay').classList.contains('hidden')) return;
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft')  document.getElementById('prev-btn').click();
    if (e.key === 'ArrowRight') document.getElementById('next-btn').click();
  });
}

init();
