const themeKey = 'cg_theme';
const toggle = document.getElementById('theme-toggle');
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  toggle.textContent = theme === 'dark' ? '☀️' : '🌙';
}
applyTheme(localStorage.getItem(themeKey) || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
toggle.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  applyTheme(next); localStorage.setItem(themeKey, next);
});
(async () => {
  const content = document.getElementById('document-content');
  try {
    const response = await fetch(`studysets/${document.body.dataset.document}.md`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    content.innerHTML = marked.parse(await response.text());
  } catch (error) { content.textContent = `Unable to load study material: ${error.message}`; }
})();
