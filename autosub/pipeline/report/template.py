HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
  h1 {{ color: #fff; margin-bottom: 16px; }}
  .stats-grid {{ display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }}
  .stat-card {{ background: #16213e; border-radius: 8px; padding: 16px; flex: 1; min-width: 150px; }}
  .stat-value {{ font-size: 1.5em; font-weight: bold; color: #fff; }}
  .stat-label {{ font-size: 0.85em; color: #888; margin-top: 4px; }}
  #sticky-top {{ position: sticky; top: 0; z-index: 10; background: #1a1a2e; padding-bottom: 4px; }}
  .filters {{ margin: 8px 0; display: flex; gap: 8px; flex-wrap: wrap; }}
  .filter-btn {{ border: none; border-radius: 6px; padding: 8px 16px; cursor: pointer;
                 font-size: 0.85em; font-weight: 600; transition: opacity 0.2s; }}
  .filter-btn.inactive {{ opacity: 0.35; }}
  .btn-all {{ background: #555; color: #fff; }}
  .btn-issues {{ background: #333; color: #fff; border: 1px solid #666; }}
  .btn-orange {{ background: #ff9933; color: #fff; }}
  .btn-yellow {{ background: #e6c300; color: #1a1a2e; }}
  .btn-red {{ background: #ff4444; color: #fff; }}
  .btn-purple {{ background: #b366ff; color: #fff; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.85em; }}
  th {{ background: #0f3460; color: #fff; padding: 10px 8px; text-align: left;
       position: sticky; top: 0; z-index: 2; }}
  td {{ padding: 8px; border-bottom: 1px solid #222; vertical-align: top; }}
  tr {{ cursor: pointer; transition: background 0.15s; }}
  tr:hover {{ background: #1e2d4a !important; }}
  tr.selected {{ outline: 2px solid #4d94ff; }}
  tr.active-line {{ background: rgba(77,148,255,0.15) !important; }}
  .severity-orange {{ background: rgba(255,153,51,0.12); }}
  .severity-yellow {{ background: rgba(230,195,0,0.10); }}
  .severity-red {{ background: rgba(255,68,68,0.12); }}
  .severity-purple {{ background: rgba(179,102,255,0.10); }}
  .jp-text {{ color: #ffcc66; }}
  .en-text {{ color: #66ccff; }}
  .hidden-row {{ display: none !important; }}
  #player-section {{ display: flex; gap: 16px; align-items: stretch; padding: 8px 0; }}
  #video-wrap {{ width: 420px; min-width: 200px; flex-shrink: 0; resize: horizontal; overflow: hidden; }}
  #video {{ width: 100%; background: #000; border-radius: 8px; display: block; }}
  #now-playing {{ flex: 1; background: #16213e; border-radius: 8px; padding: 10px 14px;
                  border: 1px solid #2a3a5c; display: flex; flex-direction: column; gap: 6px; }}
  #now-header {{ display: flex; justify-content: space-between; margin-bottom: 4px; }}
  #now-line-num {{ font-size: 0.75em; color: #4d94ff; font-weight: 600; }}
  #now-time {{ font-size: 0.75em; color: #888; font-family: monospace; }}
  .now-text {{ font-size: 1em; line-height: 1.5; background: #0d1a30;
              border: 1px solid #2a3a5c; border-radius: 6px; padding: 10px 12px;
              white-space: pre-wrap; word-break: break-word; flex: 1; overflow-y: auto; }}
  #now-jp {{ color: #ffcc66; }}
  #now-en {{ color: #66ccff; }}
</style>
</head>
"""

HTML_BODY_TEMPLATE = """\
<body>
<h1>Translation Comparison &mdash; {title}</h1>
{stats_html}
<div id="sticky-top">
{video_section}
<div class="filters">
  <button class="filter-btn btn-all" onclick="filterAll()">All</button>
  <button class="filter-btn btn-issues" onclick="filterIssuesOnly()">Issues Only</button>
  {filter_buttons}
</div>
</div>
<table>
<thead><tr>
  <th>#</th><th>Start</th><th>End</th><th>Duration</th>
  <th>Style</th><th>Japanese</th><th>English</th><th>Chars (JP/EN)</th>
</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
"""

HTML_SCRIPT = """\
<script>
const video = document.getElementById('video');
const nowLineNum = document.getElementById('now-line-num');
const nowTime = document.getElementById('now-time');
const nowJp = document.getElementById('now-jp');
const nowEn = document.getElementById('now-en');
const rows = document.querySelectorAll('tbody tr');
let selectedRow = null;
let activeRow = null;

function updateNowPlaying(row) {
  const lineNum = row.cells[0].textContent;
  const startTime = row.cells[1].textContent;
  const endTime = row.cells[2].textContent;
  if (nowLineNum) nowLineNum.textContent = 'Line #' + lineNum;
  if (nowTime) nowTime.textContent = startTime + ' \\u2192 ' + endTime;
  if (nowJp) nowJp.textContent = row.getAttribute('data-jp');
  if (nowEn) nowEn.textContent = row.getAttribute('data-en');
}

function onRowClick(row) {
  if (selectedRow) selectedRow.classList.remove('selected');
  row.classList.add('selected');
  selectedRow = row;
  updateNowPlaying(row);
  if (video) {
    const t = parseFloat(row.getAttribute('data-start'));
    video.currentTime = t;
    video.play();
  }
}

// Auto-highlight current subtitle as video plays
if (video) {
  video.addEventListener('timeupdate', function() {
    const t = video.currentTime;
    let found = null;
    for (const row of rows) {
      if (row.classList.contains('hidden-row')) continue;
      const s = parseFloat(row.dataset.start);
      const e = parseFloat(row.dataset.end);
      if (t >= s && t < e) { found = row; break; }
    }
    if (found && found !== activeRow) {
      if (activeRow) activeRow.classList.remove('active-line');
      found.classList.add('active-line');
      activeRow = found;
      updateNowPlaying(found);
    }
  });
}

// Filter logic
const activeFilters = new Set();
const issueAttrs = ['short', 'long', 'zero_duration', 'large_gap'];

function hasAnyIssue(row) {
  for (const attr of issueAttrs) {
    if (row.getAttribute('data-issue-' + attr) === '1') return true;
  }
  return false;
}

function applyFilters() {
  rows.forEach(row => {
    if (activeFilters.size === 0) {
      row.classList.remove('hidden-row');
      return;
    }
    let show = false;
    if (activeFilters.has('__issues__')) {
      show = hasAnyIssue(row);
    } else {
      for (const f of activeFilters) {
        if (row.getAttribute('data-issue-' + f) === '1') show = true;
      }
    }
    row.classList.toggle('hidden-row', !show);
  });
  document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {
    btn.classList.toggle('inactive', activeFilters.size > 0 && !activeFilters.has(btn.dataset.filter));
  });
  document.querySelector('.btn-issues').classList.toggle('inactive',
    activeFilters.size > 0 && !activeFilters.has('__issues__'));
}

function toggleFilter(btn) {
  const f = btn.dataset.filter;
  activeFilters.delete('__issues__');
  if (activeFilters.has(f)) activeFilters.delete(f); else activeFilters.add(f);
  applyFilters();
}

function filterAll() {
  activeFilters.clear();
  applyFilters();
}

function filterIssuesOnly() {
  activeFilters.clear();
  activeFilters.add('__issues__');
  applyFilters();
}

// Pin table headers below sticky top bar
(function() {
  const stickyTop = document.getElementById('sticky-top');
  if (!stickyTop) return;
  function updateThTop() {
    const h = stickyTop.offsetHeight;
    document.querySelectorAll('th').forEach(th => th.style.top = h + 'px');
  }
  updateThTop();
  window.addEventListener('resize', updateThTop);
  new ResizeObserver(updateThTop).observe(stickyTop);
})();
</script>
</body>
</html>
"""
