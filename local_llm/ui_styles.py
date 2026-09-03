"""CSS for the embedded UI (theme variables, layout, trace, panels).

Part of the embedded UI, split out of the single HTML_PAGE string for
readability. Assembled once at import time in ui.py, so there is no
runtime cost and no extra file I/O.
"""

from __future__ import annotations

UI_CSS = r"""
   :root {
     --bg: #111;
     --fg: #eee;
     --accent: #0a84ff;
     --surface: #1a1a1a;
     --border: #333;
     --error: #ff6b6b;
     --warn: #ffd93d;
     --success: #9be29b;
     --tool: #b48ead;
     --code-bg: #151515;
     --panel-bg: #131313;
     --hl-kw: #c792ea;
     --hl-str: #9be29b;
     --hl-num: #ffd93d;
     --hl-com: #7a7a7a;
   }
   /* Light theme. Applied by adding class="light" to <html>, persisted in
      localStorage, so the whole palette flips from these variables alone. */
   html.light {
     --bg: #f7f7f8;
     --fg: #1c1c1e;
     --accent: #0a63d6;
     --surface: #ffffff;
     --border: #d6d6db;
     --error: #c0392b;
     --warn: #a4750a;
     --success: #1f7a3d;
     --tool: #7a4fa3;
     --code-bg: #f2f2f5;
     --panel-bg: #ffffff;
     --hl-kw: #7719aa;
     --hl-str: #1f7a3d;
     --hl-num: #a4750a;
     --hl-com: #8a8a8f;
   }
   .hl-kw { color: var(--hl-kw); }
   .hl-str { color: var(--hl-str); }
   .hl-num { color: var(--hl-num); }
   .hl-com { color: var(--hl-com); font-style: italic; }
   * { box-sizing: border-box; }
   body {
     font-family: -apple-system, BlinkMacSystemFont, sans-serif;
     margin: 0;
     padding: 0;
     background: var(--bg);
     color: var(--fg);
     height: 100vh;
     display: flex;
     flex-direction: column;
   }
   header {
     padding: 10px 16px;
     border-bottom: 1px solid var(--border);
     display: flex;
     justify-content: space-between;
     gap: 12px;
     align-items: center;
     flex-wrap: wrap;
   }
   header .actions { display: flex; gap: 6px; }
   #status {
     font-size: 12px;
     color: var(--success);
     white-space: pre-wrap;
     font-family: ui-monospace, monospace;
     flex: 1;
     min-width: 220px;
   }
   #status.error { color: var(--error); }
   #status.warn { color: var(--warn); }
   #main { flex: 1; display: flex; overflow: hidden; }
   #chat {
     flex: 1;
     overflow-y: auto;
     padding: 16px;
     display: flex;
     flex-direction: column;
     gap: 8px;
   }
   .msg {
     max-width: 80%;
     padding: 10px 12px;
     border-radius: 12px;
     white-space: pre-wrap;
     line-height: 1.35;
     word-wrap: break-word;
   }
   .user { align-self: flex-end; background: var(--accent); color: white; }
   .assistant { align-self: flex-start; background: var(--surface); border: 1px solid #444; }
   .assistant.pending { opacity: 0.75; }
   .tool-card {
     align-self: flex-start;
     max-width: 80%;
     background: #16121a;
     border: 1px solid var(--tool);
     border-radius: 10px;
     padding: 8px 10px;
     font-family: ui-monospace, monospace;
     font-size: 12px;
   }
   .tool-card .name { color: var(--tool); font-weight: 600; }
   .tool-card .args { color: #999; margin: 4px 0; white-space: pre-wrap; }
   .tool-card pre {
     margin: 6px 0 0;
     white-space: pre-wrap;
     max-height: 220px;
     overflow: auto;
     color: #ccc;
   }
   .tool-card.failed { border-color: var(--error); }
   /* Glass box: the agent's work as a live vertical trace. */
   .turn { align-self: stretch; display: flex; flex-direction: column; gap: 8px; }
   .trace {
     align-self: flex-start;
     max-width: 88%;
     margin-left: 6px;
     padding-left: 14px;
     border-left: 2px solid var(--border);
     display: flex;
     flex-direction: column;
     gap: 10px;
   }
   .trace:empty { display: none; }
   .gnode { position: relative; font-size: 13px; }
   .gnode::before {
     content: ""; position: absolute; left: -19px; top: 5px;
     width: 8px; height: 8px; border-radius: 50%;
     background: var(--surface); border: 1.5px solid var(--border);
   }
   .gnode.router::before { border-color: var(--accent); }
   .gnode.ok::before { border-color: var(--success); background: var(--success); }
   .gnode.failed::before { border-color: var(--error); background: var(--error); }
   .gnode.notice::before { border-color: var(--warn); background: var(--warn); }
   .gnode .ghead {
     display: flex; align-items: center; gap: 8px;
     color: #9aa0a6; font-family: ui-monospace, monospace; font-size: 12px;
   }
   .gnode.tool .ghead { cursor: pointer; user-select: none; }
   .gnode .gtool { color: var(--tool); font-weight: 600; }
   .gnode .gpill {
     font-size: 11px; padding: 1px 7px; border-radius: 999px;
     background: #22331f; color: var(--success);
   }
   .gnode.failed .gpill { background: #331f1f; color: var(--error); }
   .gnode .gcaret { margin-left: auto; transition: transform 0.15s; color: #666; }
   .gnode.open .gcaret { transform: rotate(90deg); }
   .gnode .gargs { color: #8a8f94; font-family: ui-monospace, monospace; font-size: 12px; margin-top: 3px; }
   .gnode .gbody {
     margin-top: 6px; white-space: pre-wrap; font-family: ui-monospace, monospace;
     font-size: 12px; color: #cfcfcf; max-height: 240px; overflow: auto;
     background: #16121a; border-radius: 8px; padding: 8px 10px;
   }
   .gnode.tool:not(.open) .gbody { display: none; }
   .gnode.notice .gmsg { color: var(--warn); }
   .gnode.notice.info::before { border-color: var(--tool); background: var(--tool); }
   .gnode.notice.info .gmsg { color: var(--tool); }
   .gnode.answer { border-left: 0; }
   .gnode.answer .gtext {
     background: var(--surface); border: 1px solid #444; border-radius: 12px;
     padding: 10px 12px; white-space: pre-wrap; line-height: 1.4; font-size: 14px; color: var(--fg);
   }
   .gnode.answer.pending .gtext { opacity: 0.75; }
   .gmeta { color: #777; font-size: 11px; font-family: ui-monospace, monospace; margin: 2px 0 0 6px; }
   .gactivity {
     display: flex; align-items: center; gap: 8px; margin: 2px 0 0 6px;
     font-family: ui-monospace, monospace; font-size: 12px; color: var(--accent);
   }
   .gspin {
     width: 11px; height: 11px; border-radius: 50%;
     border: 2px solid #2a3b4d; border-top-color: var(--accent);
     animation: gspin 0.7s linear infinite; flex: 0 0 auto;
   }
   @keyframes gspin { to { transform: rotate(360deg); } }
   .gactivity .gelapsed { color: #888; }
   .gactivity.stalled { color: var(--warn); }
   .gactivity.stalled .gspin { border-top-color: var(--warn); }
   .gnode.tool.running .gpill { background: #1f2a33; color: var(--accent); }
   .gnode.reason::before { border-color: var(--accent); }
   .gnode.reason .gbody { color: #b9c2cc; max-height: 160px; }
   .gnode.reason.done::before { border-color: var(--success); background: var(--success); }
   .gnode.thinking::before { border-color: #8a8f94; }
   .gnode.thinking .gtool { color: #9aa0a6; }
   .gnode.thinking .gbody { color: #9aa0a6; font-style: italic; max-height: 200px; }
   .gnode.thinking:not(.open) .gbody { display: none; }
   .gnode.diff .gtool { color: var(--accent); }
   .gnode.diff:not(.open) .gbody { display: none; }
   .gdiff { max-height: 340px; overflow: auto; font-size: 11px; line-height: 1.4; }
   .gdetail {
     margin: 1px 0 1px 22px; font-family: ui-monospace, monospace;
     font-size: 11px; color: #6b7075; white-space: pre-wrap;
   }
   .gdetail::before { content: "\2699 "; opacity: 0.6; }
   .feedback {
     align-self: flex-start;
     display: flex;
     gap: 6px;
     margin-left: 6px;
     margin-bottom: 10px;
   }
   .feedback button {
     background: var(--surface);
     color: var(--fg);
     border: 1px solid #555;
     border-radius: 8px;
     padding: 4px 8px;
     cursor: pointer;
     font-size: 13px;
   }
   .feedback button:hover { background: #333; }
   .feedback button:disabled { opacity: 0.4; cursor: default; }
   footer {
     display: flex;
     gap: 8px;
     padding: 10px 16px;
     border-top: 1px solid var(--border);
     align-items: flex-end;
   }
   textarea, input[type=text], input[type=number], select {
     padding: 9px 11px;
     border-radius: 10px;
     border: 1px solid #444;
     background: var(--surface);
     color: var(--fg);
     outline: none;
     font-family: inherit;
     font-size: 14px;
   }
   #input { flex: 1; resize: vertical; min-height: 42px; max-height: 200px; }
   button {
     padding: 9px 13px;
     border-radius: 10px;
     border: 1px solid #555;
     background: #2c2c2c;
     color: var(--fg);
     cursor: pointer;
     font-size: 13px;
   }
   button:hover { background: #3a3a3a; }
   button:disabled { opacity: 0.5; cursor: not-allowed; }
   button.primary { background: var(--accent); border-color: var(--accent); color: white; }
   /* Branding in the header. */
   .brand { display: flex; align-items: center; gap: 8px; }
   .brand-slot { display: inline-flex; align-items: center; }
   .brand-logo { height: 22px; width: auto; border-radius: 5px; display: block; }
   .brand-mark { font-size: 18px; line-height: 1; color: var(--accent); }
   .brand .build { color: #666; font-size: 11px; }
   /* Hover tooltips. Any element with data-tip shows a styled bubble on hover
      and on keyboard focus, so every control can explain itself. */
   [data-tip] { position: relative; }
   [data-tip]:hover::after, [data-tip]:focus-visible::after {
     content: attr(data-tip);
     position: absolute; left: 50%; bottom: calc(100% + 8px);
     transform: translateX(-50%);
     background: #000; color: #eee; border: 1px solid #444;
     padding: 6px 9px; border-radius: 7px; font-size: 12px; font-weight: 400;
     line-height: 1.35; white-space: normal; width: max-content; max-width: 240px;
     text-align: left; z-index: 50; pointer-events: none;
     box-shadow: 0 6px 20px rgba(0,0,0,0.45);
   }
   [data-tip]:hover::before, [data-tip]:focus-visible::before {
     content: ""; position: absolute; left: 50%; bottom: calc(100% + 3px);
     transform: translateX(-50%);
     border: 5px solid transparent; border-top-color: #444; z-index: 50;
     pointer-events: none;
   }
   /* Tooltips that would clip at the top of the screen flip below the element. */
   [data-tip-below]:hover::after, [data-tip-below]:focus-visible::after {
     bottom: auto; top: calc(100% + 8px);
   }
   [data-tip-below]:hover::before, [data-tip-below]:focus-visible::before {
     bottom: auto; top: calc(100% + 3px); border-top-color: transparent; border-bottom-color: #444;
   }
   header { overflow: visible; }
   .system-msg { align-self: center; color: #888; font-size: 12px; margin: 4px 0; }
   .agent-toggle {
     display: flex;
     align-items: center;
     gap: 6px;
     font-size: 13px;
     color: #bbb;
     white-space: nowrap;
   }
   #settings {
     width: 320px;
     border-left: 1px solid var(--border);
     padding: 14px 16px;
     overflow-y: auto;
     display: none;
     background: var(--panel-bg);
   }
   #settings.open { display: block; }
   /* Prompt library panel: same shell as settings so it feels native. */
   #promptsPanel {
     width: 320px;
     border-left: 1px solid var(--border);
     padding: 14px 16px;
     overflow-y: auto;
     background: var(--panel-bg);
   }
   #promptsPanel.hidden { display: none; }
   #promptsPanel h3 { margin: 0 0 10px; font-size: 14px; }
   .prompt-item {
     border: 1px solid var(--border);
     border-radius: 8px;
     padding: 8px 10px;
     margin-bottom: 8px;
     background: #181818;
   }
   .prompt-item .pname { font-size: 13px; font-weight: 600; }
   .prompt-item .pbody {
     font-size: 12px; color: #999; margin: 4px 0 8px;
     max-height: 48px; overflow: hidden;
   }
   /* Conversation history rows */
   .conv-item {
     border: 1px solid var(--border);
     border-radius: 8px;
     padding: 10px 12px;
     margin-bottom: 8px;
     background: #181818;
   }
   .conv-item .ctitle { font-size: 13px; font-weight: 600; }
   .conv-item .cmeta { font-size: 11px; color: #888; margin-top: 2px; }
   .conv-item .csnip { font-size: 12px; color: #aaa; margin-top: 6px; }
   /* Knowledge-base drop zone */
   #dropZone {
     border: 1px dashed var(--border);
     border-radius: 10px;
     padding: 18px;
     text-align: center;
     color: #999;
     font-size: 13px;
     cursor: pointer;
     margin-top: 10px;
     transition: border-color .15s, background .15s;
   }
   #dropZone:hover { border-color: var(--accent); }
   #dropZone.dragging { border-color: var(--accent); background: rgba(10,132,255,.08); }
   /* Command palette */
   #palette {
     position: fixed; inset: 0; background: rgba(0,0,0,.55);
     display: flex; align-items: flex-start; justify-content: center;
     padding-top: 12vh; z-index: 60;
   }
   #palette.hidden { display: none; }
   #paletteBox {
     width: min(560px, 92vw);
     background: #1b1b1b; border: 1px solid var(--border);
     border-radius: 12px; overflow: hidden;
     box-shadow: 0 20px 60px rgba(0,0,0,.5);
   }
   #paletteInput {
     width: 100%; border: 0; outline: none; padding: 14px 16px;
     background: #1b1b1b; color: var(--fg); font-size: 15px;
     border-bottom: 1px solid var(--border);
   }
   #paletteList { max-height: 320px; overflow-y: auto; }
   .pal-item { padding: 10px 16px; font-size: 13px; cursor: pointer; }
   .pal-item.sel, .pal-item:hover { background: rgba(10,132,255,.15); }
   .pal-item .palhint { color: #888; font-size: 11px; margin-left: 8px; }
   /* Rendered markdown in answers */
   .md-p { margin: 0 0 10px; line-height: 1.55; white-space: pre-wrap; }
   .md-h { margin: 12px 0 6px; font-size: 14px; font-weight: 600; }
   .md-list { margin: 0 0 10px; padding-left: 22px; line-height: 1.55; }
   .md-list li { margin: 2px 0; }
   .md-quote {
     margin: 0 0 10px; padding: 6px 12px;
     border-left: 3px solid var(--border); color: #bbb;
   }
   .inline-code {
     background: #232323; border: 1px solid var(--border); border-radius: 4px;
     padding: 1px 5px; font-size: 12px; font-family: ui-monospace, Menlo, monospace;
   }
   .codeblock {
     border: 1px solid var(--border); border-radius: 8px;
     overflow: hidden; margin: 0 0 10px; background: var(--code-bg);
   }
   .codebar {
     display: flex; justify-content: space-between; align-items: center;
     padding: 5px 10px; background: #1d1d1d;
     border-bottom: 1px solid var(--border);
     font-size: 11px; color: #999;
   }
   .codebar button {
     font-size: 11px; padding: 2px 8px; border-radius: 6px;
     border: 1px solid var(--border); background: #262626; color: #ccc; cursor: pointer;
   }
   .codebar button:hover { border-color: var(--accent); color: var(--fg); }
   .codeblock pre {
     margin: 0; padding: 10px 12px; overflow-x: auto;
     font-size: 12px; line-height: 1.45;
     font-family: ui-monospace, Menlo, monospace;
   }
   /* Folder picker */
   #browser {
     position: fixed; inset: 0; background: rgba(0,0,0,.55);
     display: flex; align-items: flex-start; justify-content: center;
     padding-top: 10vh; z-index: 70;
   }
   #browser.hidden { display: none; }
   #browserBox {
     width: min(620px, 94vw); background: var(--panel-bg);
     border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
     box-shadow: 0 20px 60px rgba(0,0,0,.5);
   }
   #browserPath {
     padding: 12px 14px; font-size: 12px; color: #999;
     border-bottom: 1px solid var(--border);
     overflow-wrap: anywhere;
   }
   #browserList { max-height: 46vh; overflow-y: auto; }
   .dir-item {
     padding: 9px 14px; font-size: 13px; cursor: pointer;
     display: flex; justify-content: space-between; gap: 10px;
   }
   .dir-item:hover { background: rgba(10,132,255,.15); }
   .dir-item .repo { color: var(--success); font-size: 11px; }
   /* Per-message actions */
   .msg-actions { margin-top: 6px; display: flex; gap: 6px; }
   .msg-actions button {
     font-size: 11px; padding: 2px 8px; border-radius: 6px;
     border: 1px solid var(--border); background: #1a1a1a; color: #bbb; cursor: pointer;
   }
   .msg-actions button:hover { border-color: var(--accent); color: var(--fg); }
   #settings h3 { margin: 0 0 10px; font-size: 14px; }
   #settings label {
     display: block;
     font-size: 12px;
     color: #999;
     margin: 10px 0 4px;
   }
   #settings input, #settings select, #settings textarea { width: 100%; }
   #settings textarea { min-height: 70px; resize: vertical; }
   .row { display: flex; gap: 8px; }
   .row > div { flex: 1; }
   .meter { height: 6px; background: #222; border-radius: 3px; overflow: hidden; margin-top: 6px; }
   .meter > div { height: 100%; background: var(--accent); width: 0%; }
   .hint { font-size: 11px; color: #777; margin-top: 6px; line-height: 1.4; }
   .tools-list { font-size: 11px; color: #888; font-family: ui-monospace, monospace; line-height: 1.6; }

   nav.views { display: flex; gap: 4px; }
   nav.views button { padding: 6px 12px; }
   nav.views button.active { background: var(--accent); border-color: var(--accent); color: white; }
   .view { flex: 1; display: flex; overflow: hidden; }
   .view.hidden { display: none; }

   #tasksView { flex-direction: row; }
   .task-list { width: 340px; min-width: 300px; border-right: 1px solid var(--border); overflow-y: auto; padding: 14px; }
   .task-monitor { flex: 1; overflow-y: auto; padding: 14px 16px; }
   .card {
     border: 1px solid var(--border);
     border-radius: 10px;
     padding: 10px 12px;
     margin-bottom: 10px;
     background: var(--surface);
     cursor: pointer;
   }
   .card.selected { border-color: var(--accent); }
   .card h4 { margin: 0 0 4px; font-size: 13px; display: flex; justify-content: space-between; gap: 8px; }
   .card .goal { font-size: 12px; color: #aaa; line-height: 1.4; max-height: 48px; overflow: hidden; }
   .card .meta { font-size: 11px; color: #777; margin-top: 6px; font-family: ui-monospace, monospace; }
   .card .card-actions { display: flex; gap: 4px; margin-top: 8px; flex-wrap: wrap; }
   .card .card-actions button { padding: 3px 8px; font-size: 12px; }
   .pill {
     font-size: 10px;
     padding: 1px 7px;
     border-radius: 999px;
     border: 1px solid #555;
     color: #bbb;
     white-space: nowrap;
     font-family: ui-monospace, monospace;
   }
   .pill.ok { border-color: var(--success); color: var(--success); }
   .pill.error, .pill.interrupted { border-color: var(--error); color: var(--error); }
   .pill.running { border-color: var(--accent); color: var(--accent); }
   .pill.cancelled { border-color: var(--warn); color: var(--warn); }
   .form-grid label { display: block; font-size: 12px; color: #999; margin: 10px 0 4px; }
   .form-grid input, .form-grid textarea, .form-grid select { width: 100%; }
   .form-grid textarea { min-height: 64px; resize: vertical; }
   #runFeed { display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }
   .feed-line { font-size: 12px; color: #999; font-family: ui-monospace, monospace; }
   .feed-answer {
     border: 1px solid var(--success);
     border-radius: 10px;
     padding: 10px 12px;
     white-space: pre-wrap;
     line-height: 1.4;
   }
   .feed-partial { color: #ccc; white-space: pre-wrap; font-size: 13px; line-height: 1.4; }
   .logbox {
     background: #0c0c0c;
     border: 1px solid var(--border);
     border-radius: 8px;
     padding: 10px;
     font-family: ui-monospace, monospace;
     font-size: 11px;
     white-space: pre-wrap;
     max-height: 260px;
     overflow: auto;
     color: #bbb;
   }
   table.models { width: 100%; border-collapse: collapse; font-size: 13px; }
   table.models td { padding: 7px 6px; border-bottom: 1px solid #262626; }
   table.models tr:last-child td { border-bottom: none; }
   table.models td.id { font-family: ui-monospace, monospace; font-size: 12px; word-break: break-all; }
   .panel { max-width: 760px; padding: 16px; overflow-y: auto; flex: 1; }
   .panel h3 { margin: 18px 0 8px; font-size: 14px; }
   .panel h3:first-child { margin-top: 0; }

   /* --- Auth: login overlay, role gating, user chip, import panel --- */
   #loginOverlay {
     position: fixed; inset: 0; background: var(--bg);
     display: none; align-items: center; justify-content: center; z-index: 100;
   }
   body.locked #loginOverlay { display: flex; }
   /* When not signed in, hide the whole app so only the login box shows. */
   body.locked > header,
   body.locked > #main,
   body.locked > #composer { display: none; }
   #loginBox {
     width: 320px; max-width: 90vw; background: var(--surface);
     border: 1px solid var(--border); border-radius: 12px; padding: 22px;
     display: flex; flex-direction: column;
   }
   #loginBox input {
     width: 100%; padding: 9px 10px; border-radius: 8px; box-sizing: border-box;
     border: 1px solid var(--border); background: var(--panel-bg); color: var(--text);
   }
   .login-error { color: #ff6b6b; font-size: 12px; min-height: 16px; margin-top: 8px; }
   .user-chip { font-size: 12px; color: #999; align-self: center; padding: 0 6px; }
   /* Role gating is UX only; the server enforces RBAC on every route. */
   body:not([data-role="admin"]) .admin-only { display: none !important; }
   .import-row { border: 1px solid var(--border); border-radius: 8px; padding: 10px; margin-top: 8px; }
   .import-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
   .import-detail { font-size: 12px; color: #999; margin-top: 4px; }
   .import-actions { margin-top: 8px; display: flex; gap: 6px; }
   .import-badge { font-size: 11px; padding: 2px 7px; border-radius: 10px; background: var(--panel-bg); border: 1px solid var(--border); }
   .import-completed { color: #3fb950; }
   .import-failed { color: #ff6b6b; }
 """

__all__ = ["UI_CSS"]
