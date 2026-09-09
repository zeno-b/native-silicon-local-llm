"""CSS for the embedded UI (theme variables, layout, trace, panels).

Part of the embedded UI, split out of the single HTML_PAGE string for
readability. Assembled once at import time in ui.py, so there is no
runtime cost and no extra file I/O.
"""

from __future__ import annotations

UI_CSS = r"""
   /* Texcel Solutions palette. Brand accent is the warm hexagon gradient
      (#fee10a -> #f7b015 -> #f3921a -> #e85a18 -> #d83027); --accent is its
      readable mid-orange. The light theme mirrors texcel.be (white, charcoal,
      orange); the dark theme is a matching dark variant with the same accent. */
   :root {
     --bg: #141312;
     --fg: #ededed;
     --accent: #f3921a;
     --surface: #1c1b1a;
     --border: #302e2c;
     --error: #ff6b6b;
     --warn: #f7b015;
     --success: #9be29b;
     --tool: #f0a35a;
     --code-bg: #161514;
     --panel-bg: #171615;
     --btn-bg: #2a2825;
     --btn-border: #4a453f;
     --btn-hover: #363330;
     --logbox-bg: #0c0b0a;
     --logbox-fg: #cbc9c6;
     --muted: #9b9a98;
     --surface-2: #1b1a19;
     --chip-bg: #2a2825;
     --code-inline-bg: #232221;
     --overlay-bg: #1b1a19;
     --meter-bg: #2a2825;
     --tip-bg: #000000;
     --tip-fg: #ededed;
     --tip-border: #444444;
     --accent-fg: #1a1a1a;
     --hl-kw: #f0a35a;
     --hl-str: #9be29b;
     --hl-num: #f7b015;
     --hl-com: #7a7a7a;
   }
   /* Light theme (the primary Texcel look). Applied by adding class="light" to
      <html>, persisted in localStorage, so the whole palette flips from these
      variables alone. */
   html.light {
     --bg: #f5f5f6;
     --fg: #1a1a1a;
     --accent: #e8641c;
     --surface: #ffffff;
     --border: #e4e4e8;
     --error: #c0392b;
     --warn: #b26a08;
     --success: #1f7a3d;
     --tool: #c85a12;
     --code-bg: #f3f3f5;
     --panel-bg: #ffffff;
     --btn-bg: #ffffff;
     --btn-border: #d6d6db;
     --btn-hover: #f0f0f2;
     --logbox-bg: #f4f4f6;
     --logbox-fg: #2b2b30;
     --muted: #5f6167;
     --surface-2: #ffffff;
     --chip-bg: #f1f1f4;
     --code-inline-bg: #eeedf1;
     --overlay-bg: #ffffff;
     --meter-bg: #e5e5ea;
     --tip-bg: #1a1a1a;
     --tip-fg: #f5f5f5;
     --tip-border: #3a3a3a;
     --accent-fg: #1a1a1a;
     --hl-kw: #7719aa;
     --hl-str: #1f7a3d;
     --hl-num: #b26a08;
     --hl-com: #8a8a8f;
   }
   .hl-kw { color: var(--hl-kw); }
   .hl-str { color: var(--hl-str); }
   .hl-num { color: var(--hl-num); }
   .hl-com { color: var(--hl-com); font-style: italic; }
   * { box-sizing: border-box; }
   body {
     font-family: "Raleway", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
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
   /* Wraps rather than overflowing: the button row is wider than its share of
      the header at 1440px, and without this it pushed 3px of horizontal
      scroll onto the whole page (and far more on a narrower window). */
   header .actions { display: flex; flex-wrap: wrap; gap: 6px; min-width: 0; }
   /* One line, ellipsised. It used to wrap to three cramped lines that pushed
      the header to 91px and broke mid-phrase ("ctx\n4096"); the full detail now
      lives in its hover tooltip, so the header stays a single row. min-width:0
      is what actually lets a flex item shrink below its content width. */
   #status {
     font-size: 12px;
     color: var(--success);
     white-space: nowrap;
     overflow: hidden;
     text-overflow: ellipsis;
     font-family: ui-monospace, monospace;
     flex: 1 1 0;
     min-width: 0;
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
   .user { align-self: flex-end; background: var(--accent); color: var(--accent-fg); }
   .assistant { align-self: flex-start; background: var(--surface); border: 1px solid var(--border); }
   .assistant.pending { opacity: 0.75; }
   .tool-card {
     align-self: flex-start;
     max-width: 80%;
     background: var(--surface-2);
     border: 1px solid var(--tool);
     border-radius: 10px;
     padding: 8px 10px;
     font-family: ui-monospace, monospace;
     font-size: 12px;
   }
   .tool-card .name { color: var(--tool); font-weight: 600; }
   .tool-card .args { color: var(--muted); margin: 4px 0; white-space: pre-wrap; }
   .tool-card pre {
     margin: 6px 0 0;
     white-space: pre-wrap;
     max-height: 220px;
     overflow: auto;
     color: var(--fg);
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
     color: var(--muted); font-family: ui-monospace, monospace; font-size: 12px;
   }
   .gnode.tool .ghead { cursor: pointer; user-select: none; }
   .gnode .gtool { color: var(--tool); font-weight: 600; }
   .gnode .gpill {
     font-size: 11px; padding: 1px 7px; border-radius: 999px;
     background: color-mix(in srgb, var(--success) 16%, transparent); color: var(--success);
   }
   .gnode.failed .gpill { background: color-mix(in srgb, var(--error) 16%, transparent); color: var(--error); }
   .gnode .gcaret { margin-left: auto; transition: transform 0.15s; color: var(--muted); }
   .gnode.open .gcaret { transform: rotate(90deg); }
   .gnode .gargs { color: var(--muted); font-family: ui-monospace, monospace; font-size: 12px; margin-top: 3px; }
   .gnode .gbody {
     margin-top: 6px; white-space: pre-wrap; font-family: ui-monospace, monospace;
     font-size: 12px; color: var(--logbox-fg); max-height: 240px; overflow: auto;
     background: var(--logbox-bg); border-radius: 8px; padding: 8px 10px;
   }
   .gnode.tool:not(.open) .gbody { display: none; }
   .gnode.notice .gmsg { color: var(--warn); }
   .gnode.notice.info::before { border-color: var(--tool); background: var(--tool); }
   .gnode.notice.info .gmsg { color: var(--tool); }
   .gnode.answer { border-left: 0; }
   .gnode.answer .gtext {
     background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
     padding: 10px 12px; white-space: pre-wrap; line-height: 1.4; font-size: 14px; color: var(--fg);
   }
   .gnode.answer.pending .gtext { opacity: 0.75; }
   .gmeta { color: var(--muted); font-size: 11px; font-family: ui-monospace, monospace; margin: 2px 0 0 6px; }
   .gactivity {
     display: flex; align-items: center; gap: 8px; margin: 2px 0 0 6px;
     font-family: ui-monospace, monospace; font-size: 12px; color: var(--accent);
   }
   .gspin {
     width: 11px; height: 11px; border-radius: 50%;
     border: 2px solid var(--border); border-top-color: var(--accent);
     animation: gspin 0.7s linear infinite; flex: 0 0 auto;
   }
   @keyframes gspin { to { transform: rotate(360deg); } }
   .gactivity .gelapsed { color: var(--muted); }
   .gactivity.stalled { color: var(--warn); }
   .gactivity.stalled .gspin { border-top-color: var(--warn); }
   .gnode.tool.running .gpill { background: color-mix(in srgb, var(--accent) 16%, transparent); color: var(--accent); }
   .gnode.reason::before { border-color: var(--accent); }
   .gnode.reason .gbody { color: var(--muted); max-height: 160px; }
   .gnode.reason.done::before { border-color: var(--success); background: var(--success); }
   .gnode.thinking::before { border-color: var(--muted); }
   .gnode.thinking .gtool { color: var(--muted); }
   .gnode.thinking .gbody { color: var(--muted); font-style: italic; max-height: 200px; }
   .gnode.thinking:not(.open) .gbody { display: none; }
   .gnode.diff .gtool { color: var(--accent); }
   .gnode.diff:not(.open) .gbody { display: none; }
   .gdiff { max-height: 340px; overflow: auto; font-size: 11px; line-height: 1.4; }
   .gdetail {
     margin: 1px 0 1px 22px; font-family: ui-monospace, monospace;
     font-size: 11px; color: var(--muted); white-space: pre-wrap;
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
     border: 1px solid var(--btn-border);
     border-radius: 8px;
     padding: 4px 8px;
     cursor: pointer;
     font-size: 13px;
   }
   .feedback button:hover { background: var(--btn-hover); }
   .feedback button:disabled { opacity: 0.4; cursor: default; }
   footer {
     display: flex;
     gap: 8px;
     padding: 10px 16px;
     border-top: 1px solid var(--border);
     align-items: flex-end;
   }
   textarea, input[type=text], input[type=number], input[type=password],
   input[type=email], input[type=search], select {
     padding: 9px 11px;
     border-radius: 10px;
     border: 1px solid var(--border);
     background: var(--surface);
     color: var(--fg);
     outline: none;
     font-family: inherit;
     font-size: 14px;
   }
   /* Visible focus on every control (was outline:none with nothing to replace it). */
   textarea:focus, input[type=text]:focus, input[type=number]:focus,
   input[type=password]:focus, input[type=email]:focus, input[type=search]:focus,
   select:focus {
     border-color: var(--accent);
     box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 20%, transparent);
   }
   ::placeholder { color: var(--muted); opacity: 1; }
   input[type=checkbox] { accent-color: var(--accent); }
   #input { flex: 1; resize: vertical; min-height: 42px; max-height: 200px; }
   button {
     padding: 9px 13px;
     border-radius: 10px;
     border: 1px solid var(--btn-border);
     background: var(--btn-bg);
     color: var(--fg);
     cursor: pointer;
     font-size: 13px;
     font-family: inherit;
     font-weight: 500;
   }
   button:hover { background: var(--btn-hover); }
   button:disabled { opacity: 0.5; cursor: not-allowed; }
   button.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-fg); font-weight: 600; }
   button.primary:hover { filter: brightness(1.06); background: var(--accent); }
   /* Branding in the header. */
   .brand { display: flex; align-items: center; gap: 9px; }
   .brand-slot { display: inline-flex; align-items: center; }
   .brand-logo { height: 24px; width: auto; aspect-ratio: 160.03 / 183.2; display: block; }
   .brand-mark { font-size: 18px; line-height: 1; color: var(--accent); }
   .brand strong {
     font-weight: 800; letter-spacing: 0.09em; text-transform: uppercase; font-size: 15px;
   }
   .brand .build { color: var(--muted); font-size: 11px; letter-spacing: 0; text-transform: none; font-weight: 400; }
   /* Hover tooltips. Any element with data-tip shows a styled bubble on hover
      and on keyboard focus, so every control can explain itself. */
   [data-tip] { position: relative; }
   [data-tip]:hover::after, [data-tip]:focus-visible::after {
     content: attr(data-tip);
     position: absolute; left: 50%; bottom: calc(100% + 8px);
     transform: translateX(-50%);
     background: var(--tip-bg); color: var(--tip-fg); border: 1px solid var(--tip-border);
     padding: 6px 9px; border-radius: 7px; font-size: 12px; font-weight: 400;
     line-height: 1.35; white-space: normal; width: max-content; max-width: 240px;
     text-align: left; z-index: 50; pointer-events: none;
     box-shadow: 0 6px 20px rgba(0,0,0,0.45);
   }
   [data-tip]:hover::before, [data-tip]:focus-visible::before {
     content: ""; position: absolute; left: 50%; bottom: calc(100% + 3px);
     transform: translateX(-50%);
     border: 5px solid transparent; border-top-color: var(--tip-border); z-index: 50;
     pointer-events: none;
   }
   /* Tooltips that would clip at the top of the screen flip below the element. */
   [data-tip-below]:hover::after, [data-tip-below]:focus-visible::after {
     bottom: auto; top: calc(100% + 8px);
   }
   [data-tip-below]:hover::before, [data-tip-below]:focus-visible::before {
     bottom: auto; top: calc(100% + 3px); border-top-color: transparent; border-bottom-color: var(--tip-border);
   }
   header { overflow: visible; }
   .system-msg { align-self: center; color: var(--muted); font-size: 12px; margin: 4px 0; }
   .agent-toggle {
     display: flex;
     align-items: center;
     gap: 6px;
     font-size: 13px;
     color: var(--muted);
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
     background: var(--surface-2);
   }
   .prompt-item .pname { font-size: 13px; font-weight: 600; }
   .prompt-item .pbody {
     font-size: 12px; color: var(--muted); margin: 4px 0 8px;
     max-height: 48px; overflow: hidden;
   }
   /* Conversation history rows */
   .conv-item {
     border: 1px solid var(--border);
     border-radius: 8px;
     padding: 10px 12px;
     margin-bottom: 8px;
     background: var(--surface-2);
   }
   .conv-item .ctitle { font-size: 13px; font-weight: 600; }
   .conv-item .cmeta { font-size: 11px; color: var(--muted); margin-top: 2px; }
   .conv-item .csnip { font-size: 12px; color: var(--muted); margin-top: 6px; }
   /* Knowledge-base drop zone */
   #dropZone {
     border: 1px dashed var(--border);
     border-radius: 10px;
     padding: 18px;
     text-align: center;
     color: var(--muted);
     font-size: 13px;
     cursor: pointer;
     margin-top: 10px;
     transition: border-color .15s, background .15s;
   }
   #dropZone:hover { border-color: var(--accent); }
   #dropZone.dragging { border-color: var(--accent); background: rgba(243,146,26,.10); }
   /* Command palette */
   #palette {
     position: fixed; inset: 0; background: rgba(0,0,0,.55);
     display: flex; align-items: flex-start; justify-content: center;
     padding-top: 12vh; z-index: 60;
   }
   #palette.hidden { display: none; }
   #paletteBox {
     width: min(560px, 92vw);
     background: var(--overlay-bg); border: 1px solid var(--border);
     border-radius: 12px; overflow: hidden;
     box-shadow: 0 20px 60px rgba(0,0,0,.5);
   }
   #paletteInput {
     width: 100%; border: 0; outline: none; padding: 14px 16px;
     background: var(--overlay-bg); color: var(--fg); font-size: 15px;
     border-bottom: 1px solid var(--border);
   }
   #paletteList { max-height: 320px; overflow-y: auto; }
   .pal-item { padding: 10px 16px; font-size: 13px; cursor: pointer; }
   .pal-item.sel, .pal-item:hover { background: color-mix(in srgb, var(--accent) 14%, transparent); }
   .pal-item .palhint { color: var(--muted); font-size: 11px; margin-left: 8px; }
   /* Rendered markdown in answers */
   .md-p { margin: 0 0 10px; line-height: 1.55; white-space: pre-wrap; }
   .md-h { margin: 12px 0 6px; font-size: 14px; font-weight: 600; }
   .md-list { margin: 0 0 10px; padding-left: 22px; line-height: 1.55; }
   .md-list li { margin: 2px 0; }
   .md-quote {
     margin: 0 0 10px; padding: 6px 12px;
     border-left: 3px solid var(--border); color: var(--muted);
   }
   .inline-code {
     background: var(--code-inline-bg); border: 1px solid var(--border); border-radius: 4px;
     padding: 1px 5px; font-size: 12px; font-family: ui-monospace, Menlo, monospace;
   }
   .codeblock {
     border: 1px solid var(--border); border-radius: 8px;
     overflow: hidden; margin: 0 0 10px; background: var(--code-bg);
   }
   .codebar {
     display: flex; justify-content: space-between; align-items: center;
     padding: 5px 10px; background: var(--chip-bg);
     border-bottom: 1px solid var(--border);
     font-size: 11px; color: var(--muted);
   }
   .codebar button {
     font-size: 11px; padding: 2px 8px; border-radius: 6px;
     border: 1px solid var(--border); background: var(--chip-bg); color: var(--fg); cursor: pointer;
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
     padding: 12px 14px; font-size: 12px; color: var(--muted);
     border-bottom: 1px solid var(--border);
     overflow-wrap: anywhere;
   }
   #browserList { max-height: 46vh; overflow-y: auto; }
   .dir-item {
     padding: 9px 14px; font-size: 13px; cursor: pointer;
     display: flex; justify-content: space-between; gap: 10px;
   }
   .dir-item:hover { background: color-mix(in srgb, var(--accent) 14%, transparent); }
   .dir-item .repo { color: var(--success); font-size: 11px; }
   /* Per-message actions */
   .msg-actions { margin-top: 6px; display: flex; gap: 6px; }
   .msg-actions button {
     font-size: 11px; padding: 2px 8px; border-radius: 6px;
     border: 1px solid var(--border); background: var(--chip-bg); color: var(--muted); cursor: pointer;
   }
   .msg-actions button:hover { border-color: var(--accent); color: var(--fg); }
   #settings h3 { margin: 0 0 10px; font-size: 14px; }
   #settings label {
     display: block;
     font-size: 12px;
     color: var(--muted);
     margin: 10px 0 4px;
   }
   #settings input:not([type=checkbox]), #settings select,
   #settings textarea { width: 100%; }
   /* The agent-mode switch sits in a settings row; keep it compact. */
   #settings .agent-toggle { margin-top: 4px; font-size: 12px; }
   #settings textarea { min-height: 70px; resize: vertical; }
   .row { display: flex; gap: 8px; }
   .row > div { flex: 1; }
   .meter { height: 6px; background: var(--meter-bg); border-radius: 3px; overflow: hidden; margin-top: 6px; }
   .meter > div { height: 100%; background: var(--accent); width: 0%; }
   .hint { font-size: 11px; color: var(--muted); margin-top: 6px; line-height: 1.4; }
   .tools-list { font-size: 11px; color: var(--muted); font-family: ui-monospace, monospace; line-height: 1.6; }

   nav.views { display: flex; gap: 4px; }
   nav.views button { padding: 6px 12px; }
   nav.views button.active { background: var(--accent); border-color: var(--accent); color: var(--accent-fg); }
   .view { flex: 1; display: flex; overflow: hidden; }
   .view.hidden { display: none; }

   /* Admin view uses sub-tabs so one section shows at a time instead of a wall
      of panels crammed side by side ("use more levels"). */
   #modelsView { flex-direction: column; }
   .subnav {
     display: flex; gap: 6px; padding: 10px 16px; flex-wrap: wrap;
     border-bottom: 1px solid var(--border); flex: 0 0 auto;
   }
   .subnav button { padding: 6px 12px; }
   .subnav button.active { background: var(--accent); border-color: var(--accent); color: var(--accent-fg); }
   #adminPanels { flex: 1; display: flex; overflow: auto; }
   #adminPanels .panel[hidden] { display: none !important; }

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
   .card .goal { font-size: 12px; color: var(--muted); line-height: 1.4; max-height: 48px; overflow: hidden; }
   .card .meta { font-size: 11px; color: var(--muted); margin-top: 6px; font-family: ui-monospace, monospace; }
   .card .card-actions { display: flex; gap: 4px; margin-top: 8px; flex-wrap: wrap; }
   .card .card-actions button { padding: 3px 8px; font-size: 12px; }
   .pill {
     font-size: 10px;
     padding: 1px 7px;
     border-radius: 999px;
     border: 1px solid var(--btn-border);
     color: var(--muted);
     white-space: nowrap;
     font-family: ui-monospace, monospace;
   }
   .pill.ok { border-color: var(--success); color: var(--success); }
   .pill.error, .pill.interrupted { border-color: var(--error); color: var(--error); }
   .pill.running { border-color: var(--accent); color: var(--accent); }
   .pill.cancelled { border-color: var(--warn); color: var(--warn); }
   .form-grid label { display: block; font-size: 12px; color: var(--muted); margin: 10px 0 4px; }
   .form-grid input, .form-grid textarea, .form-grid select { width: 100%; }
   .form-grid textarea { min-height: 64px; resize: vertical; }
   #runFeed { display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }
   .feed-line { font-size: 12px; color: var(--muted); font-family: ui-monospace, monospace; }
   .feed-answer {
     border: 1px solid var(--success);
     border-radius: 10px;
     padding: 10px 12px;
     white-space: pre-wrap;
     line-height: 1.4;
   }
   .feed-partial { color: var(--muted); white-space: pre-wrap; font-size: 13px; line-height: 1.4; }
   .logbox {
     background: var(--logbox-bg);
     border: 1px solid var(--border);
     border-radius: 8px;
     padding: 10px;
     font-family: ui-monospace, monospace;
     font-size: 11px;
     white-space: pre-wrap;
     max-height: 260px;
     overflow: auto;
     color: var(--logbox-fg);
   }
   table.models { width: 100%; border-collapse: collapse; font-size: 13px; }
   table.models td { padding: 7px 6px; border-bottom: 1px solid var(--border); }
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
   /* Boot state: the served HTML starts `booting` so nothing paints until
      authBoot() resolves and reveals exactly one of the app or the login box.
      Without this the whole app flashes on screen during the /api/auth/me fetch
      and is then covered by the login overlay (a jarring flicker on every load). */
   body.booting > header,
   body.booting > #main,
   body.booting > #composer,
   body.booting #loginOverlay { display: none !important; }
   #loginBox {
     width: 320px; max-width: 90vw; background: var(--surface);
     border: 1px solid var(--border); border-radius: 12px; padding: 22px;
     display: flex; flex-direction: column;
   }
   #loginBox label { font-size: 12px; color: var(--muted); margin: 10px 0 4px; display: block; }
   #loginBox input {
     width: 100%; padding: 11px 12px; border-radius: 10px; box-sizing: border-box;
     border: 1px solid var(--border); background: var(--panel-bg); color: var(--fg);
     font-size: 14px; font-family: inherit; transition: border-color .12s, box-shadow .12s;
   }
   #loginBox input:focus {
     outline: none; border-color: var(--accent);
     box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 22%, transparent);
   }
   /* Password field with a show/hide reveal button. */
   .pw-field { position: relative; }
   .pw-field input { padding-right: 44px; }
   .pw-reveal {
     position: absolute; right: 6px; top: 50%; transform: translateY(-50%);
     border: none; background: transparent; color: var(--muted); cursor: pointer;
     font-size: 12px; padding: 4px 6px; border-radius: 6px;
   }
   .pw-reveal:hover { color: var(--fg); background: var(--btn-hover); }
   .login-error { color: var(--error); font-size: 12px; min-height: 16px; margin-top: 8px; }
   .user-chip { font-size: 12px; color: var(--fg); }
   /* Role gating is UX only; the server enforces RBAC on every route. */
   body:not([data-role="admin"]) .admin-only { display: none !important; }
   .import-row { border: 1px solid var(--border); border-radius: 8px; padding: 10px; margin-top: 8px; }
   .import-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
   .import-detail { font-size: 12px; color: var(--muted); margin-top: 4px; }
   .import-actions { margin-top: 8px; display: flex; gap: 6px; }
   .import-badge { font-size: 11px; padding: 2px 7px; border-radius: 10px; background: var(--panel-bg); border: 1px solid var(--border); }
   .import-completed { color: var(--success); }
   .import-failed { color: var(--error); }

   /* --- Settings: collapsible groups so everything nests under Settings
      instead of spreading across the header. --- */
   details.sgroup { border-top: 1px solid var(--border); }
   details.sgroup:first-of-type { border-top: 0; }
   details.sgroup > summary {
     cursor: pointer; font-size: 13px; font-weight: 700; padding: 10px 2px;
     list-style: none; display: flex; align-items: center; gap: 6px;
     color: var(--fg);
   }
   details.sgroup > summary::-webkit-details-marker { display: none; }
   details.sgroup > summary::before {
     content: "\25B8"; color: var(--muted); font-size: 10px;
     transition: transform .12s; display: inline-block;
   }
   details.sgroup[open] > summary::before { transform: rotate(90deg); }
   details.sgroup > summary:hover { color: var(--accent); }
   .sbody { padding: 0 2px 12px; }
   .sbody > label:first-child { margin-top: 0; }

   /* --- Reusable toggle switch (agent mode, agent enable) --- */
   .switch { position: relative; display: inline-block; width: 34px; height: 20px; flex: 0 0 auto; }
   .switch input { opacity: 0; width: 0; height: 0; position: absolute; margin: 0; }
   .switch .slider {
     position: absolute; inset: 0; cursor: pointer; border-radius: 999px;
     background: var(--btn-border); transition: background .15s;
   }
   .switch .slider::before {
     content: ""; position: absolute; height: 14px; width: 14px; left: 3px; top: 3px;
     background: #fff; border-radius: 50%; transition: transform .15s;
     box-shadow: 0 1px 2px rgba(0,0,0,.3);
   }
   .switch input:checked + .slider { background: var(--accent); }
   .switch input:checked + .slider::before { transform: translateX(14px); }
   .agent-toggle { display: inline-flex; align-items: center; gap: 7px; font-size: 12px;
     color: var(--fg); cursor: pointer; user-select: none; }

   /* --- Account menu --- */
   .acct { position: relative; }
   .acct-btn { display: inline-flex; align-items: center; gap: 6px; }
   .acct-btn .caret { color: var(--muted); font-size: 10px; }
   .acct-pop {
     position: absolute; right: 0; top: calc(100% + 6px); z-index: 60; min-width: 210px;
     background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
     padding: 10px; box-shadow: 0 8px 24px rgba(0,0,0,.28);
   }
   .acct-info { font-size: 12px; color: var(--muted); margin-bottom: 8px; line-height: 1.5; }
   .acct-info strong { color: var(--fg); }
   .acct-pop button { width: 100%; }

   /* --- Agents screen --- */
   .cap-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
   @media (max-width: 760px) {
     .cap-grid { grid-template-columns: 1fr; }
     /* On a phone the status strip truncates to "Model: st…", which tells
        nobody anything while costing a whole header row. The same numbers
        are in Settings and in the strip's tooltip on wider screens. */
     #status { display: none; }
   }
   .cap-item {
     display: flex; gap: 8px; align-items: flex-start; padding: 8px 10px;
     border: 1px solid var(--border); border-radius: 9px; background: var(--panel-bg);
     cursor: pointer;
   }
   .cap-item:hover { border-color: var(--accent); }
   .cap-item input { margin-top: 2px; flex: 0 0 auto; accent-color: var(--accent); }
   .cap-text { display: flex; flex-direction: column; }
   .cap-label { font-size: 13px; font-weight: 600; }
   .cap-desc { font-size: 11px; color: var(--muted); margin-top: 2px; }
   .agent-card { border: 1px solid var(--border); border-radius: 10px; padding: 12px; margin-bottom: 10px; background: var(--panel-bg); }
   .agent-head { display: flex; align-items: center; gap: 10px; }
   .agent-name { flex: 1 1 auto; font-weight: 600; font-size: 14px; padding: 7px 9px;
     border: 1px solid var(--border); border-radius: 8px; background: var(--surface); color: var(--fg); }
   .agent-desc { width: 100%; margin: 8px 0; padding: 7px 9px; border: 1px solid var(--border);
     border-radius: 8px; background: var(--surface); color: var(--fg); font-size: 13px; box-sizing: border-box; }
   .agent-merged { border: 1px solid var(--accent); border-radius: 10px; padding: 10px; margin-bottom: 10px; }
   .agent-answer { font-size: 13px; margin-top: 6px; }
   .agent-result { border: 1px solid var(--border); border-radius: 9px; padding: 8px 10px; margin-top: 8px; }
   .agent-result summary { cursor: pointer; font-size: 12px; color: var(--muted); }

   /* The catch-all for the `hidden` state class, LAST so it wins over any
      earlier same-specificity `display`. Without it a `.hidden` element with no
      id-qualified rule of its own stays on screen: that is exactly how the
      account popup ended up permanently open over the header. The id-qualified
      rules above are still needed, because an id selector outranks this one. */
   .hidden { display: none; }
 """

__all__ = ["UI_CSS"]
