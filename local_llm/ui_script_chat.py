"""Front-end: shared state, helpers, markdown rendering and the streaming chat trace.

Part of the embedded UI, split out of the single HTML_PAGE string for
readability. Assembled once at import time in ui.py, so there is no
runtime cost and no extra file I/O.
"""

from __future__ import annotations

UI_JS_CHAT = r"""
   var chat = document.getElementById("chat");
   var input = document.getElementById("input");
   var sendBtn = document.getElementById("sendBtn");
   var stopBtn = document.getElementById("stopBtn");
   var statusEl = document.getElementById("status");
   var agentToggle = document.getElementById("agentToggle");
   var settingsEl = document.getElementById("settings");

   var conversationId = localStorage.getItem("llm_conversation") || randomId();
   localStorage.setItem("llm_conversation", conversationId);
   var contextSize = 4096;
   var usedTokens = 0;
   var busy = false;
   var controller = null;

   function randomId() {
     return Math.random().toString(36).slice(2, 12);
   }

   function estimateTokens(text) {
     return Math.max(1, Math.ceil((text || "").length / 4));
   }

   input.addEventListener("keydown", function(e) {
     if (e.key === "Enter" && !e.shiftKey) {
       e.preventDefault();
       send();
     }
   });

   input.addEventListener("input", function() {
     input.style.height = "auto";
     input.style.height = Math.min(200, input.scrollHeight) + "px";
   });

   function stopStream() {
     if (controller) controller.abort();
   }

   function scrollDown() {
     chat.scrollTop = chat.scrollHeight;
   }

   // ------------------------------------------------------- markdown ---
   // Builds DOM nodes rather than assigning innerHTML, so model output can never
   // inject markup or scripts into the page. Handles fenced code (with a copy
   // button and language label), headings, lists, blockquotes, inline code,
   // bold/italic and links.

   function renderInline(text, host) {
     // Order matters: code spans first so their contents are never re-parsed.
     var pattern = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(\*[^*\n]+\*|_[^_\n]+_)|(\[[^\]\n]+\]\([^)\s]+\))/;
     var rest = text;
     while (rest) {
       var m = rest.match(pattern);
       if (!m) { host.appendChild(document.createTextNode(rest)); break; }
       if (m.index > 0) host.appendChild(document.createTextNode(rest.slice(0, m.index)));
       var tok = m[0];
       if (tok[0] === "`") {
         var code = document.createElement("code");
         code.className = "inline-code";
         code.textContent = tok.slice(1, -1);
         host.appendChild(code);
       } else if (tok.slice(0, 2) === "**") {
         var b = document.createElement("strong");
         b.textContent = tok.slice(2, -2);
         host.appendChild(b);
       } else if (tok[0] === "[") {
         var close = tok.indexOf("](");
         var a = document.createElement("a");
         a.textContent = tok.slice(1, close);
         a.href = tok.slice(close + 2, -1);
         a.target = "_blank";
         a.rel = "noopener noreferrer";
         host.appendChild(a);
       } else {
         var i = document.createElement("em");
         i.textContent = tok.slice(1, -1);
         host.appendChild(i);
       }
       rest = rest.slice(m.index + tok.length);
     }
   }

   // Minimal offline syntax highlighting. A real grammar is overkill here and a
   // CDN library would break the offline-first promise, so this tokenises the
   // few things that carry most of the visual signal: comments, strings,
   // numbers and keywords. Everything is inserted as text nodes, never HTML.
   var HL_KEYWORDS = {
     python: "def class return if elif else for while import from as try except finally raise with lambda yield pass break continue in is not and or None True False async await global nonlocal assert del",
     javascript: "function return if else for while var let const class new try catch finally throw typeof instanceof in of do switch case break continue default null undefined true false async await import export from extends this",
     json: "true false null",
     bash: "if then else fi for while do done case esac function return export local echo cd set source",
     sql: "select from where group by order having insert update delete create table drop alter join left right inner outer on as values set distinct limit"
   };
   HL_KEYWORDS.js = HL_KEYWORDS.javascript;
   HL_KEYWORDS.ts = HL_KEYWORDS.javascript;
   HL_KEYWORDS.py = HL_KEYWORDS.python;
   HL_KEYWORDS.sh = HL_KEYWORDS.bash;

   function highlightInto(code, lang, host) {
     var words = HL_KEYWORDS[(lang || "").toLowerCase()];
     if (!words) { host.textContent = code; return; }
     var keywords = {};
     words.split(" ").forEach(function(w) { keywords[w] = true; });
     var lineComment = (lang === "python" || lang === "py" || lang === "bash" || lang === "sh") ? "#" : "//";
     var i = 0;
     function emit(text, cls) {
       if (!text) return;
       if (!cls) { host.appendChild(document.createTextNode(text)); return; }
       var span = document.createElement("span");
       span.className = cls;
       span.textContent = text;
       host.appendChild(span);
     }
     while (i < code.length) {
       var ch = code[i];
       // Comment to end of line
       if (code.startsWith(lineComment, i)) {
         var nl = code.indexOf("\n", i);
         if (nl < 0) nl = code.length;
         emit(code.slice(i, nl), "hl-com");
         i = nl;
         continue;
       }
       // String literal
       if (ch === '"' || ch === "'" || ch === "`") {
         var j = i + 1;
         while (j < code.length && code[j] !== ch) {
           if (code[j] === "\\") j++;
           j++;
         }
         emit(code.slice(i, Math.min(j + 1, code.length)), "hl-str");
         i = j + 1;
         continue;
       }
       // Number
       if (ch >= "0" && ch <= "9") {
         var k = i;
         while (k < code.length && /[0-9._xa-fA-F]/.test(code[k])) k++;
         emit(code.slice(i, k), "hl-num");
         i = k;
         continue;
       }
       // Word (keyword or plain identifier)
       if (/[A-Za-z_$]/.test(ch)) {
         var w = i;
         while (w < code.length && /[A-Za-z0-9_$]/.test(code[w])) w++;
         var word = code.slice(i, w);
         emit(word, keywords[word] ? "hl-kw" : null);
         i = w;
         continue;
       }
       emit(ch, null);
       i++;
     }
   }

   function makeCodeBlock(code, lang) {
     var wrap = document.createElement("div");
     wrap.className = "codeblock";
     var bar = document.createElement("div");
     bar.className = "codebar";
     var label = document.createElement("span");
     label.textContent = lang || "code";
     var copy = document.createElement("button");
     copy.textContent = "Copy";
     copy.title = "Copy this code";
     copy.onclick = function() {
       navigator.clipboard.writeText(code).then(function() {
         copy.textContent = "Copied";
         setTimeout(function() { copy.textContent = "Copy"; }, 1200);
       }, function() { alert("Could not copy."); });
     };
     bar.appendChild(label);
     bar.appendChild(copy);
     var pre = document.createElement("pre");
     var el = document.createElement("code");
     highlightInto(code, lang, el);
     pre.appendChild(el);
     wrap.appendChild(bar);
     wrap.appendChild(pre);
     return wrap;
   }

   // Re-render streaming text as markdown, but only every ~250ms: parsing on
   // every token would be wasteful and would make half-typed code fences flicker.
   function scheduleMarkdown(node) {
     if (node.mdTimer) return;
     node.mdTimer = setTimeout(function() {
       node.mdTimer = null;
       // A code fence that is still open must not be rendered half-formed. Show
       // the raw text so the code streams visibly, and mark the node un-rendered
       // so the token handler keeps the plain view live until the fence closes.
       var fences = (node.raw.match(/```/g) || []).length;
       if (fences % 2 === 1) {
         node.text.textContent = node.raw;
         node.rendered = false;
         return;
       }
       renderMarkdown(node.raw, node.text);
       // Rendered now: the token handler stops overwriting textContent with raw
       // text every token, so the message no longer oscillates plain<->markdown
       // (the flicker). Subsequent tokens accumulate and re-render on the next
       // debounce, swapped in atomically by renderMarkdown.
       node.rendered = true;
       scrollDown();
     }, 250);
   }

   function renderMarkdown(text, host) {
     // Build into a detached fragment and swap it in atomically at the end, so
     // the message is never momentarily empty (clearing innerHTML then refilling
     // on every debounced streaming pass is what made the answer flicker).
     var frag = document.createDocumentFragment();
     var lines = String(text == null ? "" : text).split("\n");
     var i = 0;
     var list = null;
     function endList() { list = null; }
     while (i < lines.length) {
       var line = lines[i];
       var fence = line.match(/^\s*```(\w+)?\s*$/);
       if (fence) {
         endList();
         var lang = fence[1] || "";
         var buf = [];
         i++;
         while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
         i++;
         frag.appendChild(makeCodeBlock(buf.join("\n"), lang));
         continue;
       }
       var heading = line.match(/^(#{1,4})\s+(.*)$/);
       if (heading) {
         endList();
         var h = document.createElement("h" + Math.min(4, heading[1].length + 2));
         h.className = "md-h";
         renderInline(heading[2], h);
         frag.appendChild(h);
         i++;
         continue;
       }
       var item = line.match(/^\s*[-*+]\s+(.*)$/) || line.match(/^\s*(\d+)\.\s+(.*)$/);
       if (item) {
         var ordered = /^\s*\d+\./.test(line);
         if (!list || list.ordered !== ordered) {
           var el = document.createElement(ordered ? "ol" : "ul");
           el.className = "md-list";
           frag.appendChild(el);
           list = { el: el, ordered: ordered };
         }
         var li = document.createElement("li");
         renderInline(item.length === 2 ? item[1] : item[2], li);
         list.el.appendChild(li);
         i++;
         continue;
       }
       if (/^\s*>\s?/.test(line)) {
         endList();
         var q = document.createElement("blockquote");
         q.className = "md-quote";
         renderInline(line.replace(/^\s*>\s?/, ""), q);
         frag.appendChild(q);
         i++;
         continue;
       }
       if (!line.trim()) { endList(); i++; continue; }
       endList();
       var p = document.createElement("p");
       p.className = "md-p";
       var block = [line];
       i++;
       while (i < lines.length && lines[i].trim() && !/^\s*(```|#{1,4}\s|[-*+]\s|\d+\.\s|>)/.test(lines[i])) {
         block.push(lines[i]);
         i++;
       }
       renderInline(block.join("\n"), p);
       frag.appendChild(p);
     }
     host.replaceChildren(frag);
     return host;
   }

   function addMessage(role, text) {
     var div = document.createElement("div");
     div.className = "msg " + role;
     if (role === "assistant") {
       var body = document.createElement("div");
       renderMarkdown(text, body);
       div.appendChild(body);
     } else {
       div.textContent = text;
     }
     addMessageActions(div, role, text);
     chat.appendChild(div);
     scrollDown();
     return div;
   }

   // Copy / edit-and-resend controls under each message.
   function addMessageActions(div, role, text) {
     var bar = document.createElement("div");
     bar.className = "msg-actions";
     var copy = document.createElement("button");
     copy.textContent = "Copy";
     copy.title = "Copy this message to the clipboard";
     copy.onclick = function() {
       var payload = div.dataset.raw || text || div.textContent || "";
       navigator.clipboard.writeText(payload).then(function() {
         copy.textContent = "Copied";
         setTimeout(function() { copy.textContent = "Copy"; }, 1200);
       }, function() { alert("Could not copy."); });
     };
     bar.appendChild(copy);
     if (role === "user") {
       var edit = document.createElement("button");
       edit.textContent = "Edit & resend";
       edit.title = "Put this question back in the box to change and ask again";
       edit.onclick = function() {
         input.value = div.dataset.raw || text || div.textContent || "";
         input.focus();
         showView("chat");
       };
       bar.appendChild(edit);
     }
     div.dataset.raw = text || "";
     div.appendChild(bar);
   }

   function addSystem(text) {
     var div = document.createElement("div");
     div.className = "system-msg";
     div.textContent = text;
     chat.appendChild(div);
     scrollDown();
     return div;
   }

   // --- Glass box: build one trace timeline per assistant turn. ---
   function startTrace() {
     var turn = document.createElement("div");
     turn.className = "turn";
     var trace = document.createElement("div");
     trace.className = "trace";
     turn.appendChild(trace);
     // Always-visible activity line: a spinner, the current state, and a clock
     // that keeps ticking so a slow step never looks frozen.
     var activity = document.createElement("div");
     activity.className = "gactivity";
     var spin = document.createElement("span"); spin.className = "gspin";
     var label = document.createElement("span"); label.className = "glabel"; label.textContent = "starting\u2026";
     var elapsed = document.createElement("span"); elapsed.className = "gelapsed";
     activity.appendChild(spin); activity.appendChild(label); activity.appendChild(elapsed);
     turn.appendChild(activity);
     chat.appendChild(turn);
     scrollDown();
     var t = { turn: turn, trace: trace, answer: null, tool: null,
               activity: activity, label: label, elapsed: elapsed,
               started: Date.now(), stepLabel: "", timer: null };
     // Tick the clock four times a second. This is the liveness proof: as long
     // as this number moves, the turn is not dead.
     t.timer = setInterval(function() {
       var secs = (Date.now() - t.started) / 1000;
       t.elapsed.textContent = secs.toFixed(1) + "s";
       // If a single step runs long, flag it visually so a real stall is obvious.
       activity.classList.toggle("stalled", secs > 25 && !t.answered);
     }, 250);
     return t;
   }

   function setActivity(t, text) {
     if (!t.activity) return;
     t.label.textContent = t.stepLabel ? (t.stepLabel + " \u00b7 " + text) : text;
   }

   function stopActivity(t) {
     if (t.timer) { clearInterval(t.timer); t.timer = null; }
     t.answered = true;
     if (t.activity && t.activity.parentNode) t.activity.parentNode.removeChild(t.activity);
   }

   function bumpActivity(t) {
     // Keep the activity line as the last child of the turn as nodes are added.
     if (t.activity) t.turn.appendChild(t.activity);
   }

   function traceRouter(t, name, args) {
     var node = document.createElement("div");
     node.className = "gnode router";
     var head = document.createElement("div");
     head.className = "ghead";
     var loc = args && (args.location || args.query);
     head.textContent = "router \u2192 " + name + (loc ? " (" + String(loc).slice(0, 60) + ")" : "");
     node.appendChild(head);
     t.trace.appendChild(node);
     scrollDown();
   }

   function traceTool(t, name, args) {
     var node = document.createElement("div");
     node.className = "gnode tool";
     var head = document.createElement("div");
     head.className = "ghead";
     var label = document.createElement("span");
     label.className = "gtool";
     label.textContent = name;
     var pill = document.createElement("span");
     pill.className = "gpill";
     pill.textContent = "running";
     var caret = document.createElement("span");
     caret.className = "gcaret";
     caret.textContent = "\u25b8";
     head.appendChild(label);
     head.appendChild(pill);
     head.appendChild(caret);
     var argsEl = document.createElement("div");
     argsEl.className = "gargs";
     argsEl.textContent = JSON.stringify(args || {});
     var body = document.createElement("pre");
     body.className = "gbody";
     body.textContent = "";
     node.appendChild(head);
     node.appendChild(argsEl);
     node.appendChild(body);
     head.onclick = function() { node.classList.toggle("open"); };
     t.trace.appendChild(node);
     scrollDown();
     return { node: node, pill: pill, body: body };
   }

   function traceThinking(t) {
     if (t.thinking) return t.thinking;
     var node = document.createElement("div");
     node.className = "gnode thinking open";
     var head = document.createElement("div");
     head.className = "ghead";
     var tag = document.createElement("span");
     tag.className = "gtool";
     tag.textContent = "thinking";
     var caret = document.createElement("span");
     caret.className = "gcaret";
     caret.textContent = "\u25b8";
     head.appendChild(tag);
     head.appendChild(caret);
     var body = document.createElement("pre");
     body.className = "gbody";
     body.textContent = "";
     node.appendChild(head);
     node.appendChild(body);
     head.onclick = function() { node.classList.toggle("open"); };
     t.trace.appendChild(node);
     scrollDown();
     t.thinking = { node: node, body: body };
     return t.thinking;
   }

   function traceReasonStep(t, step, total, label) {
     var node = document.createElement("div");
     node.className = "gnode reason";
     var head = document.createElement("div");
     head.className = "ghead";
     var tag = document.createElement("span");
     tag.className = "gtool";
     tag.textContent = "reasoning " + step + "/" + total;
     var lab = document.createElement("span");
     lab.style.color = "#9aa0a6";
     lab.textContent = label || "";
     head.appendChild(tag); head.appendChild(lab);
     var body = document.createElement("pre");
     body.className = "gbody";
     body.textContent = "";
     node.appendChild(head); node.appendChild(body);
     t.trace.appendChild(node);
     scrollDown();
     return { node: node, body: body };
   }

   function traceDetail(t, message) {
     var el = document.createElement("div");
     el.className = "gdetail";
     el.textContent = message;
     t.turn.appendChild(el);
     scrollDown();
   }

   function traceNotice(t, message, info) {
     var node = document.createElement("div");
     node.className = "gnode notice" + (info ? " info" : "");
     var msg = document.createElement("div");
     msg.className = "gmsg";
     msg.textContent = message;
     node.appendChild(msg);
     t.trace.appendChild(node);
     scrollDown();
   }

   function traceAnswer(t) {
     if (t.answer) return t.answer;
     var node = document.createElement("div");
     node.className = "gnode answer pending";
     var text = document.createElement("div");
     text.className = "gtext";
     node.appendChild(text);
     t.turn.appendChild(node);
     t.answer = { node: node, text: text };
     scrollDown();
     return t.answer;
   }

   function addFeedbackBar(userText, botText) {
     var bar = document.createElement("div");
     bar.className = "feedback";

     var up = document.createElement("button");
     up.textContent = "yes";
     up.title = "Good answer";
     var down = document.createElement("button");
     down.textContent = "no";
     down.title = "Bad answer";
     var correct = document.createElement("button");
     correct.textContent = "edit";
     correct.title = "Provide a corrected answer";

     up.onclick = function() { vote(userText, botText, 1); up.disabled = true; down.disabled = true; };
     down.onclick = function() { vote(userText, botText, -1); up.disabled = true; down.disabled = true; };
     correct.onclick = function() { correctAnswer(userText, botText); };

     bar.appendChild(up);
     bar.appendChild(down);
     bar.appendChild(correct);
     chat.appendChild(bar);
     scrollDown();
   }

   async function fetchJSON(url, options) {
     var res = await fetch(url, options);
     // A 401 on any authenticated request means the session ended (expired,
     // revoked by an admin, or logged out in another tab): return to login.
     if (res.status === 401 && typeof onUnauthorized === "function") onUnauthorized();
     var raw = await res.text();
     var data = {};
     if (raw) {
       try {
         data = JSON.parse(raw);
       } catch (err) {
         throw new Error(
           "HTTP " + res.status + " from " + (res.url || url) +
           " returned non-JSON, so this page is probably talking to the model " +
           "server instead of the web UI. Body: " + raw.slice(0, 200)
         );
       }
     }
     // Throw on any non-2xx so callers' catch blocks surface the real error
     // (a 403/400/500/503) instead of silently reading fields off an error body
     // and rendering an empty, healthy-looking state.
     if (!res.ok) {
       throw new Error((data && (data.error || data.detail)) || ("HTTP " + res.status));
     }
     return { res: res, data: data };
   }

   // Download a URL as a file via fetch+blob (not window.location), so a
   // 401/403/404/500 shows a message instead of replacing the whole app tab.
   async function downloadFile(url, filename) {
     try {
       var resp = await fetch(url);
       if (resp.status === 401 && typeof onUnauthorized === "function") { onUnauthorized(); return; }
       if (!resp.ok) {
         var e = {}; try { e = await resp.json(); } catch (x) {}
         addSystem("Download failed: " + (e.error || e.detail || ("HTTP " + resp.status)));
         return;
       }
       var text = await resp.text();
       var a = document.createElement("a");
       a.href = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
       a.download = filename;
       document.body.appendChild(a); a.click(); a.remove();
       URL.revokeObjectURL(a.href);
     } catch (err) {
       addSystem("Download failed: " + err.message);
     }
   }

   function updateContextMeter(delta) {
     usedTokens += delta;
     var pct = Math.min(100, Math.round((usedTokens / contextSize) * 100));
     document.getElementById("ctxBar").style.width = pct + "%";
     document.getElementById("ctxHint").textContent =
       "About " + usedTokens + " of " + contextSize + " tokens used. Oldest turns drop out automatically.";
   }

   async function send() {
     if (busy) return;
     var message = input.value.trim();
     if (!message) return;

     input.value = "";
     input.style.height = "auto";
     busy = true;
     sendBtn.disabled = true;
     stopBtn.disabled = false;
     controller = new AbortController();
     addMessage("user", message);
     updateContextMeter(estimateTokens(message));

     var trace = startTrace();
     var answered = false;
     var firstTool = true;

     try {
       var res = await fetch("/api/chat/stream", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         signal: controller.signal,
         body: JSON.stringify({
           message: message,
           conversation_id: conversationId,
           agent: agentToggle.checked
         })
       });

       if (!res.ok || !res.body) {
         var text = await res.text();
         // Session ended mid-chat (this is a raw fetch, not fetchJSON): return
         // to the login screen instead of showing a cryptic error.
         if (res.status === 401 && typeof onUnauthorized === "function") {
           onUnauthorized(); stopActivity(trace); return;
         }
         // Show the clean message, not the raw {"error": ...} JSON envelope.
         var msg = text;
         try { var j = JSON.parse(text); msg = j.error || j.detail || text; } catch (e) {}
         var a0 = traceAnswer(trace);
         a0.text.textContent = "Error: " + String(msg).slice(0, 400);
         a0.node.classList.remove("pending");
         a0.node.classList.add("failed");
         stopActivity(trace);
         return;
       }

       var reader = res.body.getReader();
       var decoder = new TextDecoder();
       var buffer = "";

       while (true) {
         var chunk = await reader.read();
         if (chunk.done) break;
         buffer += decoder.decode(chunk.value, { stream: true });
         var parts = buffer.split("\n\n");
         buffer = parts.pop();

         for (var i = 0; i < parts.length; i++) {
           var line = parts[i].trim();
           if (line.indexOf("data: ") !== 0) continue;
           var payload = line.slice(6);
           if (payload === "[DONE]") continue;
           var event;
           try {
             event = JSON.parse(payload);
           } catch (err) {
             continue;
           }
           handleEvent(event);
         }
       }

       function handleEvent(event) {
         if (event.type === "start") {
           conversationId = event.conversation_id || conversationId;
           localStorage.setItem("llm_conversation", conversationId);
         } else if (event.type === "detail") {
           traceDetail(trace, event.message || "");
           bumpActivity(trace);
         } else if (event.type === "phase") {
           setActivity(trace, event.label || "working\u2026");
           bumpActivity(trace);
         } else if (event.type === "reason_step") {
           trace.reason = traceReasonStep(trace, event.step, event.total, event.label);
           bumpActivity(trace);
           setActivity(trace, "reasoning " + event.step + "/" + event.total + "\u2026");
         } else if (event.type === "reason_token") {
           if (trace.reason) { trace.reason.body.textContent += event.token; scrollDown(); }
         } else if (event.type === "reason_done") {
           if (trace.reason) {
             trace.reason.node.classList.add("done");
             if (event.conclusion) trace.reason.body.textContent = event.conclusion;
             trace.reason = null;
           }
         } else if (event.type === "context") {
           traceNotice(trace, "trimmed " + event.dropped + " old messages to fit the context window", true);
           bumpActivity(trace);
         } else if (event.type === "step") {
           // Show which step is active and out of how many, always.
           trace.stepLabel = "step " + event.step + (event.max_steps ? "/" + event.max_steps : "");
           setActivity(trace, "thinking\u2026");
           if (trace.answer && trace.answer.node.classList.contains("pending")) {
             trace.answer.text.textContent = "";
           }
         } else if (event.type === "think_token") {
           traceThinking(trace).body.textContent += event.token;
           bumpActivity(trace);
           setActivity(trace, "thinking\u2026");
           scrollDown();
         } else if (event.type === "token") {
           // First answer token: the thinking phase is over, collapse it so the
           // answer is the focus but the reasoning stays one click away.
           if (trace.thinking && !trace.thinking.done) {
             trace.thinking.done = true;
             trace.thinking.node.classList.remove("open");
           }
           var ansNode = traceAnswer(trace);
           ansNode.raw = (ansNode.raw || "") + event.token;
           // Only write the growing raw text while markdown has not rendered yet
           // (or a code fence is open). Once rendered, leave the formatted DOM in
           // place and let the debounced renderMarkdown update it — writing raw
           // text every token here is what made the answer flicker.
           if (!ansNode.rendered) ansNode.text.textContent = ansNode.raw;
           scheduleMarkdown(ansNode);
           bumpActivity(trace);
           setActivity(trace, "generating\u2026");
           scrollDown();
         } else if (event.type === "tool_call") {
           if (firstTool) { traceRouter(trace, event.name, event.args); firstTool = false; }
           trace.tool = traceTool(trace, event.name, event.args);
           trace.tool.node.classList.add("running");
           bumpActivity(trace);
           setActivity(trace, "running " + event.name + "\u2026");
         } else if (event.type === "tool_result") {
           if (trace.tool) {
             trace.tool.node.classList.remove("running");
             trace.tool.body.textContent = event.result || "";
             if (event.error) {
               trace.tool.node.classList.add("failed");
               trace.tool.pill.textContent = "failed";
             } else {
               trace.tool.node.classList.add("ok");
               trace.tool.pill.textContent = "ok";
             }
             trace.tool = null;
           }
           trace.answer = null;
           trace.thinking = null;
           setActivity(trace, "thinking\u2026");
         } else if (event.type === "notice") {
           traceNotice(trace, event.message || "", !!event.info);
           bumpActivity(trace);
           setActivity(trace, event.message || "working\u2026");
         } else if (event.type === "final") {
           var ans = traceAnswer(trace);
           if (ans.mdTimer) { clearTimeout(ans.mdTimer); ans.mdTimer = null; }
           // Streaming re-renders on a debounce; the final pass is authoritative.
           renderMarkdown(event.answer || "(no answer)", ans.text);
           ans.node.classList.remove("pending");
           answered = true;
           updateContextMeter(estimateTokens(event.answer || ""));
           var meta = document.createElement("div");
           meta.className = "gmeta";
           var bits = [];
           if (event.tools_used && event.tools_used.length) bits.push(event.tools_used.join(", "));
           bits.push((event.steps || 1) + " step" + ((event.steps || 1) === 1 ? "" : "s"));
           bits.push(Math.round((event.elapsed_ms || 0) / 100) / 10 + "s");
           if (event.truncated) bits.push("continued from summary");
           meta.textContent = bits.join(" \u00b7 ");
           stopActivity(trace);
           trace.turn.appendChild(meta);
           if (event.changed_files && event.changed_files.length) {
             var wrap = document.createElement("div");
             wrap.className = "gnode diff open";
             var head = document.createElement("div");
             head.className = "ghead";
             var tag = document.createElement("span");
             tag.className = "gtool";
             tag.textContent = "changed " + event.changed_files.length + " file"
               + (event.changed_files.length === 1 ? "" : "s") + " (review before push)";
             var caret = document.createElement("span");
             caret.className = "gcaret";
             caret.textContent = "\u25b8";
             head.appendChild(tag); head.appendChild(caret);
             var body = document.createElement("pre");
             body.className = "gbody gdiff";
             body.textContent = event.diff && event.diff.trim()
               ? event.diff
               : (event.changed_files.join("\n") + "\n\n(new files — not yet tracked by git)");
             head.onclick = function() { wrap.classList.toggle("open"); };
             wrap.appendChild(head); wrap.appendChild(body);
             trace.turn.appendChild(wrap);
           }
           addFeedbackBar(message, event.answer || "");
           loadRunCurve(conversationId);
           loadPerf();
         } else if (event.type === "cancelled") {
           traceNotice(trace, "stopped", false);
           stopActivity(trace);
           answered = true;
         } else if (event.type === "error") {
           traceNotice(trace, "error: " + event.error, false);
           stopActivity(trace);
           answered = true;
         }
       }

       if (!answered) {
         var a = traceAnswer(trace);
         a.node.classList.remove("pending");
         if (!a.text.textContent) a.text.textContent = "(no response)";
       }
       stopActivity(trace);
     } catch (err) {
       var a2 = traceAnswer(trace);
       a2.node.classList.remove("pending");
       if (err.name === "AbortError") {
         traceNotice(trace, "stopped", false);
       } else {
         // A broken stream on a local single-box setup almost always means the
         // model server ran out of memory and dropped the connection. Say that
         // plainly instead of showing a raw stream error.
         if (!a2.text.textContent) {
           a2.text.textContent = "The connection dropped, most likely the local "
             + "model server ran low on memory. Try a shorter request, or lower "
             + "AUTO_FETCH_RESULTS / the context size.";
         }
         a2.node.classList.add("failed");
       }
       stopActivity(trace);
     } finally {
       busy = false;
       controller = null;
       sendBtn.disabled = false;
       stopBtn.disabled = true;
       input.focus();
     }
   }

   async function loadMemory() {
     var list = document.getElementById("memoryList");
     try {
       var out = await fetchJSON("/api/memory?limit=50");
       var items = out.data.memories || [];
       list.innerHTML = "";
       if (!items.length) {
         list.textContent = "No stored notes.";
         return;
       }
       items.forEach(function(item) {
         var row = document.createElement("div");
         var del = document.createElement("button");
         del.textContent = "x";
         del.title = "Forget this note";
         del.style.marginRight = "6px";
         del.style.padding = "0 6px";
         del.onclick = function() { forgetMemory(item.key); };
         var label = document.createElement("span");
         label.textContent = item.key + ": " + item.value;
         row.appendChild(del);
         row.appendChild(label);
         list.appendChild(row);
       });
     } catch (err) {
       list.textContent = "Memory unavailable: " + err.message;
     }
   }

   async function saveMemory() {
     var key = document.getElementById("memKey").value.trim();
     var value = document.getElementById("memValue").value.trim();
     if (!key || !value) return;
     try {
       await fetchJSON("/api/memory", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ key: key, value: value })
       });
       document.getElementById("memKey").value = "";
       document.getElementById("memValue").value = "";
       loadMemory();
     } catch (err) {
       addSystem("Memory error: " + err.message);
     }
   }

   async function forgetMemory(key) {
     try {
       await fetchJSON("/api/memory/" + encodeURIComponent(key), { method: "DELETE" });
       loadMemory();
     } catch (err) {
       addSystem("Memory error: " + err.message);
     }
   }

   async function vote(userPrompt, assistantResponse, rating) {
     try {
       var out = await fetchJSON("/api/feedback", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({
           user_prompt: userPrompt,
           assistant_response: assistantResponse,
           rating: rating,
           corrected_response: null
         })
       });
       addSystem(out.data.status || "Feedback saved.");
     } catch (err) {
       addSystem("Feedback error: " + err.message);
     }
   }

   async function correctAnswer(userPrompt, assistantResponse) {
     var corrected = window.prompt("Corrected answer:", assistantResponse);
     if (corrected === null) return;
     try {
       var out = await fetchJSON("/api/feedback", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({
           user_prompt: userPrompt,
           assistant_response: assistantResponse,
           rating: 1,
           corrected_response: corrected
         })
       });
       addSystem(out.data.status || "Correction saved.");
     } catch (err) {
       addSystem("Correction error: " + err.message);
     }
   }

   async function retrain() {
     if (!window.confirm("Start retraining? The model server will be stopped temporarily.")) return;
     try {
       var out = await fetchJSON("/api/retrain", { method: "POST" });
       addSystem(out.data.status || JSON.stringify(out.data));
     } catch (err) {
       addSystem("Retrain error: " + err.message);
     }
   }

   async function newChat() {
     conversationId = randomId();
     localStorage.setItem("llm_conversation", conversationId);
     chat.innerHTML = "";
     usedTokens = 0;
     updateContextMeter(0);
     addSystem("New conversation started.");
   }

   function toggleSettings() {
     settingsEl.classList.toggle("open");
   }

   async function restartModel() {
     try {
       var out = await fetchJSON("/api/model/restart", { method: "POST" });
       addSystem("Model server restarting. KV cache: " + out.data.max_kv_size);
     } catch (err) {
       addSystem("Restart error: " + err.message);
     }
   }

   async function loadConfig() {
     try {
       var out = await fetchJSON("/api/config");
       var cfg = out.data.config;
       contextSize = cfg.context_size;
       document.getElementById("cfgSystem").value = cfg.system_prompt;
       document.getElementById("cfgMaxTokens").value = cfg.max_tokens;
       document.getElementById("cfgTemperature").value = cfg.temperature;
       document.getElementById("cfgContext").value = cfg.context_size;
       document.getElementById("cfgHistory").value = cfg.history_turns;
       document.getElementById("cfgAgentSteps").value = cfg.agent_max_steps;
       document.getElementById("cfgSearchBackend").value = cfg.search_backend;
       document.getElementById("cfgSearchResults").value = cfg.search_results;
       document.getElementById("cfgToolChars").value = cfg.tool_result_chars;
       document.getElementById("cfgToolChars2").value = cfg.tool_result_chars;
       document.getElementById("cfgToolTemp").value = cfg.tool_temperature;
       document.getElementById("cfgThinking").checked = !!cfg.disable_thinking;
       document.getElementById("cfgFastPath").checked = !!cfg.fast_path;
       document.getElementById("cfgStablePrefix").checked = !!cfg.stable_prefix;
       document.getElementById("cfgSummarise").checked = !!cfg.summarise_tool_results;
       agentToggle.checked = cfg.agent_enabled;
       var names = out.data.tools.map(function(t) { return t.name; });
       document.getElementById("toolsList").textContent = names.join("\n");
       updateContextMeter(0);
     } catch (err) {
       addSystem("Could not load settings: " + err.message);
     }
   }

   async function saveConfig() {
     var body = {
       system_prompt: document.getElementById("cfgSystem").value,
       max_tokens: Number(document.getElementById("cfgMaxTokens").value),
       temperature: Number(document.getElementById("cfgTemperature").value),
       context_size: Number(document.getElementById("cfgContext").value),
       history_turns: Number(document.getElementById("cfgHistory").value),
       agent_enabled: agentToggle.checked,
       agent_max_steps: Number(document.getElementById("cfgAgentSteps").value),
       search_results: Number(document.getElementById("cfgSearchResults").value),
       tool_result_chars: Number(document.getElementById("cfgToolChars2").value ||
                                 document.getElementById("cfgToolChars").value),
       tool_temperature: Number(document.getElementById("cfgToolTemp").value),
       disable_thinking: document.getElementById("cfgThinking").checked,
       fast_path: document.getElementById("cfgFastPath").checked,
       stable_prefix: document.getElementById("cfgStablePrefix").checked,
       summarise_tool_results: document.getElementById("cfgSummarise").checked
     };
     try {
       var out = await fetchJSON("/api/config", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify(body)
       });
       contextSize = out.data.config.context_size;
       updateContextMeter(0);
       addSystem(out.data.changed.length
         ? "Updated: " + out.data.changed.join(", ")
         : "No settings changed.");
       if (out.data.ignored && out.data.ignored.length)
         addSystem("Ignored invalid value(s): " + out.data.ignored.join(", "));
     } catch (err) {
       addSystem("Settings error: " + err.message);
     }
   }

   async function refreshHealth() {
     try {
       var out = await fetchJSON("/api/health");
       var data = out.data;
       // Non-admins never call loadConfig(), so take the context size (used by
       // the context meter) from health instead.
       if (typeof data.context_size === "number") contextSize = data.context_size;
       // The header line stays short enough to read at a glance; everything
       // else goes to the tooltip, which has room to wrap.
       var brief = ["Model: " + (data.model_status || "unknown")];
       var detail = [];
       var stale = data.ui_build && data.ui_build !== "{{UI_BUILD}}";
       if (stale) {
         brief = ["STALE PAGE \u2014 hard-reload"];
         detail.push("Server is build " + data.ui_build + ", this tab is {{UI_BUILD}}.");
       }
       brief.push("agent " + (data.agent_enabled ? "on" : "off"));
       brief.push("ctx " + data.context_size);
       detail.push("Max reply " + data.max_tokens + " tokens");
       detail.push((data.tools ? data.tools.length : 0) + " tools");
       var retrain = (data.retrain && data.retrain.message) || "idle";
       if (retrain !== "idle") brief.push("retrain: " + retrain);
       else detail.push("Retrain idle");
       if (data.stats) {
         brief.push(data.stats.untrained + " untrained");
         detail.push("Feedback: " + data.stats.total + " total, "
                     + data.stats.approved + " approved, "
                     + data.stats.untrained + " untrained");
         if (data.stats.pending) {
           brief.push(data.stats.pending + " awaiting approval");
         }
       }
       if (typeof data.memories === "number") {
         detail.push(data.memories + " memory notes");
       }
       var warn = document.getElementById("prefixWarn");
       if (data.prefix && data.prefix.generation > 1) {
         warn.style.display = "";
         warn.textContent = "Prompt prefix changed " + (data.prefix.generation - 1) +
           " time(s) this session (last " + (data.prefix.changed_at || "") +
           "). Each change invalidates any cached prefix.";
       } else if (warn) {
         warn.style.display = "none";
       }
       if (data.tasks) {
         var running = data.tasks.running || [];
         if (running.length) {
           brief.push(running.length + " task" + (running.length > 1 ? "s" : "") + " running");
           detail.push("Running: " + running.map(function(item) {
             return "step " + item.step + "/" + item.max_steps;
           }).join(", "));
         }
         detail.push(data.tasks.total + " tasks defined");
       }
       statusEl.textContent = brief.join("  \u00b7  ");
       statusEl.className = "";
       statusEl.setAttribute("data-tip-below", "");
       statusEl.setAttribute("data-tip",
         "Model " + (data.model || "?")
         + " \u00b7 context " + (data.context_size || contextSize) + " tokens"
         + (data.ram_gb ? " \u00b7 " + data.ram_gb + "GB RAM" : "")
         + (detail.length ? " \u00b7 " + detail.join(" \u00b7 ") : ""));
       if (stale || (data.model_status && data.model_status.indexOf("error") === 0)) {
         statusEl.className = "error";
       } else if (data.model_status === "starting" || data.model_status === "loading") {
         statusEl.className = "warn";
       }
     } catch (err) {
       statusEl.textContent = "Status unavailable: " + err.message;
       statusEl.className = "error";
     }
   }
"""

__all__ = ["UI_JS_CHAT"]
