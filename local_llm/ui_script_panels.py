"""Front-end: history, prompts, uploads, command palette, folder picker, theme, backup and boot.

Part of the embedded UI, split out of the single HTML_PAGE string for
readability. Assembled once at import time in ui.py, so there is no
runtime cost and no extra file I/O.
"""

from __future__ import annotations

UI_JS_PANELS = r"""   // ---------------------------------------------------- history panel ---

   function renderConversations(items, isSearch) {
     var host = document.getElementById("historyList");
     if (!items.length) {
       host.textContent = isSearch ? "No conversations matched." : "No conversations yet.";
       return;
     }
     host.innerHTML = "";
     items.forEach(function(c) {
       var id = c.conversation_id || c.id;
       var row = document.createElement("div");
       row.className = "conv-item";
       var title = document.createElement("div");
       title.className = "ctitle";
       title.textContent = (c.pinned ? "\u2605 " : "") + (c.title || id).toString().slice(0, 90);
       var meta = document.createElement("div");
       meta.className = "cmeta";
       var when = c.last_at || c.created_at || "";
       var count = (c.messages !== undefined) ? c.messages : null;
       meta.textContent = id.slice(0, 8) + (when ? " · " + when : "")
         + (count !== null ? " · " + count + " messages" : "");
       row.appendChild(title);
       row.appendChild(meta);
       if (c.snippet) {
         var sn = document.createElement("div");
         sn.className = "csnip";
         sn.textContent = c.snippet;
         row.appendChild(sn);
       }
       var bar = document.createElement("div");
       bar.className = "msg-actions";
       var open = document.createElement("button");
       open.textContent = "Open";
       open.title = "Reopen this conversation and continue it";
       open.onclick = function() { openConversation(id); };
       var exp = document.createElement("button");
       exp.textContent = "Export";
       exp.title = "Download as Markdown";
       exp.onclick = function() {
         window.location = "/api/conversation/" + encodeURIComponent(id) + "/export?format=markdown";
       };
       var del = document.createElement("button");
       del.textContent = "Delete";
       del.title = "Delete this conversation permanently";
       del.onclick = async function() {
         if (!confirm("Delete this conversation? This cannot be undone.")) return;
         try {
           await fetchJSON("/api/conversation/" + encodeURIComponent(id), { method: "DELETE" });
           loadHistory();
         } catch (err) { alert("Delete failed: " + err.message); }
       };
       var pin = document.createElement("button");
       pin.textContent = c.pinned ? "Unpin" : "Pin";
       pin.title = "Keep this conversation at the top of the list";
       pin.onclick = async function() {
         try {
           await fetchJSON("/api/conversation/" + encodeURIComponent(id) + "/pin?pinned="
                           + (c.pinned ? "false" : "true"), { method: "POST" });
           loadHistory();
         } catch (err) { alert("Could not pin: " + err.message); }
       };
       var ren = document.createElement("button");
       ren.textContent = "Rename";
       ren.title = "Give this conversation a name";
       ren.onclick = async function() {
         var name = prompt("Name this conversation:", c.title || "");
         if (!name) return;
         try {
           await fetchJSON("/api/conversation/" + encodeURIComponent(id) + "/title?title="
                           + encodeURIComponent(name), { method: "POST" });
           loadHistory();
         } catch (err) { alert("Could not rename: " + err.message); }
       };
       var fork = document.createElement("button");
       fork.textContent = "Fork";
       fork.title = "Copy this conversation so you can take it a different direction";
       fork.onclick = async function() {
         try {
           var fr = await fetchJSON("/api/conversation/" + encodeURIComponent(id) + "/fork",
                                    { method: "POST" });
           var nid = (fr.data || fr).conversation_id;
           addSystem("Forked into a new conversation.");
           openConversation(nid);
         } catch (err) { alert("Could not fork: " + err.message); }
       };
       bar.appendChild(open); bar.appendChild(fork); bar.appendChild(pin);
       bar.appendChild(ren); bar.appendChild(exp); bar.appendChild(del);
       row.appendChild(bar);
       host.appendChild(row);
     });
   }

   async function loadHistory() {
     var host = document.getElementById("historyList");
     if (!host) return;
     host.textContent = "loading...";
     try {
       var r = await fetchJSON("/api/conversations");
       renderConversations((r.data || r).conversations || [], false);
     } catch (err) {
       host.textContent = "Could not load conversations: " + err.message;
     }
   }

   async function runHistorySearch() {
     var q = (document.getElementById("historySearch").value || "").trim();
     if (!q) { loadHistory(); return; }
     var host = document.getElementById("historyList");
     host.textContent = "searching...";
     try {
       var r = await fetchJSON("/api/conversations/search?q=" + encodeURIComponent(q));
       renderConversations((r.data || r).results || [], true);
     } catch (err) {
       host.textContent = "Search failed: " + err.message;
     }
   }

   async function openConversation(id) {
     try {
       var r = await fetchJSON("/api/conversation/" + encodeURIComponent(id));
       var msgs = (r.data || r).messages || [];
       conversationId = id;
       localStorage.setItem("llm_conversation", id);
       chat.innerHTML = "";
       msgs.forEach(function(m) {
         if (m.role === "user" || m.role === "assistant") addMessage(m.role, m.content || "");
       });
       showView("chat");
     } catch (err) {
       alert("Could not open conversation: " + err.message);
     }
   }

   // ---------------------------------------------------- prompt library ---

   function togglePrompts() {
     var panel = document.getElementById("promptsPanel");
     panel.classList.toggle("hidden");
     if (!panel.classList.contains("hidden")) loadPrompts();
   }

   async function loadPrompts() {
     var host = document.getElementById("promptList");
     if (!host) return;
     host.textContent = "loading...";
     try {
       var r = await fetchJSON("/api/prompts");
       var items = (r.data || r).prompts || [];
       if (!items.length) { host.textContent = "No saved prompts yet."; return; }
       host.innerHTML = "";
       items.forEach(function(pr) {
         var row = document.createElement("div");
         row.className = "prompt-item";
         var n = document.createElement("div");
         n.className = "pname"; n.textContent = pr.name;
         var b = document.createElement("div");
         b.className = "pbody"; b.textContent = pr.body;
         var bar = document.createElement("div");
         bar.className = "msg-actions";
         var ins = document.createElement("button");
         ins.textContent = "Insert";
         ins.title = "Put this prompt in the message box";
         ins.onclick = function() { input.value = pr.body; input.focus(); showView("chat"); };
         var del = document.createElement("button");
         del.textContent = "Delete";
         del.onclick = async function() {
           try {
             await fetchJSON("/api/prompts/" + encodeURIComponent(pr.name), { method: "DELETE" });
             loadPrompts();
           } catch (err) { alert("Delete failed: " + err.message); }
         };
         bar.appendChild(ins); bar.appendChild(del);
         row.appendChild(n); row.appendChild(b); row.appendChild(bar);
         host.appendChild(row);
       });
     } catch (err) {
       host.textContent = "Could not load prompts: " + err.message;
     }
   }

   async function savePrompt() {
     var name = (document.getElementById("promptName").value || "").trim();
     var body = (input.value || "").trim();
     if (!name) { alert("Give the prompt a name."); return; }
     if (!body) { alert("Type the prompt in the message box first, then save it."); return; }
     try {
       await fetchJSON("/api/prompts?name=" + encodeURIComponent(name) + "&body=" + encodeURIComponent(body),
                       { method: "POST" });
       document.getElementById("promptName").value = "";
       loadPrompts();
     } catch (err) {
       alert("Could not save: " + err.message);
     }
   }

   // ------------------------------------------------- file drag & drop ---

   async function uploadFiles(files) {
     if (!files || !files.length) return;
     for (var i = 0; i < files.length; i++) {
       var form = new FormData();
       form.append("file", files[i]);
       try {
         var resp = await fetch("/api/docs/upload", { method: "POST", body: form });
         var data = await resp.json();
         if (data.error) { alert(files[i].name + ": " + data.error); continue; }
         addSystem(data.result || ("Indexed " + files[i].name));
       } catch (err) {
         alert("Upload failed for " + files[i].name + ": " + err.message);
       }
     }
     loadDocsStats();
   }

   function wireDropZone() {
     var zone = document.getElementById("dropZone");
     var picker = document.getElementById("fileInput");
     if (!zone || !picker) return;
     zone.onclick = function() { picker.click(); };
     picker.onchange = function() { uploadFiles(picker.files); picker.value = ""; };
     ["dragenter", "dragover"].forEach(function(ev) {
       zone.addEventListener(ev, function(e) {
         e.preventDefault(); e.stopPropagation(); zone.classList.add("dragging");
       });
     });
     ["dragleave", "drop"].forEach(function(ev) {
       zone.addEventListener(ev, function(e) {
         e.preventDefault(); e.stopPropagation(); zone.classList.remove("dragging");
       });
     });
     zone.addEventListener("drop", function(e) {
       if (e.dataTransfer && e.dataTransfer.files) uploadFiles(e.dataTransfer.files);
     });
   }

   // ------------------------------------------------- command palette ---

   var PALETTE_COMMANDS = [
     { label: "New chat", hint: "start a fresh conversation", run: function() { newChat(); } },
     { label: "Regenerate last answer", hint: "same question, new answer", run: function() { regenerateLast(); } },
     { label: "Search conversations", hint: "history", run: function() { showView("history"); } },
     { label: "Export this conversation", hint: "markdown", run: function() { exportChat(); } },
     { label: "Prompt library", hint: "saved prompts", run: function() { togglePrompts(); } },
     { label: "Knowledge base", hint: "index documents", run: function() { showView("models"); } },
     { label: "Toggle theme", hint: "light / dark", run: function() { toggleTheme(); } },
     { label: "Download backup", hint: "save everything", run: function() { downloadBackup(); } },
     { label: "Settings", hint: "model and generation", run: function() { toggleSettings(); } },
     { label: "Chat", hint: "back to the conversation", run: function() { showView("chat"); } },
     { label: "Tasks", hint: "scheduled jobs", run: function() { showView("tasks"); } }
   ];
   var paletteSel = 0;

   function renderPalette() {
     var q = (document.getElementById("paletteInput").value || "").toLowerCase();
     var list = document.getElementById("paletteList");
     var matches = PALETTE_COMMANDS.filter(function(c) {
       return !q || c.label.toLowerCase().indexOf(q) >= 0 || c.hint.toLowerCase().indexOf(q) >= 0;
     });
     if (paletteSel >= matches.length) paletteSel = 0;
     list.innerHTML = "";
     matches.forEach(function(c, i) {
       var row = document.createElement("div");
       row.className = "pal-item" + (i === paletteSel ? " sel" : "");
       row.textContent = c.label;
       var hint = document.createElement("span");
       hint.className = "palhint"; hint.textContent = c.hint;
       row.appendChild(hint);
       row.onclick = function() { closePalette(); c.run(); };
       list.appendChild(row);
     });
     list.dataset.count = matches.length;
     return matches;
   }

   function openPalette() {
     document.getElementById("palette").classList.remove("hidden");
     var box = document.getElementById("paletteInput");
     box.value = ""; paletteSel = 0; renderPalette(); box.focus();
   }

   function closePalette() {
     document.getElementById("palette").classList.add("hidden");
   }

   // --------------------------------------------------- folder picker ---

   var browsePath = "";

   async function openBrowser(start) {
     document.getElementById("browser").classList.remove("hidden");
     await loadBrowse(start || (document.getElementById("cfgProjectDir").value || "").trim());
   }

   function closeBrowser() {
     document.getElementById("browser").classList.add("hidden");
   }

   async function loadBrowse(path) {
     var list = document.getElementById("browserList");
     var label = document.getElementById("browserPath");
     list.textContent = "loading...";
     try {
       var r = await fetchJSON("/api/browse?path=" + encodeURIComponent(path || ""));
       var d = r.data || r;
       if (d.error) { list.textContent = d.error; return; }
       browsePath = d.path;
       label.textContent = d.path + (d.is_git_repo ? "   (git repository)" : "");
       list.innerHTML = "";
       if (d.parent) {
         var up = document.createElement("div");
         up.className = "dir-item";
         up.textContent = "../";
         up.onclick = function() { loadBrowse(d.parent); };
         list.appendChild(up);
       }
       if (!d.entries.length) {
         var none = document.createElement("div");
         none.className = "dir-item";
         none.textContent = "(no sub-folders)";
         list.appendChild(none);
       }
       d.entries.forEach(function(e) {
         var row = document.createElement("div");
         row.className = "dir-item";
         var name = document.createElement("span");
         name.textContent = e.name + "/";
         row.appendChild(name);
         if (e.is_git_repo) {
           var tag = document.createElement("span");
           tag.className = "repo";
           tag.textContent = "git repo";
           row.appendChild(tag);
         }
         row.onclick = function() { loadBrowse(e.path); };
         list.appendChild(row);
       });
     } catch (err) {
       list.textContent = "Could not browse: " + err.message;
     }
   }

   function chooseCurrentFolder() {
     if (!browsePath) return;
     document.getElementById("cfgProjectDir").value = browsePath;
     closeBrowser();
     saveProjectDir();
   }

   // ----------------------------------------------------------- theme ---

   function applyTheme(name) {
     var light = name === "light";
     document.documentElement.classList.toggle("light", light);
     var btn = document.getElementById("themeBtn");
     if (btn) btn.textContent = light ? "Dark" : "Light";
     try { localStorage.setItem("llm_theme", light ? "light" : "dark"); } catch (err) { /* private mode */ }
   }

   function toggleTheme() {
     applyTheme(document.documentElement.classList.contains("light") ? "dark" : "light");
   }

   // ---------------------------------------------------------- backup ---

   function downloadBackup() {
     window.location = "/api/backup";
   }

   async function restoreBackup(file) {
     if (!file) return;
     try {
       var text = await file.text();
       var data = JSON.parse(text);
       var r = await fetchJSON("/api/backup/restore", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify(data)
       });
       var d = (r.data || r).restored || {};
       alert("Restored " + (d.messages || 0) + " messages, " + (d.prompts || 0) + " prompts, "
             + (d.feedback || 0) + " feedback rows, " + (d.documents || 0) + " documents.");
       loadHistory();
     } catch (err) {
       alert("Restore failed: " + err.message);
     }
   }

   async function loadModels() {
     loadDatasetStats();
     loadDocsStats();
     loadProjectStatus();
     try {
       var out = await fetchJSON("/api/models");
       var current = out.data.current;
       document.getElementById("modelCurrent").textContent =
         "model    : " + current.model + (current.cached ? "  (cached)" : "  (not downloaded)") +
         "\nadapter  : " + current.adapter + (current.adapter_path ? "  " + current.adapter_path : "") +
         "\nkv cache : " + (current.max_kv_size || "unbounded") +
         "\nstatus   : " + current.status +
         "\ncache dir: " + out.data.cache_dir;

       var table = document.getElementById("modelTable");
       table.innerHTML = "";
       (out.data.catalog || []).forEach(function(item) {
         var row = document.createElement("tr");
         var id = document.createElement("td");
         id.className = "id";
         id.textContent = item.id;
         var state = document.createElement("td");
         state.style.width = "110px";
         state.appendChild(statusPill(item.current ? "running" : (item.cached ? "ok" : "")));
         state.lastChild.textContent = item.current ? "in use" : (item.cached ? "cached" : "download");
         var action = document.createElement("td");
         action.style.width = "70px";
         var button = document.createElement("button");
         button.textContent = "Use";
         button.disabled = item.current;
         button.onclick = function() { useModel(item.id); };
         action.appendChild(button);
         row.appendChild(id);
         row.appendChild(state);
         row.appendChild(action);
         table.appendChild(row);
       });

       var select = document.getElementById("adapterSelect");
       select.innerHTML = "";
       (out.data.adapters || []).forEach(function(item) {
         var option = document.createElement("option");
         option.value = item.id;
         option.textContent = item.id + (item.modified ? "  (" + relativeTime(item.modified) + ")" : "");
         select.appendChild(option);
       });
       select.value = current.adapter;
       document.getElementById("kvSize").value = current.max_kv_size || 0;
     } catch (err) {
       document.getElementById("modelCurrent").textContent = "Could not load models: " + err.message;
     }
   }

   async function postModel(body) {
     try {
       var out = await fetchJSON("/api/models/select", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify(body)
       });
       if (out.data.error) {
         alert(out.data.error);
         return;
       }
       loadModels();
       loadModelLog();
     } catch (err) {
       alert("Could not switch: " + err.message);
     }
   }

   function useModel(modelId) {
     if (!modelId) return;
     if (!window.confirm("Switch to " + modelId + "? The model server restarts, " +
                         "and an uncached model downloads first.")) return;
     postModel({ model: modelId, restart: true });
   }

   function applyModel() {
     postModel({
       adapter: document.getElementById("adapterSelect").value,
       max_kv_size: Number(document.getElementById("kvSize").value) || 0,
       restart: true
     });
   }

   async function loadModelLog() {
     try {
       var out = await fetchJSON("/api/logs/model?lines=120");
       var box = document.getElementById("modelLog");
       box.textContent = out.data.text || "(no output yet)";
       box.scrollTop = box.scrollHeight;
     } catch (err) {
       document.getElementById("modelLog").textContent = "Log unavailable: " + err.message;
     }
   }

   loadConfig();
   loadMemory();
   loadPerf();
   wireDropZone();

   // Restore the saved colour scheme before anything renders.
   try {
     applyTheme(localStorage.getItem("llm_theme") === "light" ? "light" : "dark");
   } catch (err) { applyTheme("dark"); }

   var _restoreInput = document.getElementById("restoreInput");
   if (_restoreInput) _restoreInput.addEventListener("change", function() {
     if (_restoreInput.files && _restoreInput.files[0]) restoreBackup(_restoreInput.files[0]);
     _restoreInput.value = "";
   });

   // Cmd/Ctrl+K opens the command palette; arrows and Enter drive it, Esc closes.
   document.addEventListener("keydown", function(e) {
     var pal = document.getElementById("palette");
     var open = pal && !pal.classList.contains("hidden");
     if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
       e.preventDefault();
       open ? closePalette() : openPalette();
       return;
     }
     if (!open) return;
     if (e.key === "Escape") { e.preventDefault(); closePalette(); return; }
     var matches = renderPalette();
     if (e.key === "ArrowDown") {
       e.preventDefault(); paletteSel = Math.min(paletteSel + 1, matches.length - 1); renderPalette();
     } else if (e.key === "ArrowUp") {
       e.preventDefault(); paletteSel = Math.max(paletteSel - 1, 0); renderPalette();
     } else if (e.key === "Enter") {
       e.preventDefault();
       var chosen = matches[paletteSel];
       closePalette();
       if (chosen) chosen.run();
     }
   });
   var _palInput = document.getElementById("paletteInput");
   if (_palInput) _palInput.addEventListener("input", function() { paletteSel = 0; renderPalette(); });
   var _palOverlay = document.getElementById("palette");
   if (_palOverlay) _palOverlay.addEventListener("click", function(e) {
     if (e.target === _palOverlay) closePalette();
   });

   setInterval(refreshHealth, 3000);
   refreshHealth();
 """

__all__ = ["UI_JS_PANELS"]
