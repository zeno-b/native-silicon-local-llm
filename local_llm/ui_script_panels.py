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
         downloadFile("/api/conversation/" + encodeURIComponent(id) + "/export?format=markdown",
                      "conversation-" + id + ".md");
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
         if (resp.status === 401 && typeof onUnauthorized === "function") { onUnauthorized(); return; }
         var data = {};
         try { data = await resp.json(); } catch (e) {}
         if (!resp.ok) {
           alert(files[i].name + ": " + (data.error || data.detail || ("HTTP " + resp.status)));
           continue;
         }
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
     { label: "Knowledge base", hint: "index documents", admin: true, run: function() { showAdminSection("knowledge"); } },
     { label: "Agents", hint: "what each agent can do", admin: true, run: function() { showAdminSection("agents"); } },
     { label: "Users", hint: "accounts and roles", admin: true, run: function() { showAdminSection("users"); } },
     { label: "Cluster & routing", hint: "node health", admin: true, run: function() { showAdminSection("cluster"); } },
     { label: "Toggle theme", hint: "light / dark", run: function() { toggleTheme(); } },
     { label: "Download backup", hint: "save everything", admin: true, run: function() { downloadBackup(); } },
     { label: "Settings", hint: "model and generation", admin: true, run: function() { toggleSettings(); } },
     { label: "Chat", hint: "back to the conversation", run: function() { showView("chat"); } },
     { label: "Tasks", hint: "scheduled jobs", admin: true, run: function() { showView("tasks"); } }
   ];
   var paletteSel = 0;

   function renderPalette() {
     var q = (document.getElementById("paletteInput").value || "").toLowerCase();
     var list = document.getElementById("paletteList");
     var isAdmin = document.body.dataset.role === "admin";
     var matches = PALETTE_COMMANDS.filter(function(c) {
       if (c.admin && !isAdmin) return false;  // never expose admin actions to a non-admin
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

   async function downloadBackup() {
     // Fetch + blob-download instead of navigating the tab, so a 403/404/500
     // shows an alert rather than replacing the whole app with an error body.
     try {
       var resp = await fetch("/api/backup");
       if (resp.status === 401 && typeof onUnauthorized === "function") { onUnauthorized(); return; }
       if (!resp.ok) {
         var e = {}; try { e = await resp.json(); } catch (x) {}
         alert("Backup failed: " + (e.error || e.detail || ("HTTP " + resp.status)));
         return;
       }
       var text = await resp.text();
       var a = document.createElement("a");
       a.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
       a.download = "backup-" + Date.now() + ".json";
       document.body.appendChild(a); a.click(); a.remove();
       URL.revokeObjectURL(a.href);
     } catch (err) {
       alert("Backup failed: " + err.message);
     }
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
     loadUsers();
     loadAgents();
     loadCluster();
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
       // Only select an adapter that is actually offered. `current.adapter` is the
       // REQUESTED choice (often "latest"), which is not in the list on an install
       // that has never trained one -- assigning it left the dropdown blank and a
       // subsequent Apply submitted an empty adapter.
       var wanted = current.adapter;
       var offered = Array.prototype.map.call(select.options, function(o) { return o.value; });
       if (offered.indexOf(wanted) < 0) {
         var opt = document.createElement("option");
         opt.value = wanted;
         opt.textContent = wanted + "  (not built yet)";
         select.insertBefore(opt, select.firstChild);
       }
       select.value = wanted;
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

   // ------------------------------------------------------- user admin ---

   async function loadUsers() {
     var host = document.getElementById("usersList");
     if (!host) return;
     try {
       var out = await fetchJSON("/api/users");
       var users = out.data.users || [];
       host.innerHTML = "";
       if (!users.length) { host.textContent = "No users yet."; return; }
       users.forEach(function(u) {
         var row = document.createElement("div");
         row.className = "row";
         row.style.alignItems = "center";
         row.style.padding = "4px 0";
         var who = document.createElement("div");
         who.style.flex = "1 1 auto";
         who.textContent = u.username +
           (u.display_name && u.display_name !== u.username ? " (" + u.display_name + ")" : "") +
           (u.source && u.source !== "local" ? "  · " + u.source : "");
         var pillWrap = document.createElement("div");
         pillWrap.style.flex = "0 0 auto";
         var pill = statusPill(u.disabled ? "" : "ok");
         pill.textContent = u.role + (u.disabled ? " · disabled" : "");
         pillWrap.appendChild(pill);
         var actions = document.createElement("div");
         actions.style.flex = "0 0 auto";
         actions.style.display = "flex";
         actions.style.gap = "6px";
         var roleBtn = document.createElement("button");
         roleBtn.textContent = u.role === "admin" ? "Make user" : "Make admin";
         roleBtn.onclick = function() { updateUser(u.id, { role: u.role === "admin" ? "user" : "admin" }); };
         var disBtn = document.createElement("button");
         disBtn.textContent = u.disabled ? "Enable" : "Disable";
         disBtn.onclick = function() { updateUser(u.id, { disabled: !u.disabled }); };
         var delBtn = document.createElement("button");
         delBtn.textContent = "Delete";
         delBtn.onclick = function() {
           if (window.confirm("Delete user " + u.username + "? Their data stays but they can "
               + "no longer sign in.")) deleteUser(u.id);
         };
         actions.appendChild(roleBtn);
         actions.appendChild(disBtn);
         actions.appendChild(delBtn);
         row.appendChild(who);
         row.appendChild(pillWrap);
         row.appendChild(actions);
         host.appendChild(row);
       });
     } catch (err) {
       host.textContent = "Could not load users: " + err.message;
     }
   }

   async function createUser() {
     var name = (document.getElementById("newUserName").value || "").trim();
     var pass = document.getElementById("newUserPass").value || "";
     var role = document.getElementById("newUserRole").value || "user";
     if (!name || !pass) { alert("Enter a username and a password for the new user."); return; }
     try {
       await fetchJSON("/api/users", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ username: name, password: pass, role: role })
       });
       document.getElementById("newUserName").value = "";
       document.getElementById("newUserPass").value = "";
       loadUsers();
     } catch (err) {
       alert("Could not create user: " + err.message);
     }
   }

   async function updateUser(id, patch) {
     try {
       await fetchJSON("/api/users/" + encodeURIComponent(id), {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify(patch)
       });
       loadUsers();
     } catch (err) {
       alert("Could not update user: " + err.message);
     }
   }

   async function deleteUser(id) {
     try {
       await fetchJSON("/api/users/" + encodeURIComponent(id), { method: "DELETE" });
       loadUsers();
     } catch (err) {
       alert("Could not delete user: " + err.message);
     }
   }

   // --------------------------------------------------- cluster / routing ---

   async function loadCluster() {
     var host = document.getElementById("clusterNodes");
     if (host) {
       try {
         var out = await fetchJSON("/api/cluster/nodes");
         var d = out.data;
         var nodes = d.nodes || [];
         var lines = [];
         lines.push("Topology: " + (d.multi_node ? "multi-node" : "single node")
                    + "   (this node's role: " + (d.node_role || "primary") + ")");
         nodes.forEach(function(n) {
           lines.push("");
           lines.push(n.name + "  [" + n.role + "]  " + n.state + (n.is_local ? "  (local)" : ""));
           if (n.machine || n.ram_gb) {
             lines.push("  hardware " + (n.machine || "?")
                        + (n.cores ? "  " + n.cores + " cores" : "")
                        + (n.ram_gb ? "  " + n.ram_gb + "GB RAM" : ""));
           }
           lines.push("  active " + n.active
                      + "   latency " + (n.last_latency_ms != null ? n.last_latency_ms + "ms" : "-")
                      + "   load " + (n.load_ratio != null ? n.load_ratio + "x/core" : "-")
                      + "   cpu " + (n.cpu_pct != null ? n.cpu_pct + "%" : "n/a")
                      + "   mem " + (n.mem_pct != null ? n.mem_pct + "%" : "n/a"));
           lines.push("  caps " + ((n.capabilities || []).join(", ") || "-")
                      + (n.consecutive_failures ? "   failures " + n.consecutive_failures : "")
                      + (n.cooldown_remaining_s != null ? "   cooldown " + n.cooldown_remaining_s + "s" : ""));
           if (n.model) lines.push("  model " + n.model);
         });
         host.textContent = lines.join("\n");
         var sum = document.getElementById("routingSummary");
         if (sum) {
           var byNode = (d.routing_summary && d.routing_summary.by_node) || {};
           var parts = Object.keys(byNode).map(function(node) {
             var st = byNode[node];
             var inner = Object.keys(st).map(function(k) { return k + " " + st[k]; }).join(", ");
             return node + " (" + inner + ")";
           });
           sum.textContent = parts.length ? ("Routing totals — " + parts.join("; ")) : "";
         }
       } catch (err) {
         host.textContent = "Could not load cluster: " + err.message;
       }
     }
     var evbox = document.getElementById("routingEvents");
     if (!evbox) return;
     try {
       var eo = await fetchJSON("/api/routing/events?limit=40");
       var events = eo.data.events || [];
       if (!events.length) { evbox.textContent = "No routing decisions recorded yet."; return; }
       evbox.textContent = events.map(function(e) {
         return (e.created_at || "") + "  " + (e.selected_node || "?")
                + "  [" + (e.status || "") + "]  " + (e.kind || "")
                + (e.requested_model ? "  " + e.requested_model : "")
                + (e.reason ? "  — " + e.reason : "");
       }).join("\n");
     } catch (err) {
       evbox.textContent = "Could not load routing events: " + err.message;
     }
   }

   // Open the admin view on a specific section (the view alone defaults to Model).
   function showAdminSection(name) {
     window._adminTab = name;
     showView("models");
     showAdminTab(name);
   }

   // Admin sub-tabs: show one section of the admin view at a time.
   function showAdminTab(name) {
     window._adminTab = name;
     document.querySelectorAll("#adminPanels .panel").forEach(function(p) {
       p.hidden = (p.dataset.mg !== name);
     });
     document.querySelectorAll("#adminTabs button").forEach(function(b) {
       b.classList.toggle("active", b.getAttribute("data-mg-tab") === name);
     });
   }

   // -------------------------------------------------------- agents ------ #

   var _agentCaps = [];
   var _agentRuntime = {};

   async function loadAgentCaps() {
     try {
       var out = await fetchJSON("/api/agents/capabilities");
       _agentCaps = out.data.capabilities || [];
       _agentRuntime = out.data.runtime || {};
     } catch (err) { _agentCaps = []; _agentRuntime = {}; }
   }

   // Render the capability checkboxes into `host`, pre-checking `selected` keys.
   // Returns a reader for the currently-checked capability keys.
   function renderCaps(host, selected) {
     host.innerHTML = "";
     var sel = {};
     (selected || []).forEach(function(k) { sel[k] = true; });
     var boxes = [];
     if (!_agentCaps.length) {
       // The catalogue failed to load; say so instead of showing an empty grid
       // that looks like "this agent has no capabilities to choose from".
       host.textContent = "Could not load the capability list — refresh to try again.";
       return function() { return (selected || []).slice(); };
     }
     _agentCaps.forEach(function(c) {
       var row = document.createElement("label");
       row.className = "cap-item";
       var cb = document.createElement("input");
       cb.type = "checkbox"; cb.value = c.key; cb.checked = !!sel[c.key];
       var note = "";
       if (c.key === "code_exec" && !(_agentRuntime.allow_shell || _agentRuntime.allow_python)) {
         note = " (server execution is off)";
       }
       if (c.key === "office365" && !_agentRuntime.office365_configured) {
         note = " (not connected)";
       }
       var text = document.createElement("div"); text.className = "cap-text";
       var lab = document.createElement("div"); lab.className = "cap-label";
       lab.textContent = c.label + note;
       var desc = document.createElement("div"); desc.className = "cap-desc";
       desc.textContent = c.description;
       text.appendChild(lab); text.appendChild(desc);
       row.appendChild(cb); row.appendChild(text);
       host.appendChild(row);
       boxes.push(cb);
     });
     return function() {
       return boxes.filter(function(b) { return b.checked; })
                   .map(function(b) { return b.value; });
     };
   }

   async function loadAgents() {
     if (!_agentCaps.length) await loadAgentCaps();
     var host = document.getElementById("agentsList");
     if (!host) return;
     try {
       var out = await fetchJSON("/api/agents");
       var agents = out.data.agents || [];
       host.innerHTML = "";
       if (!agents.length) host.textContent = "No agents yet — create one below.";
       agents.forEach(function(a) {
         var card = document.createElement("div"); card.className = "agent-card";
         var head = document.createElement("div"); head.className = "agent-head";
         var name = document.createElement("input");
         name.type = "text"; name.value = a.name; name.className = "agent-name";
         var sw = document.createElement("label"); sw.className = "switch";
         var enc = document.createElement("input"); enc.type = "checkbox"; enc.checked = a.enabled;
         var sl = document.createElement("span"); sl.className = "slider";
         sw.appendChild(enc); sw.appendChild(sl); sw.title = "Enabled";
         head.appendChild(name); head.appendChild(sw);
         var desc = document.createElement("input");
         desc.type = "text"; desc.value = a.description || ""; desc.placeholder = "description";
         desc.className = "agent-desc";
         var caps = document.createElement("div"); caps.className = "cap-grid";
         var readCaps = renderCaps(caps, a.capabilities);
         var actions = document.createElement("div"); actions.className = "row";
         actions.style.marginTop = "6px";
         var save = document.createElement("button"); save.className = "primary"; save.textContent = "Save";
         save.onclick = function() { saveAgent(a.id, name.value, desc.value, readCaps(), enc.checked); };
         var del = document.createElement("button"); del.textContent = "Delete";
         del.onclick = function() { if (confirm("Delete agent " + a.name + "?")) deleteAgent(a.id); };
         actions.appendChild(save); actions.appendChild(del);
         card.appendChild(head); card.appendChild(desc); card.appendChild(caps); card.appendChild(actions);
         host.appendChild(card);
       });
       var np = document.getElementById("newAgentCaps");
       if (np) window._newAgentReadCaps = renderCaps(np, []);
       var pick = document.getElementById("agentRunPick");
       if (pick) {
         pick.innerHTML = "";
         window._agentRunBoxes = [];
         var enabled = agents.filter(function(a) { return a.enabled; });
         if (!enabled.length) { pick.textContent = "Enable at least one agent to run."; return; }
         enabled.forEach(function(a) {
           var row = document.createElement("label"); row.className = "cap-item";
           var cb = document.createElement("input"); cb.type = "checkbox"; cb.value = a.id;
           var text = document.createElement("div"); text.className = "cap-text";
           var lab = document.createElement("div"); lab.className = "cap-label"; lab.textContent = a.name;
           var d = document.createElement("div"); d.className = "cap-desc";
           // Show the same human labels as the capability grid above, not the
           // internal keys (file_ops, web_api, ...).
           var labels = (a.capabilities || []).map(function(k) {
             for (var i = 0; i < _agentCaps.length; i++) {
               if (_agentCaps[i].key === k) return _agentCaps[i].label;
             }
             return k;
           });
           d.textContent = labels.join(", ") || "no capabilities";
           text.appendChild(lab); text.appendChild(d);
           row.appendChild(cb); row.appendChild(text);
           pick.appendChild(row);
           window._agentRunBoxes.push(cb);
         });
       }
     } catch (err) {
       host.textContent = "Could not load agents: " + err.message;
     }
   }

   async function createAgent() {
     var name = (document.getElementById("newAgentName").value || "").trim();
     if (!name) { alert("Enter a name for the agent."); return; }
     var desc = (document.getElementById("newAgentDesc").value || "").trim();
     var caps = window._newAgentReadCaps ? window._newAgentReadCaps() : [];
     try {
       await fetchJSON("/api/agents", { method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ name: name, description: desc, capabilities: caps }) });
       document.getElementById("newAgentName").value = "";
       document.getElementById("newAgentDesc").value = "";
       loadAgents();
     } catch (err) { alert("Could not create agent: " + err.message); }
   }

   async function saveAgent(id, name, description, capabilities, enabled) {
     try {
       await fetchJSON("/api/agents/" + encodeURIComponent(id), { method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ name: name, description: description,
                                capabilities: capabilities, enabled: enabled }) });
       loadAgents();
     } catch (err) { alert("Could not save agent: " + err.message); }
   }

   async function deleteAgent(id) {
     try {
       await fetchJSON("/api/agents/" + encodeURIComponent(id), { method: "DELETE" });
       loadAgents();
     } catch (err) { alert("Could not delete agent: " + err.message); }
   }

   async function runAgents() {
     var ids = (window._agentRunBoxes || []).filter(function(b) { return b.checked; })
                 .map(function(b) { return b.value; });
     var prompt = (document.getElementById("agentRunPrompt").value || "").trim();
     var out = document.getElementById("agentRunResult");
     if (!ids.length) { alert("Select at least one agent to run."); return; }
     if (!prompt) { alert("Enter a task for the agents."); return; }
     out.textContent = "Running " + ids.length + " agent(s)... this can take a while.";
     try {
       var r = await fetchJSON("/api/agents/run", { method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ prompt: prompt, agent_ids: ids }) });
       var d = r.data || {};
       out.innerHTML = "";
       if (d.skipped) {
         var warn = document.createElement("div");
         warn.className = "hint";
         warn.textContent = d.skipped + " selected agent(s) were not run: at most "
           + (d.max_agents || "a few") + " run together.";
         out.appendChild(warn);
       }
       if (d.merged) {
         var m = document.createElement("div"); m.className = "agent-merged";
         var h = document.createElement("div"); h.className = "cap-label"; h.textContent = "Merged answer";
         var b = document.createElement("div"); b.className = "agent-answer";
         renderMarkdown(d.merged, b);
         m.appendChild(h); m.appendChild(b); out.appendChild(m);
       }
       (d.results || []).forEach(function(res) {
         var box = document.createElement("details"); box.className = "agent-result";
         var sum = document.createElement("summary");
         sum.textContent = res.name + " — " + ((res.tools_used || []).join(", ") || "no tools used");
         var body = document.createElement("div"); body.className = "agent-answer";
         renderMarkdown(res.answer || "(no answer)", body);
         box.appendChild(sum); box.appendChild(body); out.appendChild(box);
       });
       if (!d.merged && !(d.results || []).length) out.textContent = "No results.";
     } catch (err) { out.textContent = "Run failed: " + err.message; }
   }

   wireDropZone();

   // Restore the saved colour scheme before anything renders.
   try {
     // Default to light: it mirrors the texcel.be brand look. The toggle and
     // localStorage still let a user switch to (and keep) the dark variant.
     applyTheme(localStorage.getItem("llm_theme") === "dark" ? "dark" : "light");
   } catch (err) { applyTheme("light"); }

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

   // Resolve auth, show the login screen if required, then load the data that
   // suits the signed-in user's role. Defined in the auth script part.
   authBoot();
 """

__all__ = ["UI_JS_PANELS"]
