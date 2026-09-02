"""Front-end: the models view (catalogue, adapters, dataset, knowledge base, codebase).

Part of the embedded UI, split out of the single HTML_PAGE string for
readability. Assembled once at import time in ui.py, so there is no
runtime cost and no extra file I/O.
"""

from __future__ import annotations

UI_JS_MODELS = r"""   // ------------------------------------------------------------ models ---

   async function loadDatasetStats() {
     var box = document.getElementById("datasetStats");
     if (!box) return;
     try {
       var s = await fetchJSON("/api/dataset/stats");
       var bySource = Object.keys(s.by_source || {})
         .map(function(k) { return k + ": " + s.by_source[k]; }).join(", ") || "none";
       box.textContent =
         "Total examples:       " + s.total + "\\n" +
         "Approved (to imitate): " + s.approved + "\\n" +
         "Rejected (bad answers): " + s.rejected + "\\n" +
         "Corrections:          " + s.corrected + "\\n" +
         "Preference pairs:     " + s.preference_pairs + "\\n" +
         "Reviewed (curated):   " + s.reviewed + "\\n" +
         "By source:            " + bySource;
     } catch (err) {
       box.textContent = "Could not load dataset stats: " + err.message;
     }
   }

   async function exportDataset(format) {
     var reviewed = document.getElementById("exportReviewedOnly");
     var url = "/api/dataset/export?format=" + encodeURIComponent(format)
       + "&reviewed_only=" + (reviewed && reviewed.checked ? "true" : "false");
     try {
       var resp = await fetch(url);
       if (!resp.ok) throw new Error("HTTP " + resp.status);
       var count = resp.headers.get("X-Example-Count") || "?";
       var text = await resp.text();
       var blob = new Blob([text], { type: "application/x-ndjson" });
       var a = document.createElement("a");
       a.href = URL.createObjectURL(blob);
       a.download = "dataset_" + format + ".jsonl";
       document.body.appendChild(a); a.click(); a.remove();
       URL.revokeObjectURL(a.href);
       alert(count + " examples exported as " + format + " (also saved to data/exports/).");
     } catch (err) {
       alert("Export failed: " + err.message);
     }
   }

   async function loadProjectStatus() {
     var box = document.getElementById("projectStatus");
     if (!box) return;
     try {
       var s = await fetchJSON("/api/project/status");
       if (!s.project_dir) {
         box.textContent = "No project set — file tools use the sandbox workspace.\\n"
           + "Set a project directory above to work on a local codebase.";
         var inp = document.getElementById("cfgProjectDir");
         if (inp && !inp.value) inp.value = "";
         return;
       }
       var lines = [];
       lines.push("Directory: " + s.root);
       lines.push("Git repo:  " + (s.is_git_repo ? ("yes (branch " + (s.branch || "?") + ")")
         : "NO — you will have no easy way to review or revert edits"));
       var changed = s.changed_this_session || [];
       lines.push("Changed this session: " + (changed.length ? changed.join(", ") : "none"));
       if (s.is_git_repo) {
         var st = (s.status || "").trim();
         lines.push("");
         lines.push(st ? ("Uncommitted (git status):\\n" + st) : "Working tree clean.");
       }
       box.textContent = lines.join("\\n");
       var hint = document.getElementById("sandboxHint");
       if (hint) {
         var exec = [];
         if (s.allow_shell) exec.push("shell");
         if (s.allow_python) exec.push("python");
         if (exec.length) {
           hint.textContent = "Execution enabled (" + exec.join(" + ")
             + "), confined to this directory with a timeout. The agent can run "
             + (s.test_command ? ("tests (" + s.test_command + "), ") : "")
             + "read the output, and fix the code. Not a security sandbox — it runs "
             + "with your privileges, so use it only on code you trust and review.";
         } else {
           hint.textContent = "Execution disabled. Start the app with --allow-shell "
             + "(and/or --allow-python) to let the agent run and test the code, then "
             + "iterate on failures. Confined to this directory; review with git.";
         }
       }
       var inp2 = document.getElementById("cfgProjectDir");
       if (inp2 && !inp2.value) inp2.value = s.project_dir;
     } catch (err) {
       box.textContent = "Could not load project status: " + err.message;
     }
   }

   async function loadProjectDiff() {
     var pre = document.getElementById("projectDiff");
     if (!pre) return;
     try {
       var d = await fetchJSON("/api/project/diff");
       if (!d.is_git_repo) { pre.style.display = "block"; pre.textContent = "Not a git repository — no diff available."; return; }
       pre.style.display = "block";
       pre.textContent = (d.diff && d.diff.trim()) ? d.diff : "No uncommitted changes.";
     } catch (err) {
       pre.style.display = "block"; pre.textContent = "Could not load diff: " + err.message;
     }
   }

   async function saveProjectDir() {
     var val = (document.getElementById("cfgProjectDir").value || "").trim();
     try {
       await fetchJSON("/api/config", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ project_dir: val })
       });
       alert(val ? ("Project set to " + val) : "Project cleared — using sandbox workspace.");
       loadProjectStatus();
     } catch (err) {
       alert("Could not set project directory: " + err.message);
     }
   }

   async function revertChanges() {
     if (!confirm("Undo this session's edits? Modified files are restored to the "
       + "last commit and files created this session are deleted. This cannot be undone.")) return;
     try {
       var r = await fetchJSON("/api/project/revert", { method: "POST" });
       if (r.data && r.data.error) { alert(r.data.error); return; }
       var d = r.data || r;
       alert("Restored " + (d.restored || []).length + ", removed " + (d.removed || []).length
         + (d.failed && d.failed.length ? (", failed " + d.failed.length) : "") + ".");
       loadProjectStatus();
     } catch (err) {
       alert("Revert failed: " + err.message);
     }
   }

   async function loadDocsStats() {
     var box = document.getElementById("docsStats");
     if (!box) return;
     try {
       var s = await fetchJSON("/api/docs/stats");
       if (!s.enabled) { box.textContent = "Knowledge base unavailable."; return; }
       var lines = ["Documents: " + s.documents, "Passages:  " + s.chunks];
       if (s.search_mode && s.search_mode !== "fts5") {
         lines.push("Search:    built-in ranking (this Python's SQLite has no FTS5)");
       }
       if (s.items && s.items.length) {
         lines.push("");
         lines.push("Indexed:");
         s.items.slice(0, 15).forEach(function(i) {
           lines.push("  " + i.path + "  (" + i.chunks + " passages)");
         });
         if (s.items.length > 15) lines.push("  ... and " + (s.items.length - 15) + " more");
       } else {
         lines.push("");
         lines.push("Nothing indexed yet — answers use the model's own knowledge.");
       }
       box.textContent = lines.join("\n");
       renderScope(s.items || []);
     } catch (err) {
       box.textContent = "Could not load knowledge base stats: " + err.message;
     }
   }

   async function renderScope(items) {
     var host = document.getElementById("docScope");
     if (!host) return;
     var current = [];
     try {
       var sc = await fetchJSON("/api/docs/scope");
       current = ((sc.data || sc).scope || "").split(",").map(function(x) { return x.trim(); })
                   .filter(Boolean);
     } catch (err) { current = []; }
     if (!items.length) { host.textContent = "Nothing indexed yet."; return; }
     host.innerHTML = "";
     items.forEach(function(it) {
       var row = document.createElement("label");
       row.className = "agent-toggle";
       row.style.display = "block";
       var cb = document.createElement("input");
       cb.type = "checkbox";
       cb.value = it.path;
       cb.checked = current.indexOf(it.path) >= 0;
       row.appendChild(cb);
       row.appendChild(document.createTextNode(" " + it.path));
       host.appendChild(row);
     });
   }

   async function applyScope() {
     var boxes = document.querySelectorAll("#docScope input[type=checkbox]");
     var picked = [];
     boxes.forEach(function(b) { if (b.checked) picked.push(b.value); });
     try {
       await fetchJSON("/api/docs/scope?paths=" + encodeURIComponent(picked.join(",")),
                       { method: "POST" });
       alert(picked.length
         ? ("Answers will use only these " + picked.length + " document(s).")
         : "No documents ticked — using the whole knowledge base.");
     } catch (err) {
       alert("Could not set scope: " + err.message);
     }
   }

   async function clearScope() {
     try {
       await fetchJSON("/api/docs/scope?paths=", { method: "POST" });
       loadDocsStats();
       alert("Using the whole knowledge base.");
     } catch (err) {
       alert("Could not clear scope: " + err.message);
     }
   }

   async function indexDocs() {
     var path = (document.getElementById("docsPath").value || "").trim();
     if (!path) { alert("Enter a file or folder path relative to the project."); return; }
     try {
       var r = await fetchJSON("/api/docs/index?path=" + encodeURIComponent(path), { method: "POST" });
       var d = r.data || r;
       if (d.error) { alert(d.error); return; }
       alert(d.result || "Indexed.");
       loadDocsStats();
     } catch (err) {
       alert("Indexing failed: " + err.message);
     }
   }

   async function searchDocs() {
     var q = (document.getElementById("docsQuery").value || "").trim();
     var pre = document.getElementById("docsResult");
     if (!q) { alert("Type a question to test retrieval."); return; }
     try {
       var r = await fetchJSON("/api/docs/search?q=" + encodeURIComponent(q));
       var hits = (r.data || r).hits || [];
       pre.style.display = "block";
       pre.textContent = hits.length
         ? hits.map(function(h) { return "[" + h.path + "]\n" + h.chunk; }).join("\n\n")
         : "No matching passages.";
     } catch (err) {
       pre.style.display = "block";
       pre.textContent = "Search failed: " + err.message;
     }
   }

   async function clearDocs() {
     if (!confirm("Remove every indexed document? Your actual files are not touched.")) return;
     try {
       var r = await fetchJSON("/api/docs/clear", { method: "POST" });
       alert("Cleared " + ((r.data || r).cleared || 0) + " document(s).");
       loadDocsStats();
     } catch (err) {
       alert("Clear failed: " + err.message);
     }
   }

   async function indexUrl() {
     var url = (document.getElementById("docsUrl").value || "").trim();
     if (!url) { alert("Enter a URL to fetch and index."); return; }
     try {
       var r = await fetchJSON("/api/docs/index_url?url=" + encodeURIComponent(url), { method: "POST" });
       var d = r.data || r;
       if (d.error) { alert(d.error); return; }
       alert(d.result || "Indexed.");
       loadDocsStats();
     } catch (err) {
       alert("Indexing failed: " + err.message);
     }
   }

   async function regenerateLast() {
     if (busy) { alert("Wait for the current answer to finish."); return; }
     try {
       var r = await fetchJSON("/api/conversation/" + encodeURIComponent(conversationId) + "/regenerate",
                               { method: "POST" });
       var d = r.data || r;
       if (d.error) { alert(d.error); return; }
       // Drop the last rendered answer, then re-send the same question.
       var nodes = chat.querySelectorAll(".turn, .msg.assistant");
       if (nodes.length) nodes[nodes.length - 1].remove();
       input.value = d.prompt;
       send();
     } catch (err) {
       alert("Could not regenerate: " + err.message);
     }
   }

   function exportChat() {
     window.location = "/api/conversation/" + encodeURIComponent(conversationId) + "/export?format=markdown";
   }
"""

__all__ = ["UI_JS_MODELS"]
