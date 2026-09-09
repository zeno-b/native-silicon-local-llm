"""Front-end auth: login gate, role gating, logout, and the admin import panel.

Part of the single embedded <script>. It owns application boot: authBoot() runs
last (called from the panels boot), decides whether to show the login overlay or
the app, applies the signed-in user's role to the DOM (so admin-only chrome is
hidden for non-admins), and then loads the data appropriate to that role.

Session auth is cookie-based, so every fetch/EventSource/download already carries
credentials; there is nothing to add to individual requests here.
"""

from __future__ import annotations

UI_JS_AUTH = r"""
   // ---------------------------------------------------------------- auth ---
   async function authBoot() {
     var info = {};
     try {
       var out = await fetchJSON("/api/auth/me");
       info = (out && out.data) || {};
     } catch (err) { info = {}; }
     window.AUTH_ENABLED = !!info.auth_enabled;
     window.OIDC_ENABLED = !!info.oidc_enabled;
     if (info.auth_enabled && !info.authenticated) {
       showLogin(info);
       return;
     }
     var user = info.user || { role: "admin", username: "local" };
     // Apply the role BEFORE revealing the app, so admin-only chrome does not
     // pop in a frame after the rest of the UI.
     applyRole(user);
     hideLogin();
     runDataBoot(user.role === "admin");
     if (user.role === "admin") loadImports();
   }

   function showLogin(info) {
     // Leave the boot state (nothing shown) for the login box, in one step.
     document.body.classList.remove("booting");
     document.body.classList.add("locked");
     var oidc = document.getElementById("oidcBtn");
     if (oidc) oidc.style.display = info && info.oidc_enabled ? "" : "none";
     var u = document.getElementById("loginUser");
     if (u) u.focus();
   }

   function hideLogin() {
     document.body.classList.remove("booting");
     document.body.classList.remove("locked");
   }

   function applyRole(user) {
     document.body.dataset.role = user.role || "user";
     var chip = document.getElementById("userChip");
     if (chip) chip.textContent = user.username || "local";
     var info = document.getElementById("acctInfo");
     if (info) {
       info.innerHTML = "";
       var line = document.createElement("div");
       var who = document.createElement("strong");
       who.textContent = user.username || "local";
       line.appendChild(who);
       line.appendChild(document.createTextNode(" · " + (user.role || "user")));
       info.appendChild(line);
       if (!window.AUTH_ENABLED) {
         var note = document.createElement("div");
         note.textContent = "Single-user mode — sign-in is off. Start the server with "
           + "AUTH_ENABLED=1 to turn on accounts, roles and logout.";
         info.appendChild(note);
       }
     }
     var logout = document.getElementById("logoutBtn");
     // With auth disabled there is no session to end, so hide the logout action
     // (the account menu still explains the single-user state above).
     if (logout) logout.style.display = window.AUTH_ENABLED ? "" : "none";
   }

   function toggleAcct() {
     var pop = document.getElementById("acctPop");
     if (pop) pop.classList.toggle("hidden");
   }

   // Reveal/hide a password field, flipping the little show/hide button's label.
   function togglePw(id, btn) {
     var el = document.getElementById(id);
     if (!el) return;
     var reveal = el.type === "password";
     el.type = reveal ? "text" : "password";
     if (btn) btn.textContent = reveal ? "hide" : "show";
   }

   // Close the account menu when clicking anywhere outside it.
   document.addEventListener("click", function(e) {
     var menu = document.getElementById("acctMenu");
     var pop = document.getElementById("acctPop");
     if (!menu || !pop || pop.classList.contains("hidden")) return;
     if (!e.target.closest || !e.target.closest("#acctMenu")) pop.classList.add("hidden");
   });

   function runDataBoot(isAdmin) {
     refreshHealth();
     if (window._healthTimer) clearInterval(window._healthTimer);
     window._healthTimer = setInterval(refreshHealth, 3000);
     loadMemory();
     if (isAdmin) { loadConfig(); loadPerf(); }
   }

   async function doLogin() {
     var status = document.getElementById("loginError");
     var user = document.getElementById("loginUser").value.trim();
     var pass = document.getElementById("loginPass").value;
     if (!user || !pass) { if (status) status.textContent = "Enter a username and password."; return; }
     if (status) status.textContent = "";
     try {
       var res = await fetch("/api/auth/login", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ username: user, password: pass })
       });
       if (!res.ok) {
         if (status) status.textContent = "Invalid username or password.";
         return;
       }
       // Reload so the whole app re-initialises cleanly with the new session.
       window.location.reload();
     } catch (err) {
       if (status) status.textContent = "Login error: " + err.message;
     }
   }

   // A protected request returned 401 while we thought we were signed in: the
   // session expired, was revoked by an admin (disable / role change / password
   // reset all drop sessions), or was ended in another tab. Return to the login
   // screen instead of leaving a dead app on screen with silently failing calls.
   function onUnauthorized() {
     if (!window.AUTH_ENABLED) return;                       // auth off: nothing to do
     if (document.body.classList.contains("locked")) return; // already on login
     if (window._healthTimer) { clearInterval(window._healthTimer); window._healthTimer = null; }
     var err = document.getElementById("loginError");
     if (err) err.textContent = "Your session ended. Please sign in again.";
     showLogin({ oidc_enabled: window.OIDC_ENABLED });
   }

   function loginKey(e) { if (e.key === "Enter") { e.preventDefault(); doLogin(); } }

   function oidcLogin() { window.location = "/api/auth/oidc/login"; }

   async function doLogout() {
     try { await fetch("/api/auth/logout", { method: "POST" }); } catch (err) {}
     window.location.reload();
   }

   // -------------------------------------------------- Claude import panel ---
   function importSummary(it) {
     var counts = it.counts;
     if (typeof counts === "string") { try { counts = JSON.parse(counts); } catch (e) { counts = null; } }
     if (!counts) return "";
     return counts.conversations + " conversations, " + counts.messages + " messages, " +
            counts.knowledge_docs + " knowledge docs, " + counts.skills + " skills" +
            (counts.duplicates ? ", " + counts.duplicates + " duplicates skipped" : "");
   }

   async function loadImports() {
     var box = document.getElementById("importList");
     if (!box) return;
     try {
       var out = await fetchJSON("/api/imports");
       var items = (out.data && out.data.imports) || [];
       box.textContent = "";
       if (!items.length) { box.textContent = "No imports yet."; return; }
       var active = false;
       items.forEach(function(it) {
         var row = document.createElement("div");
         row.className = "import-row";
         var head = document.createElement("div");
         head.className = "import-head";
         var title = document.createElement("strong");
         title.textContent = it.filename || "(export)";
         var badge = document.createElement("span");
         badge.className = "import-badge import-" + (it.status || "pending");
         badge.textContent = (it.status || "pending") + " " + (it.progress || 0) + "%";
         head.appendChild(title); head.appendChild(badge);
         row.appendChild(head);
         var detail = document.createElement("div");
         detail.className = "import-detail";
         detail.textContent = importSummary(it) || (it.error ? ("error: " + it.error) : "");
         row.appendChild(detail);
         var actions = document.createElement("div");
         actions.className = "import-actions";
         if (it.status === "failed") {
           var retry = document.createElement("button");
           retry.textContent = "Retry";
           retry.onclick = function() { retryImportJob(it.id); };
           actions.appendChild(retry);
         }
         var del = document.createElement("button");
         del.textContent = "Remove";
         del.onclick = function() { deleteImportJob(it.id); };
         actions.appendChild(del);
         row.appendChild(actions);
         box.appendChild(row);
         if (it.status !== "completed" && it.status !== "failed") active = true;
       });
       // Poll while any import is still running.
       if (active) setTimeout(loadImports, 1500);
     } catch (err) {
       box.textContent = "Could not load imports: " + err.message;
     }
   }

   async function startImport() {
     var inp = document.getElementById("importFile");
     var status = document.getElementById("importStatus");
     if (!inp || !inp.files || !inp.files[0]) {
       if (status) status.textContent = "Choose a chat export .zip first.";
       return;
     }
     var file = inp.files[0];
     if (status) status.textContent = "Uploading " + file.name + " (" +
       Math.round(file.size / 1024) + " KB)...";
     try {
       var res = await fetch("/api/import", {
         method: "POST", body: file, headers: { "X-Filename": file.name }
       });
       var data = {};
       try { data = await res.json(); } catch (e) {}
       if (!res.ok) {
         if (status) status.textContent = "Import failed: " + (data.error || ("HTTP " + res.status));
         return;
       }
       if (status) status.textContent = "Processing import " + (data.import_id || "") + "...";
       inp.value = "";
       loadImports();
     } catch (err) {
       if (status) status.textContent = "Import error: " + err.message;
     }
   }

   async function retryImportJob(id) {
     try { await fetch("/api/imports/" + encodeURIComponent(id) + "/retry", { method: "POST" }); }
     catch (err) {}
     loadImports();
   }

   async function deleteImportJob(id) {
     if (!window.confirm("Remove this import and the conversations and knowledge it created?")) return;
     try { await fetch("/api/imports/" + encodeURIComponent(id), { method: "DELETE" }); }
     catch (err) {}
     loadImports();
   }
"""

__all__ = ["UI_JS_AUTH"]
