"""Front-end: view switching.

Part of the embedded UI, split out of the single HTML_PAGE string for
readability. Assembled once at import time in ui.py, so there is no
runtime cost and no extra file I/O.
"""

from __future__ import annotations

UI_JS_VIEWS = r"""   // ------------------------------------------------------------- views ---

   var currentView = "chat";
   var selectedTask = null;
   var runSource = null;
   var currentRunId = null;
   var tasksTimer = null;
   var modelLogTimer = null;

   function showView(name) {
     currentView = name;
     ["chat", "tasks", "history", "models"].forEach(function(view) {
       var el = document.getElementById(view + "View");
       if (el) el.classList.toggle("hidden", view !== name);
       var nav = document.getElementById("nav" + view.charAt(0).toUpperCase() + view.slice(1));
       if (nav) nav.classList.toggle("active", view === name);
     });
     document.getElementById("composer").style.display = name === "chat" ? "flex" : "none";

     if (tasksTimer) { clearInterval(tasksTimer); tasksTimer = null; }
     if (modelLogTimer) { clearInterval(modelLogTimer); modelLogTimer = null; }

     if (name === "tasks") {
       loadTasks();
       tasksTimer = setInterval(loadTasks, 3000);
     } else if (name === "history") {
       loadHistory();
     } else if (name === "models") {
       loadModels();
       loadModelLog();
       modelLogTimer = setInterval(loadModelLog, 4000);
     }
   }
"""

__all__ = ["UI_JS_VIEWS"]
