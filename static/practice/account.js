(function () {
  const autumnSettings = document.querySelector("[data-autumn-settings]");
  if (!autumnSettings) {
    return;
  }

  const projectField = autumnSettings.querySelector("[name='autumn_project']");
  const subprojectsBox = autumnSettings.querySelector("[data-autumn-subprojects]");
  const hiddenText = autumnSettings.querySelector("[name='autumn_subprojects_text']");
  const subprojectsUrl = autumnSettings.dataset.subprojectsUrl;
  if (!projectField || !subprojectsBox || !subprojectsUrl) {
    return;
  }

  function selectedSubprojects() {
    return Array.from(subprojectsBox.querySelectorAll("input[name='autumn_subprojects']:checked"))
      .map((input) => input.value)
      .filter(Boolean);
  }

  function syncHiddenText() {
    if (hiddenText) {
      hiddenText.value = selectedSubprojects().join(", ");
    }
  }

  function renderSubprojects(names, selectedNames) {
    let choices = subprojectsBox.querySelector("#id_autumn_subprojects");
    if (!choices) {
      choices = document.createElement("div");
      choices.id = "id_autumn_subprojects";
      subprojectsBox.appendChild(choices);
    }
    choices.replaceChildren();
    const selected = new Set(selectedNames || []);
    if (!names.length) {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "No subprojects found for this project.";
      choices.appendChild(empty);
      syncHiddenText();
      return;
    }
    names.forEach((name, index) => {
      const id = `id_autumn_subprojects_${index}`;
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.name = "autumn_subprojects";
      input.value = name;
      input.id = id;
      input.checked = selected.has(name);
      input.addEventListener("change", syncHiddenText);
      label.setAttribute("for", id);
      label.appendChild(input);
      label.append(document.createTextNode(name));
      choices.appendChild(label);
    });
    syncHiddenText();
  }

  subprojectsBox.addEventListener("change", function (event) {
    if (event.target && event.target.name === "autumn_subprojects") {
      syncHiddenText();
    }
  });

  projectField.addEventListener("change", async function () {
    const project = projectField.value.trim();
    const currentSelection = selectedSubprojects();
    if (!project) {
      renderSubprojects([], []);
      return;
    }
    subprojectsBox.classList.add("is-loading");
    try {
      const url = new URL(subprojectsUrl, window.location.origin);
      url.searchParams.set("project", project);
      const response = await fetch(url.toString(), {
        headers: { Accept: "application/json" },
        credentials: "same-origin",
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || "Subprojects unavailable.");
      }
      renderSubprojects(payload.subprojects || [], currentSelection);
    } catch (error) {
      renderSubprojects([], []);
    } finally {
      subprojectsBox.classList.remove("is-loading");
    }
  });

  syncHiddenText();
})();
