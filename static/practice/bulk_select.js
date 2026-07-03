(function () {
  const forms = document.querySelectorAll("form[data-bulk-form]");

  forms.forEach(function (form) {
    const selectAll = form.querySelector("[data-bulk-select-all]");
    const countLabel = form.querySelector("[data-bulk-count]");
    const submitButton = form.querySelector("[data-bulk-submit]");
    const sections = Array.from(document.querySelectorAll('[data-bulk-section="' + form.id + '"]'));

    function itemBoxes() {
      return Array.from(form.elements).filter(function (el) {
        return el.matches('input[type="checkbox"][name="selected"]');
      });
    }

    function sectionBoxes(section) {
      return itemBoxes().filter(function (box) {
        return section.contains(box);
      });
    }

    function syncToggle(toggle, boxes) {
      const checked = boxes.filter(function (box) { return box.checked; });
      toggle.checked = boxes.length > 0 && checked.length === boxes.length;
      toggle.indeterminate = checked.length > 0 && checked.length < boxes.length;
    }

    function refresh() {
      const boxes = itemBoxes();
      const checked = boxes.filter(function (box) { return box.checked; });
      form.hidden = boxes.length === 0;
      if (countLabel) {
        countLabel.textContent = checked.length + " selected";
      }
      if (submitButton) {
        submitButton.disabled = checked.length === 0;
      }
      if (selectAll) {
        syncToggle(selectAll, boxes);
      }
      sections.forEach(function (section) {
        const boxesInSection = sectionBoxes(section);
        const tools = section.querySelector("[data-bulk-section-tools]");
        if (tools) {
          tools.hidden = boxesInSection.length === 0;
        }
        const toggle = section.querySelector("[data-bulk-section-select]");
        if (toggle) {
          syncToggle(toggle, boxesInSection);
        }
      });
      boxes.forEach(function (box) {
        const row = box.closest("[data-bulk-row]");
        if (row) {
          row.classList.toggle("is-bulk-selected", box.checked);
        }
      });
    }

    function submitForm() {
      if (typeof form.requestSubmit === "function") {
        form.requestSubmit();
      } else if (submitButton) {
        submitButton.click();
      }
    }

    if (selectAll) {
      selectAll.addEventListener("change", function () {
        itemBoxes().forEach(function (box) {
          box.checked = selectAll.checked;
        });
        refresh();
      });
    }

    sections.forEach(function (section) {
      const toggle = section.querySelector("[data-bulk-section-select]");
      if (toggle) {
        toggle.addEventListener("change", function () {
          sectionBoxes(section).forEach(function (box) {
            box.checked = toggle.checked;
          });
          refresh();
        });
      }
      const deleteButton = section.querySelector("[data-bulk-section-delete]");
      if (deleteButton) {
        deleteButton.addEventListener("click", function () {
          const boxesInSection = sectionBoxes(section);
          if (boxesInSection.length === 0) {
            return;
          }
          itemBoxes().forEach(function (box) {
            box.checked = boxesInSection.indexOf(box) !== -1;
          });
          refresh();
          submitForm();
        });
      }
    });

    document.addEventListener("change", function (event) {
      if (event.target instanceof HTMLInputElement && event.target.form === form && event.target.name === "selected") {
        refresh();
      }
    });

    form.addEventListener("submit", function (event) {
      const checked = itemBoxes().filter(function (box) { return box.checked; });
      if (checked.length === 0) {
        event.preventDefault();
        return;
      }
      const template = form.dataset.bulkConfirm || "Delete the selected items?";
      const message = template.replace("{count}", String(checked.length));
      if (!window.confirm(message)) {
        event.preventDefault();
      }
    });

    refresh();
  });
})();
