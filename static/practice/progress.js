(function () {
  const dataElement = document.getElementById("progress-data");
  if (!dataElement) {
    return;
  }

  const points = JSON.parse(dataElement.textContent || "[]");
  const metricConfig = {
    score: { max: 5, decimals: 2, suffix: "", goodDirection: 1, color: "#0f766e" },
    wer: { max: 1, decimals: 2, suffix: "", goodDirection: -1, color: "#b84421" },
    clarity: { max: 1, decimals: 2, suffix: "", goodDirection: 1, color: "#285f9f" },
  };

  function metricValue(point, metric) {
    const value = Number(point && point[metric]);
    return Number.isFinite(value) ? value : null;
  }

  function formatValue(value, metric) {
    if (!Number.isFinite(value)) {
      return "-";
    }
    const config = metricConfig[metric] || metricConfig.score;
    return value.toFixed(config.decimals) + config.suffix;
  }

  function updateStatCards() {
    Object.keys(metricConfig).forEach((metric) => {
      const card = document.querySelector(`[data-progress-stat="${metric}"]`);
      if (!card) {
        return;
      }
      const values = points.map((point) => metricValue(point, metric)).filter((value) => value !== null);
      const strong = card.querySelector("strong");
      const small = card.querySelector("small");
      if (!values.length) {
        strong.textContent = "-";
        small.textContent = "No data";
        return;
      }
      const latest = values[values.length - 1];
      const first = values[0];
      const delta = latest - first;
      const improved = delta * metricConfig[metric].goodDirection >= 0;
      strong.textContent = formatValue(latest, metric);
      small.textContent = `${improved ? "improved" : "shifted"} ${Math.abs(delta).toFixed(2)} across range`;
      card.classList.toggle("is-improved", improved);
      card.classList.toggle("is-regressed", !improved);
    });
  }

  function drawChart(canvas, metric, maxValue) {
    const ctx = canvas.getContext("2d");
    const ratio = window.devicePixelRatio || 1;
    const cssWidth = Math.max(320, Math.floor(canvas.clientWidth || canvas.getBoundingClientRect().width || 320));
    const cssHeight = Math.max(220, Math.floor(canvas.clientHeight || Number(canvas.getAttribute("height")) || 260));
    const width = Math.floor(cssWidth * ratio);
    const height = Math.floor(cssHeight * ratio);
    canvas.width = width;
    canvas.height = height;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

    ctx.clearRect(0, 0, cssWidth, cssHeight);
    const gradient = ctx.createLinearGradient(0, 0, cssWidth, cssHeight);
    gradient.addColorStop(0, "#fffaf0");
    gradient.addColorStop(1, "#f4eadb");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, cssWidth, cssHeight);

    const values = points
      .map((point) => Number(point[metric]))
      .filter((value) => Number.isFinite(value));

    if (!values.length) {
      ctx.fillStyle = "#6f665b";
      ctx.font = "15px Candara, Segoe UI, sans-serif";
      ctx.fillText("No data for this filter", 18, cssHeight / 2);
      return;
    }

    const margin = { top: 22, right: 24, bottom: 38, left: 54 };
    const plotWidth = cssWidth - margin.left - margin.right;
    const plotHeight = cssHeight - margin.top - margin.bottom;
    const usablePoints = points.filter((point) => Number.isFinite(Number(point[metric])));
    const localMax = maxValue || Math.max(...values, 1);
    const localMin = 0;

    ctx.strokeStyle = "rgba(112, 93, 71, 0.18)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i += 1) {
      const y = margin.top + (plotHeight * i) / 4;
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + plotWidth, y);
      ctx.stroke();
    }
    ctx.strokeStyle = "rgba(36, 31, 26, 0.32)";
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + plotHeight);
    ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
    ctx.stroke();

    function xFor(index) {
      if (usablePoints.length === 1) {
        return margin.left + plotWidth / 2;
      }
      return margin.left + (plotWidth * index) / (usablePoints.length - 1);
    }

    function yFor(value) {
      const normalized = (value - localMin) / (localMax - localMin || 1);
      return margin.top + plotHeight - normalized * plotHeight;
    }

    const color = (metricConfig[metric] || metricConfig.score).color;
    const areaGradient = ctx.createLinearGradient(0, margin.top, 0, margin.top + plotHeight);
    areaGradient.addColorStop(0, color + "44");
    areaGradient.addColorStop(1, color + "00");
    ctx.beginPath();
    usablePoints.forEach((point, index) => {
      const x = xFor(index);
      const y = yFor(Number(point[metric]));
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.lineTo(xFor(usablePoints.length - 1), margin.top + plotHeight);
    ctx.lineTo(xFor(0), margin.top + plotHeight);
    ctx.closePath();
    ctx.fillStyle = areaGradient;
    ctx.fill();

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    usablePoints.forEach((point, index) => {
      const x = xFor(index);
      const y = yFor(Number(point[metric]));
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    usablePoints.forEach((point, index) => {
      const x = xFor(index);
      const y = yFor(Number(point[metric]));
      ctx.fillStyle = "#fffaf0";
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#b84421";
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    ctx.fillStyle = "#6f665b";
    ctx.font = "12px Candara, Segoe UI, sans-serif";
    ctx.fillText(formatValue(localMax, metric), 12, margin.top + 4);
    ctx.fillText("0", 28, margin.top + plotHeight + 4);
    if (usablePoints[0]) {
      ctx.fillText(usablePoints[0].label, margin.left, cssHeight - 10);
    }
    if (usablePoints.length > 1) {
      const label = usablePoints[usablePoints.length - 1].label;
      const labelWidth = ctx.measureText(label).width;
      ctx.fillText(label, cssWidth - margin.right - labelWidth, cssHeight - 10);
    }
  }

  function drawAll() {
    document.querySelectorAll("[data-progress-chart]").forEach((canvas) => {
      const metric = canvas.dataset.progressChart;
      const max = metric === "score" ? 5 : 1;
      drawChart(canvas, metric, max);
    });
  }

  updateStatCards();
  drawAll();
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(drawAll);
    document.querySelectorAll("[data-progress-chart]").forEach((canvas) => observer.observe(canvas));
  } else {
    window.addEventListener("resize", drawAll);
  }
})();
