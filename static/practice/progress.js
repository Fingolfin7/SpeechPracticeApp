(function () {
  const dataElement = document.getElementById("progress-data");
  if (!dataElement) {
    return;
  }

  const points = JSON.parse(dataElement.textContent || "[]");
  // Colors mirror the app.css tokens (grove, clay, gold).
  const metricConfig = {
    clarity: {
      color: "#3e7a5c",
      format: (value) => `${Math.round(value * 100)}%`,
      // Clarity lives in a narrow band near the top, so zoom the axis to it.
      range: (values) => [Math.min(0.8, Math.floor(Math.min(...values) * 20) / 20), 1],
      lines: [
        { at: 0.95, label: "95% excellent" },
        { at: 0.9, label: "90% good" },
      ],
    },
    artic_rate: {
      color: "#e8703a",
      format: (value) => `${Math.round(value)}`,
      range: (values) => [
        Math.max(0, Math.floor(Math.min(80, ...values) / 20) * 20),
        Math.ceil(Math.max(200, ...values) / 20) * 20,
      ],
      band: [120, 160],
    },
  };

  function rollingAverage(values, size) {
    return values.map((_value, index) => {
      const window = values.slice(Math.max(0, index - size + 1), index + 1);
      return window.reduce((sum, value) => sum + value, 0) / window.length;
    });
  }

  function drawChart(canvas, metric) {
    const config = metricConfig[metric];
    if (!config) {
      return;
    }
    const ctx = canvas.getContext("2d");
    const ratio = window.devicePixelRatio || 1;
    const cssWidth = Math.max(320, Math.floor(canvas.clientWidth || canvas.getBoundingClientRect().width || 320));
    const cssHeight = Math.max(200, Math.floor(Number(canvas.getAttribute("height")) || 260));
    canvas.width = Math.floor(cssWidth * ratio);
    canvas.height = Math.floor(cssHeight * ratio);
    canvas.style.height = `${cssHeight}px`;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    ctx.fillStyle = "#fffdf9";
    ctx.fillRect(0, 0, cssWidth, cssHeight);

    const usable = points.filter((point) => Number.isFinite(Number(point[metric])) && Number(point[metric]) > 0);
    if (!usable.length) {
      ctx.fillStyle = "#6b5e52";
      ctx.font = "15px Inter, Segoe UI, sans-serif";
      ctx.fillText("No takes in this range", 18, cssHeight / 2);
      return;
    }
    const values = usable.map((point) => Number(point[metric]));
    const [low, high] = config.range(values);

    const margin = { top: 18, right: 20, bottom: 34, left: 48 };
    const plotWidth = cssWidth - margin.left - margin.right;
    const plotHeight = cssHeight - margin.top - margin.bottom;
    const xFor = (index) =>
      usable.length === 1 ? margin.left + plotWidth / 2 : margin.left + (plotWidth * index) / (usable.length - 1);
    const yFor = (value) => {
      const clamped = Math.max(low, Math.min(high, value));
      return margin.top + plotHeight - ((clamped - low) / (high - low || 1)) * plotHeight;
    };

    ctx.font = "12px Inter, Segoe UI, sans-serif";
    if (config.band) {
      const top = yFor(config.band[1]);
      ctx.fillStyle = "rgba(62, 122, 92, 0.1)";
      ctx.fillRect(margin.left, top, plotWidth, yFor(config.band[0]) - top);
    }

    // Axis labels at the bottom and top of the zoomed range.
    ctx.fillStyle = "#6b5e52";
    ctx.fillText(config.format(high), 8, margin.top + 4);
    ctx.fillText(config.format(low), 8, margin.top + plotHeight + 4);

    (config.lines || []).forEach((line) => {
      if (line.at <= low || line.at >= high) {
        return;
      }
      const y = yFor(line.at);
      ctx.strokeStyle = "rgba(112, 93, 71, 0.35)";
      ctx.setLineDash([4, 5]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + plotWidth, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#6b5e52";
      ctx.fillText(line.label, margin.left + 6, y - 5);
    });

    ctx.strokeStyle = "rgba(36, 31, 26, 0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top + plotHeight);
    ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
    ctx.stroke();

    // Individual takes: small, quiet dots.
    usable.forEach((point, index) => {
      ctx.fillStyle = `${config.color}66`;
      ctx.beginPath();
      ctx.arc(xFor(index), yFor(values[index]), 3.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Rolling average: the line to read for the trend.
    if (usable.length >= 2) {
      const smooth = rollingAverage(values, 5);
      ctx.strokeStyle = config.color;
      ctx.lineWidth = 3;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      smooth.forEach((value, index) => {
        const x = xFor(index);
        const y = yFor(value);
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }

    ctx.fillStyle = "#6b5e52";
    ctx.fillText(usable[0].label, margin.left, cssHeight - 10);
    if (usable.length > 1) {
      const label = usable[usable.length - 1].label;
      ctx.fillText(label, cssWidth - margin.right - ctx.measureText(label).width, cssHeight - 10);
    }
  }

  function drawAll() {
    document.querySelectorAll("[data-progress-chart]").forEach((canvas) => {
      drawChart(canvas, canvas.dataset.progressChart);
    });
  }

  drawAll();
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(drawAll);
    document.querySelectorAll("[data-progress-chart]").forEach((canvas) => observer.observe(canvas));
  } else {
    window.addEventListener("resize", drawAll);
  }
})();
