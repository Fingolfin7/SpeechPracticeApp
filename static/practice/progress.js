(function () {
  const dataElement = document.getElementById("progress-data");
  if (!dataElement) {
    return;
  }

  const points = JSON.parse(dataElement.textContent || "[]");

  function drawChart(canvas, metric, maxValue) {
    const ctx = canvas.getContext("2d");
    const ratio = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width * ratio));
    const height = Math.max(1, Math.floor(canvas.height * ratio));
    canvas.width = width;
    canvas.height = height;
    ctx.scale(ratio, ratio);

    const cssWidth = width / ratio;
    const cssHeight = height / ratio;
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    ctx.fillStyle = "#fffaf0";
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

    const margin = { top: 18, right: 16, bottom: 32, left: 42 };
    const plotWidth = cssWidth - margin.left - margin.right;
    const plotHeight = cssHeight - margin.top - margin.bottom;
    const usablePoints = points.filter((point) => Number.isFinite(Number(point[metric])));
    const localMax = maxValue || Math.max(...values, 1);
    const localMin = 0;

    ctx.strokeStyle = "#ded0bb";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i += 1) {
      const y = margin.top + (plotHeight * i) / 4;
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + plotWidth, y);
      ctx.stroke();
    }

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

    ctx.strokeStyle = "#0f766e";
    ctx.lineWidth = 2;
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
      ctx.fillStyle = "#b84421";
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = "#6f665b";
    ctx.font = "12px Candara, Segoe UI, sans-serif";
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

  drawAll();
  window.addEventListener("resize", drawAll);
})();
