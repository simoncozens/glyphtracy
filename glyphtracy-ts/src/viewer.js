import { Vectorizer } from "./index.ts";

const state = {
  debugData: null,
  imageUrl: null,
  imageRaster: null,
  contourIndex: null,
  stage: null,
};

const contourSelect = document.getElementById("contour-select");
const stageSelect = document.getElementById("stage-select");
const statusEl = document.getElementById("status");
const summaryEl = document.getElementById("summary");
const debugSvg = document.getElementById("debug-svg");
const finalSvg = document.getElementById("final-svg");
const sourceImage = document.getElementById("source-image");
const imageInput = document.getElementById("image-input");
const downloadSvgButton = document.getElementById("download-svg");
const paramControls = document.getElementById("param-controls");
const resetParamsButton = document.getElementById("reset-params");
const hoverTooltip = document.getElementById("hover-tooltip");

const PARAM_DEFS = [
  {
    key: "sharpThresholdDegrees",
    title: "Sharp Threshold",
    description: "Corner detection threshold in degrees.",
    min: 5,
    max: 120,
    step: 1,
    defaultValue: 30,
    format: (value) => `${value.toFixed(0)} deg`,
  },
  {
    key: "pixelTolerance",
    title: "Pixel Tolerance",
    description:
      "Straight-line deviation tolerance before a span becomes a curve.",
    min: 0.1,
    max: 8,
    step: 0.1,
    defaultValue: 1.5,
    format: (value) => value.toFixed(1),
  },
  {
    key: "extremaMinIndexGap",
    title: "Extrema Index Gap",
    description: "Minimum contour index distance between extrema nodes.",
    min: 1,
    max: 24,
    step: 1,
    defaultValue: 3,
    format: (value) => value.toFixed(0),
  },
  {
    key: "extremaMinSpanPoints",
    title: "Extrema Span Points",
    description:
      "Smallest open span size considered during iterative extrema splitting.",
    min: 2,
    max: 48,
    step: 1,
    defaultValue: 8,
    format: (value) => value.toFixed(0),
  },
  {
    key: "extremaMaxIterations",
    title: "Extrema Iterations",
    description: "Maximum iterations for recursive extrema discovery.",
    min: 1,
    max: 20,
    step: 1,
    defaultValue: 8,
    format: (value) => value.toFixed(0),
  },
  {
    key: "fitTolerance",
    title: "Fit Tolerance",
    description:
      "Maximum allowed fitting error before a curve is split and refit.",
    min: 0.5,
    max: 16,
    step: 0.1,
    defaultValue: 5,
    format: (value) => value.toFixed(1),
  },
  {
    key: "nodeBalanceWeight",
    title: "Node Balance",
    description:
      "Bias toward equal incoming and outgoing handles around a node.",
    min: 0,
    max: 4,
    step: 0.05,
    defaultValue: 0.8,
    format: (value) => value.toFixed(2),
  },
  {
    key: "segmentBalanceWeight",
    title: "Segment Balance",
    description:
      "Bias toward similar handle lengths across each fitted segment.",
    min: 0,
    max: 4,
    step: 0.05,
    defaultValue: 0.8,
    format: (value) => value.toFixed(2),
  },
  {
    key: "g2Weight",
    title: "G2 Weight",
    description:
      "Strength of curvature continuity preference at eligible nodes.",
    min: 0,
    max: 4,
    step: 0.05,
    defaultValue: 1,
    format: (value) => value.toFixed(2),
  },
  {
    key: "handleShrinkWeight",
    title: "Handle Shrink",
    description:
      "Regularization that discourages excessively long control handles.",
    min: 0,
    max: 1,
    step: 0.01,
    defaultValue: 0.05,
    format: (value) => value.toFixed(2),
  },
  {
    key: "maxSplitDepth",
    title: "Max Split Depth",
    description: "Maximum recursive refit depth for difficult spans.",
    min: 0,
    max: 10,
    step: 1,
    defaultValue: 5,
    format: (value) => value.toFixed(0),
  },
  {
    key: "resolutionScale",
    title: "Resolution Scale",
    description:
      "Override for scaling resolution-dependent tolerances. Set to 0 to use auto.",
    min: 0,
    max: 4,
    step: 0.05,
    defaultValue: 0,
    format: (value) => (value === 0 ? "auto" : value.toFixed(2)),
  },
];

const paramInputs = new Map();

state.vectorizerParams = Object.fromEntries(
  PARAM_DEFS.map((param) => [param.key, param.defaultValue]),
);
state.currentImageLabel = null;
state.rerunTimer = null;

function getVectorizerOptions() {
  const params = state.vectorizerParams;
  return {
    sharpThreshold: (params.sharpThresholdDegrees * Math.PI) / 180,
    pixelTolerance: params.pixelTolerance,
    extremaMinIndexGap: params.extremaMinIndexGap,
    extremaMinSpanPoints: params.extremaMinSpanPoints,
    extremaMaxIterations: params.extremaMaxIterations,
    fitTolerance: params.fitTolerance,
    nodeBalanceWeight: params.nodeBalanceWeight,
    segmentBalanceWeight: params.segmentBalanceWeight,
    g2Weight: params.g2Weight,
    handleShrinkWeight: params.handleShrinkWeight,
    maxSplitDepth: params.maxSplitDepth,
    resolutionScale:
      params.resolutionScale > 0 ? params.resolutionScale : undefined,
  };
}

function syncParamValue(param) {
  const input = paramInputs.get(param.key);
  if (!input) {
    return;
  }
  const value = Number(input.value);
  state.vectorizerParams[param.key] = value;
  const valueEl = input.closest(".param-card")?.querySelector(".param-value");
  if (valueEl) {
    valueEl.textContent = param.format(value);
  }
}

function renderParamControls() {
  paramControls.innerHTML = "";

  for (const param of PARAM_DEFS) {
    const card = document.createElement("div");
    card.className = "param-card";

    const topline = document.createElement("div");
    topline.className = "param-topline";

    const title = document.createElement("div");
    title.className = "param-title";
    title.textContent = param.title;

    const value = document.createElement("div");
    value.className = "param-value";
    value.textContent = param.format(state.vectorizerParams[param.key]);

    const description = document.createElement("p");
    description.className = "param-description";
    description.textContent = param.description;

    const input = document.createElement("input");
    input.className = "param-input";
    input.type = "range";
    input.min = String(param.min);
    input.max = String(param.max);
    input.step = String(param.step);
    input.value = String(state.vectorizerParams[param.key]);
    input.dataset.paramKey = param.key;

    input.addEventListener("input", () => {
      syncParamValue(param);
      scheduleVectorizerRerun();
    });

    topline.append(title, value);
    card.append(topline, description, input);
    paramControls.appendChild(card);
    paramInputs.set(param.key, input);
  }
}

function resetParamsToDefaults() {
  for (const param of PARAM_DEFS) {
    state.vectorizerParams[param.key] = param.defaultValue;
    const input = paramInputs.get(param.key);
    if (input) {
      input.value = String(param.defaultValue);
      syncParamValue(param);
    }
  }
}

function scheduleVectorizerRerun() {
  if (!state.imageRaster) {
    setStatus(
      "Parameters updated. Upload a PNG/JPEG to run glyphtracy-ts vectorization.",
    );
    return;
  }
  if (state.rerunTimer !== null) {
    window.clearTimeout(state.rerunTimer);
  }
  state.rerunTimer = window.setTimeout(() => {
    state.rerunTimer = null;
    runVectorizerOnRaster(
      state.imageRaster,
      state.currentImageLabel ?? "current image",
    );
  }, 120);
}

function setStatus(message) {
  statusEl.textContent = message;
}

function fragmentColor(fragmentId, fragmentCount) {
  const hue = ((fragmentId / fragmentCount) * 720) % 360;
  return `hsl(${hue} 78% 62%)`;
}

function getContourId(contour, fallbackIndex) {
  if (typeof contour?.contour_id === "number") {
    return contour.contour_id;
  }
  return fallbackIndex;
}

function getEntryId(entry, fallbackIndex) {
  return (
    entry.fragment_id ?? entry.segment_id ?? entry.node_id ?? fallbackIndex + 1
  );
}

function getEntryContourId(entry) {
  return entry.contour_id ?? entry.root_contour_index;
}

function getRenderableEntries() {
  return state.debugData?.segments ?? [];
}

function getCurrentSelectionSnapshot() {
  const contours = state.debugData?.contours ?? [];
  const contour = contours[state.contourIndex] ?? null;
  return {
    contourId:
      contour && state.contourIndex != null
        ? getContourId(contour, state.contourIndex)
        : null,
    contourIndex: state.contourIndex,
    stage: state.stage,
  };
}

function stageExistsForContour(debugData, contourIndex, stage) {
  if (stage == null || contourIndex == null) {
    return false;
  }

  const contours = debugData?.contours ?? [];
  const contour = contours[contourIndex];
  if (!contour) {
    return false;
  }

  const selectedContourId = getContourId(contour, contourIndex);
  const entries = debugData?.segments ?? [];
  const nodes = debugData?.nodes ?? [];

  if (stage === "nodes") {
    return nodes.some((node) => node.contour_id === selectedContourId);
  }
  if (stage === "vectorization") {
    return Array.isArray(contour.fitted) && contour.fitted.length > 0;
  }

  return entries.some(
    (entry) =>
      getEntryContourId(entry) === selectedContourId && entry.stage === stage,
  );
}

function restoreSelection(snapshot, debugData) {
  const contours = debugData?.contours ?? [];
  if (contours.length === 0) {
    state.contourIndex = null;
    state.stage = null;
    return;
  }

  let nextContourIndex = 0;
  if (snapshot.contourId != null) {
    const matchedIndex = contours.findIndex(
      (contour, index) => getContourId(contour, index) === snapshot.contourId,
    );
    if (matchedIndex !== -1) {
      nextContourIndex = matchedIndex;
    } else if (
      snapshot.contourIndex != null &&
      snapshot.contourIndex >= 0 &&
      snapshot.contourIndex < contours.length
    ) {
      nextContourIndex = snapshot.contourIndex;
    }
  }

  state.contourIndex = nextContourIndex;

  if (stageExistsForContour(debugData, nextContourIndex, snapshot.stage)) {
    state.stage = snapshot.stage;
  } else {
    state.stage = null;
  }
}

function rcToXy(point) {
  return { x: Number(point[1]), y: Number(point[0]) };
}

function getContourBounds(points) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const point of points) {
    const { x, y } = rcToXy(point);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  return { minX, minY, maxX, maxY };
}

function populateContourOptions() {
  const contours = state.debugData?.contours ?? [];
  contourSelect.innerHTML = "";

  contours.forEach((contour, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `Contour ${index} (${contour.points.length} points)`;
    contourSelect.appendChild(option);
  });

  if (!contours.length) {
    state.contourIndex = null;
    return;
  }

  if (state.contourIndex == null || state.contourIndex >= contours.length) {
    state.contourIndex = 0;
  }

  contourSelect.value = String(state.contourIndex);
}

function populateStageOptions() {
  const entries = getRenderableEntries();
  const contour = state.debugData?.contours?.[state.contourIndex];
  const selectedContourId = getContourId(contour, state.contourIndex);

  const stages = [
    ...new Set(
      entries
        .filter((entry) => getEntryContourId(entry) === selectedContourId)
        .map((entry) => entry.stage)
        .filter(Boolean),
    ),
  ].sort();

  if (Array.isArray(state.debugData?.nodes)) {
    const hasNodesForContour = state.debugData.nodes.some(
      (node) => node.contour_id === selectedContourId,
    );
    if (hasNodesForContour) {
      stages.push("nodes");
    }
  }

  if (Array.isArray(contour?.fitted) && contour.fitted.length) {
    stages.push("vectorization");
  }

  stageSelect.innerHTML = "";
  stages.forEach((stage) => {
    const option = document.createElement("option");
    option.value = stage;
    option.textContent = stage;
    stageSelect.appendChild(option);
  });

  if (!stages.length) {
    state.stage = null;
    return;
  }

  if (!stages.includes(state.stage)) {
    state.stage = stages[0];
  }

  stageSelect.value = state.stage;
}

function renderSummary(entries, contourPoints) {
  const pointsShown = entries.reduce(
    (sum, entry) => sum + (entry.contour_indices?.length ?? 0),
    0,
  );
  summaryEl.innerHTML = "";

  const chips = [
    `Contour ${state.contourIndex}`,
    `Stage ${state.stage ?? "none"}`,
    `${entries.length} entries`,
    `${pointsShown} entry points`,
    `${contourPoints.length} contour points`,
  ];

  for (const text of chips) {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.textContent = text;
    summaryEl.appendChild(chip);
  }
}

function formatNodeTooltip(entry, entryId) {
  const entryLabel = entry.stage === "path_segment" ? "Segment" : "Node";
  const reasons = Array.isArray(entry.reasons)
    ? entry.reasons.join(", ")
    : Array.isArray(entry.reason_tags)
      ? entry.reason_tags.join(", ")
      : "n/a";

  const transition = entry.transition ?? entry.kind ?? "n/a";
  const continuity = entry.continuity ?? "n/a";
  const index =
    typeof entry.contour_index === "number"
      ? entry.contour_index
      : Array.isArray(entry.contour_indices)
        ? entry.contour_indices[0]
        : "n/a";

  return `
    <div class="tooltip-title">${entryLabel} #${entryId}</div>
    <div class="tooltip-line">Index: ${index}</div>
    <div class="tooltip-line">Reasons: ${reasons}</div>
    <div class="tooltip-line">Split type: ${transition}</div>
    <div class="tooltip-line">Continuity: ${continuity}</div>
  `;
}

function withNodeMetadata(entry, nodesById) {
  if (entry.stage === "nodes") {
    return entry;
  }

  const startNode = nodesById.get(entry.start_node_id);
  const endNode = nodesById.get(entry.end_node_id);
  if (!startNode) {
    return entry;
  }

  return {
    ...entry,
    contour_index: startNode.contour_index,
    reasons: startNode.reasons,
    reason_tags: startNode.reason_tags,
    transition: startNode.transition,
    continuity: startNode.continuity,
    tooltip_suffix: endNode
      ? `\n    <div class="tooltip-line">End node: #${endNode.node_id}</div>`
      : "",
  };
}

function showTooltip(event, html) {
  hoverTooltip.innerHTML = html;
  hoverTooltip.classList.add("visible");
  hoverTooltip.setAttribute("aria-hidden", "false");
  positionTooltip(event);
}

function positionTooltip(event) {
  const margin = 14;
  const x = event.clientX + margin;
  const y = event.clientY + margin;
  const rect = hoverTooltip.getBoundingClientRect();

  let left = x;
  let top = y;
  if (left + rect.width > window.innerWidth - 8) {
    left = event.clientX - rect.width - margin;
  }
  if (top + rect.height > window.innerHeight - 8) {
    top = event.clientY - rect.height - margin;
  }

  hoverTooltip.style.left = `${Math.max(8, left)}px`;
  hoverTooltip.style.top = `${Math.max(8, top)}px`;
}

function hideTooltip() {
  hoverTooltip.classList.remove("visible");
  hoverTooltip.setAttribute("aria-hidden", "true");
}

function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, String(v));
  }
  return el;
}

function renderVectorizationStage(svg, fittedSegments) {
  const anchors = [];
  const handles = [];

  let d = "";
  let anchorIndex = 0;

  fittedSegments.forEach((seg, si) => {
    if (si === 0) {
      d += `M ${seg.start[0]},${seg.start[1]} `;
      anchors.push({
        x: seg.start[0],
        y: seg.start[1],
        label: String(anchorIndex++),
      });
    }

    if (seg.type === "line") {
      d += `L ${seg.end[0]},${seg.end[1]} `;
    } else if (seg.type === "cubic") {
      d += `C ${seg.control1[0]},${seg.control1[1]} ${seg.control2[0]},${seg.control2[1]} ${seg.end[0]},${seg.end[1]} `;
      handles.push({
        ax: seg.start[0],
        ay: seg.start[1],
        cx: seg.control1[0],
        cy: seg.control1[1],
      });
      handles.push({
        ax: seg.end[0],
        ay: seg.end[1],
        cx: seg.control2[0],
        cy: seg.control2[1],
      });
    }

    if (si < fittedSegments.length - 1) {
      anchors.push({
        x: seg.end[0],
        y: seg.end[1],
        label: String(anchorIndex++),
      });
    }
  });

  svg.appendChild(
    svgEl("path", {
      d,
      fill: "none",
      stroke: "rgba(255,255,255,0.55)",
      "stroke-width": "1.5",
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }),
  );

  for (const h of handles) {
    svg.appendChild(
      svgEl("line", {
        x1: h.ax,
        y1: h.ay,
        x2: h.cx,
        y2: h.cy,
        stroke: "rgba(255,180,60,0.55)",
        "stroke-width": "0.8",
        "stroke-dasharray": "2 2",
      }),
    );
  }

  for (const h of handles) {
    svg.appendChild(
      svgEl("circle", {
        cx: h.cx,
        cy: h.cy,
        r: "1.8",
        fill: "rgba(255,180,60,0.85)",
        stroke: "rgba(0,0,0,0.4)",
        "stroke-width": "0.5",
      }),
    );
  }

  const nAnchors = anchors.length;
  anchors.forEach(({ x, y, label }, i) => {
    const color = fragmentColor(i, Math.max(1, nAnchors));
    svg.appendChild(
      svgEl("circle", {
        cx: x,
        cy: y,
        r: "3",
        fill: color,
        stroke: "rgba(0,0,0,0.55)",
        "stroke-width": "0.7",
      }),
    );
    const txt = svgEl("text", {
      x: x + 4,
      y: y - 4,
      fill: color,
      "font-size": "8",
      "font-family": "IBM Plex Sans, Avenir Next, sans-serif",
      "font-weight": "700",
    });
    txt.textContent = `#${label}`;
    svg.appendChild(txt);
  });
}

function renderSvg() {
  hideTooltip();
  debugSvg.innerHTML = "";
  finalSvg.innerHTML = "";

  const contours = state.debugData?.contours ?? [];
  const entries = getRenderableEntries();
  const nodes = state.debugData?.nodes ?? [];
  const nodesById = new Map(nodes.map((node) => [node.node_id, node]));
  const finalPathData = state.debugData?.final_path ?? "";
  const contour = contours[state.contourIndex];

  if (!contour || !state.stage) {
    setStatus("No contour data available.");
    return;
  }

  const contourPoints = contour.points;
  const selectedContourId = getContourId(contour, state.contourIndex);

  const stageEntries =
    state.stage === "nodes"
      ? nodes
          .filter((node) => node.contour_id === selectedContourId)
          .map((node) => ({
            ...node,
            contour_indices: [node.contour_index],
            stage: "nodes",
          }))
      : entries.filter(
          (entry) =>
            getEntryContourId(entry) === selectedContourId &&
            entry.stage === state.stage,
        );

  const bounds = getContourBounds(contourPoints);
  const padding = 24;
  const width = Math.max(1, bounds.maxX - bounds.minX) + padding * 2;
  const height = Math.max(1, bounds.maxY - bounds.minY) + padding * 2;
  debugSvg.setAttribute(
    "viewBox",
    `${bounds.minX - padding} ${bounds.minY - padding} ${width} ${height}`,
  );

  const background = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "polyline",
  );
  background.setAttribute("fill", "none");
  background.setAttribute("stroke", "rgba(255,255,255,0.22)");
  background.setAttribute("stroke-width", "1.5");
  background.setAttribute(
    "points",
    contourPoints
      .map((point) => {
        const { x, y } = rcToXy(point);
        return `${x},${y}`;
      })
      .join(" "),
  );
  debugSvg.appendChild(background);

  if (state.stage === "vectorization") {
    renderVectorizationStage(debugSvg, contour.fitted ?? []);
    renderSummary([], contourPoints);
    setStatus(
      `Showing contour ${state.contourIndex}, vectorization stage, ${(contour.fitted ?? []).length} fitted segments.`,
    );

    const finalPath = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "path",
    );
    finalPath.setAttribute("fill", "white");
    finalPath.setAttribute("stroke", "black");
    finalPath.setAttribute("stroke-width", "0.5");
    finalPath.setAttribute("d", finalPathData);
    finalSvg.setAttribute(
      "viewBox",
      `${bounds.minX - padding} ${bounds.minY - padding} ${width} ${height}`,
    );
    finalSvg.appendChild(finalPath);
    return;
  }

  stageEntries.forEach((entry, entryIndex) => {
    const entryId = getEntryId(entry, entryIndex);
    const color = fragmentColor(entryId, Math.max(1, stageEntries.length));
    const points = (entry.contour_indices ?? [])
      .map((index) => contourPoints[index])
      .filter(Boolean)
      .map(rcToXy);

    if (!points.length) {
      return;
    }

    const tooltipEntry = withNodeMetadata(entry, nodesById);

    if (points.length > 1) {
      const path = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "polyline",
      );
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", color);
      path.setAttribute("stroke-width", "2.6");
      path.setAttribute("stroke-linecap", "round");
      path.setAttribute("stroke-linejoin", "round");
      path.setAttribute(
        "points",
        points.map((point) => `${point.x},${point.y}`).join(" "),
      );
      debugSvg.appendChild(path);
    }

    points.forEach((point, index) => {
      const circle = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "circle",
      );
      circle.setAttribute("cx", String(point.x));
      circle.setAttribute("cy", String(point.y));
      circle.setAttribute("r", index === 0 ? "3" : "0.8");
      circle.setAttribute("fill", index === 0 ? color : "none");
      circle.setAttribute("stroke", "rgba(0,0,0,0.55)");
      circle.setAttribute("stroke-width", index === 0 ? "0.7" : "0.2");

      if (index === 0) {
        let tooltipHtml = formatNodeTooltip(tooltipEntry, entryId);
        if (tooltipEntry.tooltip_suffix) {
          tooltipHtml = `${tooltipHtml}${tooltipEntry.tooltip_suffix}`;
        }
        circle.addEventListener("mouseenter", (event) => {
          showTooltip(event, tooltipHtml);
        });
        circle.addEventListener("mousemove", (event) => {
          positionTooltip(event);
        });
        circle.addEventListener("mouseleave", () => {
          hideTooltip();
        });
      }

      debugSvg.appendChild(circle);
    });

    const firstPoint = points[0];
    const label = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "text",
    );
    label.setAttribute("x", String(firstPoint.x + 4));
    label.setAttribute("y", String(firstPoint.y - 4));
    label.setAttribute("fill", color);
    label.setAttribute("font-size", "8");
    label.setAttribute("font-family", "IBM Plex Sans, Avenir Next, sans-serif");
    label.setAttribute("font-weight", "700");
    label.textContent = `#${entryId}`;
    debugSvg.appendChild(label);
  });

  renderSummary(stageEntries, contourPoints);
  setStatus(
    `Showing contour ${state.contourIndex}, stage ${state.stage}, ${stageEntries.length} entries.`,
  );

  const finalPath = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "path",
  );
  finalPath.setAttribute("fill", "white");
  finalPath.setAttribute("stroke", "black");
  finalPath.setAttribute("stroke-width", "0.5");
  finalPath.setAttribute("d", finalPathData);
  finalSvg.setAttribute(
    "viewBox",
    `${bounds.minX - padding} ${bounds.minY - padding} ${width} ${height}`,
  );
  finalSvg.appendChild(finalPath);
}

function refreshControlsAndRender() {
  populateContourOptions();
  populateStageOptions();
  renderSvg();
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () =>
      reject(reader.error || new Error("FileReader failed"));
    reader.readAsDataURL(file);
  });
}

function decodeImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to decode image"));
    img.src = url;
  });
}

function rasterFromImage(img) {
  const canvas = document.createElement("canvas");
  const width = img.naturalWidth || img.width;
  const height = img.naturalHeight || img.height;
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("Could not get 2D canvas context");
  }

  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(img, 0, 0, width, height);

  const imageData = ctx.getImageData(0, 0, width, height).data;
  const raster = [];

  for (let y = 0; y < height; y += 1) {
    const row = [];
    for (let x = 0; x < width; x += 1) {
      const base = (y * width + x) * 4;
      const r = imageData[base + 0];
      const g = imageData[base + 1];
      const b = imageData[base + 2];
      const a = imageData[base + 3] / 255;
      const gray = (0.299 * r + 0.587 * g + 0.114 * b) * a + 255 * (1 - a);
      row.push(1 - gray / 255);
    }
    raster.push(row);
  }

  return raster;
}

function getSvgExportDimensions() {
  const rasterHeight = state.imageRaster?.length ?? 0;
  const rasterWidth = state.imageRaster?.[0]?.length ?? 0;

  const width = Math.max(
    1,
    Math.round(
      rasterWidth || Number(sourceImage.naturalWidth) || Number(sourceImage.width) || 1,
    ),
  );
  const height = Math.max(
    1,
    Math.round(
      rasterHeight || Number(sourceImage.naturalHeight) || Number(sourceImage.height) || 1,
    ),
  );
  return { width, height };
}

function safeBaseName(name) {
  const base = (name || "glyphtracy").replace(/\.[^/.]+$/, "");
  const safe = base.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^_+|_+$/g, "");
  return safe || "glyphtracy";
}

function downloadFinalSvg() {
  const pathData = state.debugData?.final_path;
  if (!pathData) {
    setStatus("Nothing to download yet. Upload an image and run vectorization first.");
    return;
  }

  const { width, height } = getSvgExportDimensions();
  const svgContent = `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}">\n  <path d="${pathData}" fill="black" />\n</svg>\n`;

  const blob = new Blob([svgContent], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  const imageLabel = state.currentImageLabel || "glyphtracy";
  anchor.href = url;
  anchor.download = `${safeBaseName(imageLabel)}.svg`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);

  setStatus(`Downloaded ${anchor.download}.`);
}

async function loadJsonFromUrl(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status}`);
  }
  return response.json();
}

async function loadImageFromUrl(url) {
  sourceImage.src = url;
  await sourceImage.decode();
}

async function runVectorizerOnRaster(raster, label) {
  setStatus(`Running glyphtracy-ts on ${label}...`);
  await new Promise((resolve) => requestAnimationFrame(resolve));

  const selectionSnapshot = getCurrentSelectionSnapshot();
  const vectorizer = new Vectorizer(raster, getVectorizerOptions());
  const { debugData } = vectorizer.run();

  state.debugData = debugData;
  restoreSelection(selectionSnapshot, debugData);
  refreshControlsAndRender();

  const contourCount = debugData?.contours?.length ?? 0;
  setStatus(
    `Vectorized ${label}: ${contourCount} contours, ${debugData?.segments?.length ?? 0} segments.`,
  );
}

async function loadJsonFile(file) {
  const text = await file.text();
  state.debugData = JSON.parse(text);
  refreshControlsAndRender();
  setStatus(`Loaded ${file.name}.`);
}

async function loadImageFile(file) {
  if (state.imageUrl?.startsWith("blob:")) {
    URL.revokeObjectURL(state.imageUrl);
  }

  const dataUrl = await readFileAsDataUrl(file);
  const img = await decodeImage(dataUrl);
  const raster = rasterFromImage(img);

  state.imageUrl = dataUrl;
  state.imageRaster = raster;
  state.currentImageLabel = file.name;
  await loadImageFromUrl(state.imageUrl);
  await runVectorizerOnRaster(raster, file.name);
}

contourSelect.addEventListener("change", () => {
  state.contourIndex = Number(contourSelect.value);
  populateStageOptions();
  renderSvg();
});

stageSelect.addEventListener("change", () => {
  state.stage = stageSelect.value;
  renderSvg();
});

imageInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (file) {
    await loadImageFile(file);
  }
});

resetParamsButton.addEventListener("click", () => {
  resetParamsToDefaults();
  scheduleVectorizerRerun();
});

downloadSvgButton.addEventListener("click", () => {
  downloadFinalSvg();
});

renderParamControls();

setStatus(
  "Ready. Upload a PNG/JPEG to run glyphtracy-ts in browser, then adjust the controls to rerun vectorization.",
);
