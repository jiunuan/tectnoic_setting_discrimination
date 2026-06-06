export function mountWorkflowUi() {
const CANVAS_SIZE = { width: 3400, height: 1100 };
const NODE_SIZE = { width: 196, height: 108 };
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 2.0;
const ZOOM_STEP = 0.1;
const RESIZER_SIZE = 14;
const PANEL_LIMITS = {
  left: { min: 180, max: 560 },
  right: { min: 220, max: 800 },
  rightTop: { min: 180 },
  rightBottom: { min: 180 },
  canvasMin: 420,
};
const PANEL_STORAGE_KEY = "geochem-workflow-layout-v1";

const initialState = {
  catalog: [],
  catalogMap: {},
  workflow: { name: "default", nodes: [], edges: [] },
  savedWorkflows: [],
  selectedNodeId: null,
  selectedEdgeId: null,
  dragNodeId: null,
  dragOffset: { x: 0, y: 0 },
  connectionDraft: null,
  runtimeContext: {},
  activeRunId: null,
  runData: null,
  pollTimer: null,
  zoom: 1,
  panelResize: null,
  panelSizes: { left: 270, right: 320, rightTop: 360 },
  currentStepInfo: {
    status: "idle",
    title: "\u7b49\u5f85\u8fd0\u884c",
    subtitle: "\u542f\u52a8\u540e\u4f1a\u5728\u8fd9\u91cc\u663e\u793a\u5f53\u524d\u8282\u70b9\u548c\u811a\u672c\u3002",
  },
  displayLogs: [],
};

const state = window.Vue?.reactive ? window.Vue.reactive(initialState) : initialState;

const dom = {
  workspace: document.getElementById("workspace"),
  palette: document.getElementById("palette"),
  canvas: document.getElementById("canvas"),
  edgeLayer: document.getElementById("edgeLayer"),
  canvasScaler: document.getElementById("canvasScaler"),
  viewport: document.getElementById("canvasViewport"),
  propertyPanel: document.getElementById("propertyPanel"),
  logPanel: document.getElementById("logPanel"),
  rightPanelSections: document.getElementById("rightPanelSections"),
  savedWorkflowSelect: document.getElementById("savedWorkflowSelect"),
  runStatusBadge: document.getElementById("runStatusBadge"),
  currentStepIndicator: document.getElementById("currentStepIndicator"),
  currentStepText: document.getElementById("currentStepText"),
  currentStepSubtext: document.getElementById("currentStepSubtext"),
  contextRepoRoot: document.getElementById("contextRepoRoot"),
  workflowName: document.getElementById("workflowName"),
  loadPresetBtn: document.getElementById("loadPresetBtn"),
  saveWorkflowBtn: document.getElementById("saveWorkflowBtn"),
  exportWorkflowBtn: document.getElementById("exportWorkflowBtn"),
  deleteSelectedBtn: document.getElementById("deleteSelectedBtn"),
  stopWorkflowBtn: document.getElementById("stopWorkflowBtn"),
  runWorkflowBtn: document.getElementById("runWorkflowBtn"),
  loadSavedWorkflowBtn: document.getElementById("loadSavedWorkflowBtn"),
  refreshWorkflowsBtn: document.getElementById("refreshWorkflowsBtn"),
  zoomOutBtn: document.getElementById("zoomOutBtn"),
  zoomResetBtn: document.getElementById("zoomResetBtn"),
  zoomInBtn: document.getElementById("zoomInBtn"),
  zoomFitBtn: document.getElementById("zoomFitBtn"),
  leftResizeHandle: document.getElementById("leftResizeHandle"),
  rightResizeHandle: document.getElementById("rightResizeHandle"),
  bottomResizeHandle: document.getElementById("bottomResizeHandle"),
};

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function uid(prefix = "node") {
  return `${prefix}_${Math.random().toString(16).slice(2, 10)}`;
}

function api(path, options = {}) {
  return fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  }).then(async (response) => {
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || `Request failed: ${response.status}`);
    }
    return data;
  });
}

function clampZoom(value) {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Number(value.toFixed(2))));
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function loadPanelSizes() {
  try {
    const raw = window.localStorage.getItem(PANEL_STORAGE_KEY);
    if (!raw) {
      return;
    }
    const parsed = JSON.parse(raw);
    if (Number.isFinite(parsed.left)) {
      state.panelSizes.left = parsed.left;
    }
    if (Number.isFinite(parsed.right)) {
      state.panelSizes.right = parsed.right;
    }
    if (Number.isFinite(parsed.rightTop)) {
      state.panelSizes.rightTop = parsed.rightTop;
    }
  } catch (error) {
    console.warn("Failed to restore panel sizes", error);
  }
}

function savePanelSizes() {
  try {
    window.localStorage.setItem(PANEL_STORAGE_KEY, JSON.stringify(state.panelSizes));
  } catch (error) {
    console.warn("Failed to persist panel sizes", error);
  }
}

function getMaxResizableWidth(side) {
  const totalSideWidth = dom.workspace.clientWidth - RESIZER_SIZE * 2 - PANEL_LIMITS.canvasMin;
  const otherWidth = side === "left" ? state.panelSizes.right : state.panelSizes.left;
  const limit = side === "left" ? PANEL_LIMITS.left : PANEL_LIMITS.right;
  return Math.max(limit.min, Math.min(limit.max, totalSideWidth - otherWidth));
}

function getMaxResizableRightTopHeight() {
  if (!dom.rightPanelSections) {
    return PANEL_LIMITS.rightTop.min;
  }
  const totalHeight = dom.rightPanelSections.clientHeight - RESIZER_SIZE;
  return Math.max(PANEL_LIMITS.rightTop.min, totalHeight - PANEL_LIMITS.rightBottom.min);
}

function applyPanelSizes(persist = false) {
  if (!dom.workspace) {
    return;
  }

  state.panelSizes.left = clamp(Math.round(state.panelSizes.left), PANEL_LIMITS.left.min, PANEL_LIMITS.left.max);
  state.panelSizes.right = clamp(Math.round(state.panelSizes.right), PANEL_LIMITS.right.min, PANEL_LIMITS.right.max);

  const totalSideWidth = dom.workspace.clientWidth - RESIZER_SIZE * 2 - PANEL_LIMITS.canvasMin;
  if (totalSideWidth > 0 && state.panelSizes.left + state.panelSizes.right > totalSideWidth) {
    const overflow = state.panelSizes.left + state.panelSizes.right - totalSideWidth;
    const rightReducible = Math.max(0, state.panelSizes.right - PANEL_LIMITS.right.min);
    const reduceRight = Math.min(overflow, rightReducible);
    state.panelSizes.right -= reduceRight;
    const remaining = overflow - reduceRight;
    if (remaining > 0) {
      state.panelSizes.left = Math.max(PANEL_LIMITS.left.min, state.panelSizes.left - remaining);
    }
  }

  dom.workspace.style.setProperty("--left-panel-width", `${state.panelSizes.left}px`);
  dom.workspace.style.setProperty("--right-panel-width", `${state.panelSizes.right}px`);

  const maxRightTop = getMaxResizableRightTopHeight();
  state.panelSizes.rightTop = clamp(
    Math.round(state.panelSizes.rightTop),
    PANEL_LIMITS.rightTop.min,
    maxRightTop,
  );
  dom.workspace.style.setProperty("--right-top-panel-height", `${state.panelSizes.rightTop}px`);

  if (persist) {
    savePanelSizes();
  }
}

function getViewportPoint(clientX, clientY) {
  const rect = dom.viewport.getBoundingClientRect();
  return {
    x: (clientX - rect.left + dom.viewport.scrollLeft) / state.zoom,
    y: (clientY - rect.top + dom.viewport.scrollTop) / state.zoom,
  };
}

function getAnchorOffset(anchor) {
  const rect = dom.viewport.getBoundingClientRect();
  if (anchor?.clientX != null && anchor?.clientY != null) {
    return {
      x: anchor.clientX - rect.left,
      y: anchor.clientY - rect.top,
    };
  }
  return {
    x: rect.width / 2,
    y: rect.height / 2,
  };
}

function applyZoom() {
  const scaledWidth = Math.round(CANVAS_SIZE.width * state.zoom);
  const scaledHeight = Math.round(CANVAS_SIZE.height * state.zoom);
  dom.canvasScaler.style.width = `${scaledWidth}px`;
  dom.canvasScaler.style.height = `${scaledHeight}px`;
  dom.canvas.style.transform = `scale(${state.zoom})`;
  dom.edgeLayer.style.transform = `scale(${state.zoom})`;
  dom.zoomResetBtn.textContent = `${Math.round(state.zoom * 100)}%`;
}

function setZoom(nextZoom, anchor = null) {
  const zoom = clampZoom(nextZoom);
  if (zoom === state.zoom) {
    applyZoom();
    return;
  }

  const anchorOffset = getAnchorOffset(anchor);
  const logicalX = (dom.viewport.scrollLeft + anchorOffset.x) / state.zoom;
  const logicalY = (dom.viewport.scrollTop + anchorOffset.y) / state.zoom;

  state.zoom = zoom;
  applyZoom();

  requestAnimationFrame(() => {
    dom.viewport.scrollLeft = Math.max(0, logicalX * state.zoom - anchorOffset.x);
    dom.viewport.scrollTop = Math.max(0, logicalY * state.zoom - anchorOffset.y);
  });
}

function fitCanvasToViewport() {
  const padding = 32;
  const fitWidth = (dom.viewport.clientWidth - padding) / CANVAS_SIZE.width;
  const fitHeight = (dom.viewport.clientHeight - padding) / CANVAS_SIZE.height;
  state.zoom = clampZoom(Math.min(fitWidth, fitHeight, 1));
  applyZoom();
  requestAnimationFrame(() => {
    dom.viewport.scrollLeft = 0;
    dom.viewport.scrollTop = 0;
  });
}

function escapeRegExp(text) {
  return String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function getNodeRuntime(nodeId) {
  return state.runData?.node_statuses?.[nodeId] || { status: "", detail: "" };
}

function getNodeStatus(nodeId) {
  return getNodeRuntime(nodeId).status || "";
}

function getPrimaryNodeByStatus(status) {
  return state.workflow.nodes.find((node) => getNodeStatus(node.id) === status) || null;
}

function getLastNodeByStatus(status) {
  const nodes = [...state.workflow.nodes].reverse();
  return nodes.find((node) => getNodeStatus(node.id) === status) || null;
}

function getCommandSummary(commandText) {
  const matches = [...String(commandText || "").matchAll(/"([^"]+\.(?:py|ps1|bat|cmd|exe))"|([^\s"]+\.(?:py|ps1|bat|cmd|exe))/gi)]
    .map((match) => match[1] || match[2])
    .filter(Boolean)
    .filter((item) => !/^(?:python|python\.exe|py|py\.exe)$/i.test(item.split(/[\/]/).pop()));
  if (!matches.length) {
    return "";
  }
  return matches[matches.length - 1].split(/[\/]/).pop();
}

function resolveRuntimeText(value) {
  return String(value ?? "").replace(/\{([^{}]+)\}/g, (_, key) => {
    return Object.prototype.hasOwnProperty.call(state.runtimeContext || {}, key)
      ? state.runtimeContext[key]
      : `{${key}}`;
  });
}

function getNodeCommandSummary(node) {
  if (node.type !== "command_task") {
    return "";
  }
  const resolvedCommand = resolveRuntimeText(node.params?.command || "");
  return getCommandSummary(resolvedCommand) || node.params?.label || "";
}

function getLatestNodeLogMessage(nodeTitle) {
  if (!nodeTitle) {
    return "";
  }
  const logs = state.runData?.logs || [];
  const prefix = new RegExp(`^\[[^\]]+\]\s${escapeRegExp(nodeTitle)}:\s?(.*)$`);
  for (let index = logs.length - 1; index >= 0; index -= 1) {
    const match = logs[index].match(prefix);
    if (match) {
      return match[1];
    }
  }
  return "";
}

function getNodeRuntimeCaption(node, template, status) {
  const commandSummary = getNodeCommandSummary(node);
  if (status === "running") {
    return commandSummary ? `执行中 · ${commandSummary}` : "执行中";
  }
  if (status === "completed") {
    return commandSummary ? `已完成 · ${commandSummary}` : "已完成";
  }
  if (status === "failed") {
    return commandSummary ? `失败 · ${commandSummary}` : "执行失败";
  }
  if (status === "cancelled") {
    return commandSummary ? `已停止 · ${commandSummary}` : "已停止";
  }
  if (status === "skipped") {
    return "已跳过";
  }
  if (commandSummary) {
    return `脚本 · ${commandSummary}`;
  }
  return `步骤 · ${template?.label || node.type}`;
}

function getCurrentStepInfo() {
  const status = state.runData?.status || "idle";
  if (status === "running") {
    const runningNode = getPrimaryNodeByStatus("running");
    if (!runningNode) {
      return {
        status,
        title: "正在启动工作流",
        subtitle: state.runData?.cancel_requested ? "正在停止当前工作流..." : "等待第一个节点进入运行状态。",
      };
    }
    const latestMessage = getLatestNodeLogMessage(runningNode.title);
    const commandSummary = getNodeCommandSummary(runningNode);
    let subtitle = latestMessage || "节点运行中";
    if (!latestMessage || latestMessage === "running") {
      subtitle = commandSummary ? `当前脚本: ${commandSummary}` : "节点运行中";
    }
    return {
      status,
      title: runningNode.title,
      subtitle,
    };
  }

  if (status === "failed") {
    const failedNode = getPrimaryNodeByStatus("failed") || getPrimaryNodeByStatus("running");
    return {
      status,
      title: failedNode ? `${failedNode.title} 执行失败` : "工作流执行失败",
      subtitle: state.runData?.error || "请查看下方运行日志。",
    };
  }

  if (status === "cancelled") {
    const cancelledNode = getPrimaryNodeByStatus("cancelled") || getPrimaryNodeByStatus("running");
    return {
      status,
      title: cancelledNode ? `${cancelledNode.title} 已停止` : "工作流已停止",
      subtitle: "当前运行已被用户取消。",
    };
  }

  if (status === "completed") {
    const lastNode = getLastNodeByStatus("completed");
    return {
      status,
      title: "流程执行完成",
      subtitle: lastNode ? `最后完成: ${lastNode.title}` : "所有节点已执行完毕。",
    };
  }

  return {
    status: "idle",
    title: "等待运行",
    subtitle: "启动后会在这里显示当前节点和脚本。",
  };
}

function renderRunProgress() {
  const info = getCurrentStepInfo();
  state.currentStepInfo = info;
  if (!window.Vue) {
    dom.currentStepIndicator.className = `current-step ${info.status}`;
    dom.currentStepText.textContent = info.title;
    dom.currentStepSubtext.textContent = info.subtitle;
  }
}

function getNodeById(nodeId) {
  return state.workflow.nodes.find((node) => node.id === nodeId);
}

function getNodeRect(node) {
  return {
    x: node.position.x,
    y: node.position.y,
    width: NODE_SIZE.width,
    height: NODE_SIZE.height,
  };
}

function bezierPath(source, target) {
  const c1 = source.x + 90;
  const c2 = target.x - 90;
  return `M ${source.x} ${source.y} C ${c1} ${source.y}, ${c2} ${target.y}, ${target.x} ${target.y}`;
}

function renderPalette() {
  const grouped = {};
  state.catalog.forEach((template) => {
    grouped[template.category] = grouped[template.category] || [];
    grouped[template.category].push(template);
  });
  dom.palette.innerHTML = Object.entries(grouped)
    .map(
      ([category, items]) => `
        <section class="palette-group">
          <h3>${escapeHtml(category)}</h3>
          ${items
            .map(
              (item) => `
                <div class="palette-item" draggable="true" data-node-type="${escapeHtml(item.type)}">
                  <strong>${escapeHtml(item.label)}</strong>
                  <span>${escapeHtml(item.description)}</span>
                </div>
              `,
            )
            .join("")}
        </section>
      `,
    )
    .join("");

  dom.palette.querySelectorAll(".palette-item").forEach((item) => {
    item.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData("text/plain", item.dataset.nodeType);
    });
    item.addEventListener("dblclick", () => {
      addNode(item.dataset.nodeType, { x: 160, y: 120 });
    });
  });
}

function renderSavedWorkflows() {
  const options = state.savedWorkflows
    .map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)
    .join("");
  dom.savedWorkflowSelect.innerHTML = options || `<option value="">暂无</option>`;
}

function renderCanvas() {
  if (!window.Vue && dom.workflowName) {
    dom.workflowName.textContent = state.workflow.name || "untitled";
  }
  dom.canvas.innerHTML = state.workflow.nodes
    .map((node) => {
      const template = state.catalogMap[node.type];
      const status = getNodeStatus(node.id);
      const selected = node.id === state.selectedNodeId ? "selected" : "";
      const runtimeCaption = node.type === "note" ? "" : getNodeRuntimeCaption(node, template, status);
      const nodeContent = node.type === "note"
        ? `<div class="node-note">${escapeHtml(node.params.content || "")}</div>`
        : `
            <div class="node-runtime-pill ${status || "idle"}">${escapeHtml(runtimeCaption)}</div>
            <div class="node-desc">${escapeHtml(template?.description || "")}</div>
          `;
      return `
        <div class="node ${selected} ${status}" data-node-id="${escapeHtml(node.id)}"
             style="left:${node.position.x}px; top:${node.position.y}px;">
          <div class="handle input" data-handle-role="input" data-node-id="${escapeHtml(node.id)}"></div>
          <div class="handle output" data-handle-role="output" data-node-id="${escapeHtml(node.id)}"></div>
          <div class="node-header">
            <div class="node-title">${escapeHtml(node.title || template?.label || node.type)}</div>
            <div class="node-status-dot ${status}"></div>
          </div>
          <div class="node-meta">${escapeHtml(node.type)}</div>
          ${nodeContent}
        </div>
      `;
    })
    .join("");

  dom.canvas.querySelectorAll(".node").forEach((nodeEl) => {
    nodeEl.addEventListener("pointerdown", handleNodePointerDown);
    nodeEl.addEventListener("click", (event) => {
      if (event.target.classList.contains("handle")) {
        return;
      }
      selectNode(nodeEl.dataset.nodeId);
    });
  });
  dom.canvas.querySelectorAll(".handle").forEach((handle) => {
    if (handle.dataset.handleRole === "output") {
      handle.addEventListener("pointerdown", startConnectionDraft);
    } else {
      handle.addEventListener("pointerup", finishConnectionDraft);
    }
  });
  renderEdges();
  applyZoom();
}

function renderEdges() {
  const runningNode = getPrimaryNodeByStatus("running");
  const runningNodeId = runningNode?.id || "";
  const completedIds = new Set(
    state.workflow.nodes
      .filter((node) => getNodeStatus(node.id) === "completed")
      .map((node) => node.id),
  );

  const paths = state.workflow.edges
    .map((edge) => {
      const sourceNode = getNodeById(edge.source);
      const targetNode = getNodeById(edge.target);
      if (!sourceNode || !targetNode) {
        return "";
      }
      const sourceRect = getNodeRect(sourceNode);
      const targetRect = getNodeRect(targetNode);
      const path = bezierPath(
        { x: sourceRect.x + sourceRect.width, y: sourceRect.y + sourceRect.height / 2 },
        { x: targetRect.x, y: targetRect.y + targetRect.height / 2 },
      );
      const classes = ["edge-path"];
      if (edge.id === state.selectedEdgeId) {
        classes.push("selected");
      }
      if (edge.source === runningNodeId || edge.target === runningNodeId) {
        classes.push("active-flow");
      } else if (completedIds.has(edge.source) && completedIds.has(edge.target)) {
        classes.push("completed-flow");
      }
      return `<path class="${classes.join(" ")}" data-edge-id="${escapeHtml(edge.id)}" d="${path}"></path>`;
    })
    .join("");

  const draft = state.connectionDraft
    ? (() => {
        const sourceNode = getNodeById(state.connectionDraft.sourceId);
        if (!sourceNode) return "";
        const sourceRect = getNodeRect(sourceNode);
        const path = bezierPath(
          { x: sourceRect.x + sourceRect.width, y: sourceRect.y + sourceRect.height / 2 },
          state.connectionDraft.cursor,
        );
        return `<path class="edge-preview" d="${path}"></path>`;
      })()
    : "";

  dom.edgeLayer.innerHTML = paths + draft;
  dom.edgeLayer.querySelectorAll(".edge-path").forEach((path) => {
    path.addEventListener("click", () => {
      state.selectedEdgeId = path.dataset.edgeId;
      state.selectedNodeId = null;
      renderCanvas();
      renderProperties();
    });
  });
}

function renderProperties() {
  if (!state.selectedNodeId) {
    dom.propertyPanel.className = "property-panel empty";
    dom.propertyPanel.textContent = "选择一个节点以编辑参数。";
    return;
  }

  const node = getNodeById(state.selectedNodeId);
  const template = state.catalogMap[node.type];
  const fields = template.params
    .map((param) => {
      const value = node.params?.[param.key] ?? "";
      if (param.type === "textarea") {
        return `
          <label>
            ${escapeHtml(param.label)}
            <textarea data-param-key="${escapeHtml(param.key)}">${escapeHtml(value)}</textarea>
          </label>
        `;
      }
      if (param.type === "number") {
        return `
          <label>
            ${escapeHtml(param.label)}
            <input type="number" step="any" data-param-key="${escapeHtml(param.key)}" value="${escapeHtml(value)}">
          </label>
        `;
      }
      return `
        <label>
          ${escapeHtml(param.label)}
          <input type="text" data-param-key="${escapeHtml(param.key)}" value="${escapeHtml(value)}">
        </label>
      `;
    })
    .join("");

  dom.propertyPanel.className = "property-panel";
  dom.propertyPanel.innerHTML = `
    <label>
      节点标题
      <input type="text" id="nodeTitleInput" value="${escapeHtml(node.title || "")}">
    </label>
    ${fields}
  `;

  dom.propertyPanel.querySelector("#nodeTitleInput").addEventListener("input", (event) => {
    node.title = event.target.value;
    renderCanvas();
  });

  dom.propertyPanel.querySelectorAll("[data-param-key]").forEach((input) => {
    input.addEventListener("input", (event) => {
      const key = event.target.dataset.paramKey;
      let value = event.target.value;
      const schema = template.params.find((item) => item.key === key);
      if (schema?.type === "number") {
        value = Number(value);
      }
      node.params[key] = value;
    });
  });
}

function parseStructuredLog(line) {
  const match = String(line ?? "").match(/^\[([^\]]+)\]\s(.+?):\s(.*)$/);
  if (!match) {
    return { time: "", source: "", message: String(line ?? "") };
  }
  return {
    time: match[1],
    source: match[2],
    message: match[3],
  };
}

function cleanLogMessage(message) {
  return String(message ?? "")
    .replace(/^\[stderr\]\s*/i, "")
    .replace(/\r/g, "")
    .trim();
}

function parseTqdmProgress(message) {
  const clean = cleanLogMessage(message).trimStart();
  const patterns = [
    /^(Epoch\s+\d+\/\d+\s+\[[^\]]+\]):\s+(\d+)%\|(.*)$/i,
    /^(\[[^\]]+\]):\s+(\d+)%\|(.*)$/i,
  ];
  for (const pattern of patterns) {
    const match = clean.match(pattern);
    if (match) {
      const percent = Math.max(0, Math.min(100, Number(match[2]) || 0));
      return {
        label: match[1],
        percent,
        detail: `${match[2]}%|${match[3]}`.trim(),
      };
    }
  }
  return null;
}

function parseEpochSummary(message) {
  const clean = cleanLogMessage(message);
  const match = clean.match(/^Epoch\s+(\d+)\/(\d+)\s+[?|]\s+Train Loss:\s*([0-9.]+)\s+Acc:\s*([0-9.]+)\s+[?|]\s+Val Loss:\s*([0-9.]+)\s+Acc:\s*([0-9.]+)\s+[?|]\s+LR:\s*([0-9.eE+-]+)$/);
  if (!match) {
    return null;
  }
  return {
    epoch: Number(match[1]),
    totalEpochs: Number(match[2]),
    trainLoss: match[3],
    trainAcc: match[4],
    valLoss: match[5],
    valAcc: match[6],
    learningRate: match[7],
  };
}

function getLogClasses(line, progress, summary) {
  const classes = ["log-line"];
  if (summary) {
    classes.push("epoch-summary", "completed");
    return classes;
  }
  if (progress) {
    classes.push("progress", progress.percent >= 100 ? "completed" : "running");
    return classes;
  }
  if (/\[stderr\]/i.test(line)) {
    classes.push("stream-stderr");
  }
  if (/\[warn\]/i.test(line)) {
    classes.push("warn");
  } else if (/cancelled|cancellation requested/i.test(line)) {
    classes.push("cancelled");
  } else if (/completed|finished successfully/i.test(line)) {
    classes.push("completed");
  } else if (/traceback|exception|\berror\b|failed/i.test(line)) {
    classes.push("failed");
  } else if (/running/i.test(line)) {
    classes.push("running");
  }
  return classes;
}

function buildDisplayLogs(logs) {
  const entries = [];
  logs.forEach((line) => {
    const parsed = parseStructuredLog(line);
    const cleanedMessage = cleanLogMessage(parsed.message || line);
    if (!cleanedMessage) {
      return;
    }

    const progress = parseTqdmProgress(parsed.message);
    const summary = parseEpochSummary(parsed.message);
    const entry = {
      raw: line,
      parsed,
      progress,
      summary,
      classes: getLogClasses(line, progress, summary),
    };

    if (progress && entries.length) {
      const previous = entries[entries.length - 1];
      if (
        previous.progress
        && previous.parsed.source === parsed.source
        && previous.progress.label === progress.label
      ) {
        entries[entries.length - 1] = entry;
        return;
      }
    }

    entries.push(entry);
  });
  return entries;
}

function renderSummaryEntry(entry, classes) {
  return `
    <span class="${classes.join(" ")}">
      <span class="log-meta">
        <strong>${escapeHtml(entry.parsed.source || "CNN-BiLSTM")}</strong>
        <span>${escapeHtml(entry.parsed.time || "")}</span>
      </span>
      <span class="epoch-summary-head">
        <span class="epoch-badge">Epoch ${entry.summary.epoch}/${entry.summary.totalEpochs}</span>
        <span class="epoch-lr">LR ${escapeHtml(entry.summary.learningRate)}</span>
      </span>
      <span class="epoch-metrics">
        <span><label>Train Loss</label><strong>${escapeHtml(entry.summary.trainLoss)}</strong></span>
        <span><label>Train Acc</label><strong>${escapeHtml(entry.summary.trainAcc)}</strong></span>
        <span><label>Val Loss</label><strong>${escapeHtml(entry.summary.valLoss)}</strong></span>
        <span><label>Val Acc</label><strong>${escapeHtml(entry.summary.valAcc)}</strong></span>
      </span>
    </span>
  `;
}

function renderProgressEntry(entry, classes) {
  const title = entry.parsed.source || "\u8bad\u7ec3\u8fdb\u5ea6";
  return `
    <span class="${classes.join(" ")}">
      <span class="log-meta">
        <strong>${escapeHtml(title)}</strong>
        <span>${escapeHtml(entry.parsed.time || "")}</span>
      </span>
      <span class="log-progress-head">
        <span>${escapeHtml(entry.progress.label)}</span>
        <strong>${entry.progress.percent}%</strong>
      </span>
      <span class="log-progress-bar"><span style="width:${entry.progress.percent}%"></span></span>
      <span class="log-progress-detail">${escapeHtml(entry.progress.detail)}</span>
    </span>
  `;
}

function renderLogEntry(entry, isLatest) {
  const classes = [...entry.classes];
  if (isLatest) {
    classes.push("latest");
  }

  if (entry.summary) {
    return renderSummaryEntry(entry, classes);
  }
  if (entry.progress) {
    return renderProgressEntry(entry, classes);
  }
  return `<span class="${classes.join(" ")}">${escapeHtml(entry.raw)}</span>`;
}

function renderLogs() {
  const logs = state.runData?.logs || [];
  const displayEntries = buildDisplayLogs(logs);
  const entries = displayEntries.map((entry, index, all) => {
    const className = [...entry.classes, index === all.length - 1 ? "latest" : ""]
      .filter(Boolean)
      .join(" ");
    if (entry.summary) {
      return {
        id: `summary-${entry.parsed.time}-${entry.summary.epoch}-${index}`,
        kind: "summary",
        className,
        source: entry.parsed.source || "CNN-BiLSTM",
        time: entry.parsed.time || "",
        summary: entry.summary,
      };
    }
    if (entry.progress) {
      return {
        id: `progress-${entry.parsed.time}-${entry.progress.label}-${index}`,
        kind: "progress",
        className,
        source: entry.parsed.source || "训练进度",
        time: entry.parsed.time || "",
        progress: entry.progress,
      };
    }
    return {
      id: `log-${entry.parsed.time}-${index}`,
      kind: "text",
      className,
      raw: entry.raw,
    };
  });

  state.displayLogs = entries;
  if (!window.Vue) {
    dom.logPanel.innerHTML = displayEntries.length
      ? displayEntries.map((entry, index, all) => renderLogEntry(entry, index === all.length - 1)).join("")
      : '<span class="log-line empty">等待工作流开始运行。</span>';
  }
  const status = state.runData?.status || "idle";
  if (!window.Vue) {
    dom.runStatusBadge.textContent = status;
    dom.runStatusBadge.className = `status-badge ${status}`;
  }
  renderRunProgress();
  updateRunActionState();
  const nextTick = window.Vue?.nextTick || ((callback) => requestAnimationFrame(callback));
  nextTick(() => {
    if (dom.logPanel) {
      dom.logPanel.scrollTop = dom.logPanel.scrollHeight;
    }
  });
}

function updateRunActionState() {
  const isCurrentRunRunning = Boolean(state.activeRunId) && state.runData?.status === "running";
  dom.runWorkflowBtn.disabled = false;
  dom.stopWorkflowBtn.disabled = !isCurrentRunRunning;
}

function selectNode(nodeId) {
  state.selectedNodeId = nodeId;
  state.selectedEdgeId = null;
  renderCanvas();
  renderProperties();
}

function addNode(nodeType, position) {
  const template = state.catalogMap[nodeType];
  if (!template) return;
  const params = {};
  template.params.forEach((param) => {
    params[param.key] = param.default;
  });
  const node = {
    id: uid("n"),
    type: nodeType,
    title: template.label,
    position,
    params,
  };
  state.workflow.nodes.push(node);
  selectNode(node.id);
}

function startPanelResize(event) {
  if (window.innerWidth <= 1180) {
    return;
  }
  event.preventDefault();
  event.currentTarget.focus();
  const side = event.currentTarget.dataset.resizeSide;
  state.panelResize = {
    side,
    startX: event.clientX,
    startY: event.clientY,
    startLeft: state.panelSizes.left,
    startRight: state.panelSizes.right,
    startRightTop: state.panelSizes.rightTop,
    handle: event.currentTarget,
  };
  event.currentTarget.classList.add("active");
  document.body.classList.add(side === "bottom" ? "is-resizing-panels-y" : "is-resizing-panels-x");
}

function nudgePanelSize(side, delta) {
  if (side === "left") {
    state.panelSizes.left = clamp(state.panelSizes.left + delta, PANEL_LIMITS.left.min, getMaxResizableWidth("left"));
  } else if (side === "right") {
    state.panelSizes.right = clamp(state.panelSizes.right + delta, PANEL_LIMITS.right.min, getMaxResizableWidth("right"));
  } else {
    state.panelSizes.rightTop = clamp(state.panelSizes.rightTop + delta, PANEL_LIMITS.rightTop.min, getMaxResizableRightTopHeight());
  }
  applyPanelSizes(true);
}

function handleNodePointerDown(event) {
  if (event.target.classList.contains("handle")) {
    return;
  }
  const nodeId = event.currentTarget.dataset.nodeId;
  const node = getNodeById(nodeId);
  const point = getViewportPoint(event.clientX, event.clientY);
  state.dragNodeId = nodeId;
  state.dragOffset = {
    x: point.x - node.position.x,
    y: point.y - node.position.y,
  };
  selectNode(nodeId);
}

function handlePointerMove(event) {
  if (state.panelResize) {
    if (state.panelResize.side === "bottom") {
      const dy = event.clientY - state.panelResize.startY;
      state.panelSizes.rightTop = clamp(
        state.panelResize.startRightTop + dy,
        PANEL_LIMITS.rightTop.min,
        getMaxResizableRightTopHeight(),
      );
    } else {
      const dx = event.clientX - state.panelResize.startX;
      if (state.panelResize.side === "left") {
        state.panelSizes.left = clamp(
          state.panelResize.startLeft + dx,
          PANEL_LIMITS.left.min,
          getMaxResizableWidth("left"),
        );
      } else {
        state.panelSizes.right = clamp(
          state.panelResize.startRight - dx,
          PANEL_LIMITS.right.min,
          getMaxResizableWidth("right"),
        );
      }
    }
    applyPanelSizes(false);
    return;
  }

  if (state.dragNodeId) {
    const node = getNodeById(state.dragNodeId);
    const point = getViewportPoint(event.clientX, event.clientY);
    node.position.x = Math.max(10, point.x - state.dragOffset.x);
    node.position.y = Math.max(10, point.y - state.dragOffset.y);
    renderCanvas();
    renderProperties();
  }
  if (state.connectionDraft) {
    state.connectionDraft.cursor = getViewportPoint(event.clientX, event.clientY);
    renderEdges();
  }
}

function handlePointerUp() {
  if (state.panelResize) {
    state.panelResize.handle?.classList.remove("active");
    document.body.classList.remove("is-resizing-panels-x", "is-resizing-panels-y");
    state.panelResize = null;
    applyPanelSizes(true);
  }

  state.dragNodeId = null;
  if (state.connectionDraft) {
    state.connectionDraft = null;
    renderEdges();
  }
}

function startConnectionDraft(event) {
  event.stopPropagation();
  const sourceId = event.target.dataset.nodeId;
  state.connectionDraft = {
    sourceId,
    cursor: getViewportPoint(event.clientX, event.clientY),
  };
}

function finishConnectionDraft(event) {
  event.stopPropagation();
  if (!state.connectionDraft) return;
  const targetId = event.target.dataset.nodeId;
  if (!targetId || targetId === state.connectionDraft.sourceId) {
    state.connectionDraft = null;
    renderEdges();
    return;
  }
  const exists = state.workflow.edges.some(
    (edge) => edge.source === state.connectionDraft.sourceId && edge.target === targetId,
  );
  if (!exists) {
    state.workflow.edges.push({
      id: uid("e"),
      source: state.connectionDraft.sourceId,
      target: targetId,
    });
  }
  state.connectionDraft = null;
  renderCanvas();
}

function deleteSelected() {
  if (state.selectedNodeId) {
    state.workflow.nodes = state.workflow.nodes.filter((node) => node.id !== state.selectedNodeId);
    state.workflow.edges = state.workflow.edges.filter(
      (edge) => edge.source !== state.selectedNodeId && edge.target !== state.selectedNodeId,
    );
    state.selectedNodeId = null;
  } else if (state.selectedEdgeId) {
    state.workflow.edges = state.workflow.edges.filter((edge) => edge.id !== state.selectedEdgeId);
    state.selectedEdgeId = null;
  }
  renderCanvas();
  renderProperties();
}

async function loadCatalog() {
  const data = await api("/api/catalog");
  state.catalog = data.catalog;
  state.catalogMap = Object.fromEntries(data.catalog.map((item) => [item.type, item]));
  state.runtimeContext = data.context;
  if (!window.Vue && dom.contextRepoRoot) {
    dom.contextRepoRoot.textContent = data.context.repo_root;
  }
  renderPalette();
}

async function loadDefaultWorkflow() {
  const data = await api("/api/preset/default");
  state.workflow = data.workflow;
  state.selectedNodeId = null;
  state.selectedEdgeId = null;
  state.runData = null;
  state.activeRunId = null;
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  renderCanvas();
  renderProperties();
  renderLogs();
}

async function loadSavedWorkflowList() {
  const data = await api("/api/workflows");
  state.savedWorkflows = data.workflows;
  renderSavedWorkflows();
}

async function loadWorkflowByName(name) {
  if (!name) return;
  const data = await api(`/api/workflows/${encodeURIComponent(name)}`);
  state.workflow = data.workflow;
  state.selectedNodeId = null;
  state.selectedEdgeId = null;
  state.runData = null;
  state.activeRunId = null;
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  renderCanvas();
  renderProperties();
  renderLogs();
}

async function saveWorkflow() {
  const filename = window.prompt("输入保存文件名", `${state.workflow.name || "workflow"}.json`);
  if (!filename) return;
  await api("/api/workflows/save", {
    method: "POST",
    body: JSON.stringify({ filename, workflow: state.workflow }),
  });
  await loadSavedWorkflowList();
}

function exportWorkflow() {
  const blob = new Blob([JSON.stringify(state.workflow, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${state.workflow.name || "workflow"}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

async function runWorkflow() {
  const data = await api("/api/runs", {
    method: "POST",
    body: JSON.stringify({ workflow: state.workflow }),
  });
  state.activeRunId = data.run_id;
  state.runData = { status: "running", logs: [], node_statuses: {}, cancel_requested: false };
  renderLogs();
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
  }
  state.pollTimer = setInterval(pollRun, 700);
  await pollRun();
}

async function stopWorkflow() {
  if (!state.activeRunId || state.runData?.status !== "running") {
    return;
  }
  await api(`/api/runs/${state.activeRunId}/cancel`, {
    method: "POST",
    body: JSON.stringify({}),
  });
  await pollRun();
}

async function pollRun() {
  if (!state.activeRunId) return;
  const data = await api(`/api/runs/${state.activeRunId}`);
  state.runData = data.run;
  renderCanvas();
  renderLogs();
  if (["completed", "failed", "cancelled"].includes(state.runData.status) && state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function bindEvents() {
  dom.viewport.addEventListener("dragover", (event) => event.preventDefault());
  dom.viewport.addEventListener("drop", (event) => {
    event.preventDefault();
    const nodeType = event.dataTransfer.getData("text/plain");
    const point = getViewportPoint(event.clientX, event.clientY);
    addNode(nodeType, {
      x: point.x - NODE_SIZE.width / 2,
      y: point.y - NODE_SIZE.height / 2,
    });
  });

  dom.viewport.addEventListener(
    "wheel",
    (event) => {
      if (!event.ctrlKey) {
        return;
      }
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1 + ZOOM_STEP : 1 - ZOOM_STEP;
      setZoom(state.zoom * factor, { clientX: event.clientX, clientY: event.clientY });
    },
    { passive: false },
  );

  [dom.leftResizeHandle, dom.rightResizeHandle, dom.bottomResizeHandle].forEach((handle) => {
    handle.addEventListener("pointerdown", startPanelResize);
    handle.addEventListener("keydown", (event) => {
      const side = handle.dataset.resizeSide;
      const isHorizontal = side === "bottom";
      const validKeys = isHorizontal ? ["ArrowUp", "ArrowDown"] : ["ArrowLeft", "ArrowRight"];
      if (!validKeys.includes(event.key)) {
        return;
      }
      event.preventDefault();
      if (isHorizontal) {
        const delta = event.key === "ArrowDown" ? 16 : -16;
        nudgePanelSize(side, delta);
        return;
      }
      const isLeft = side === "left";
      const delta = event.key === "ArrowRight"
        ? (isLeft ? 16 : -16)
        : (isLeft ? -16 : 16);
      nudgePanelSize(side, delta);
    });
  });

  window.addEventListener("pointermove", handlePointerMove);
  window.addEventListener("pointerup", handlePointerUp);
  window.addEventListener("resize", () => applyPanelSizes(false));

  dom.loadPresetBtn.addEventListener("click", loadDefaultWorkflow);
  dom.saveWorkflowBtn.addEventListener("click", saveWorkflow);
  dom.exportWorkflowBtn.addEventListener("click", exportWorkflow);
  dom.deleteSelectedBtn.addEventListener("click", deleteSelected);
  dom.stopWorkflowBtn.addEventListener("click", stopWorkflow);
  dom.runWorkflowBtn.addEventListener("click", runWorkflow);
  dom.loadSavedWorkflowBtn.addEventListener("click", () => loadWorkflowByName(dom.savedWorkflowSelect.value));
  dom.refreshWorkflowsBtn.addEventListener("click", loadSavedWorkflowList);
  dom.zoomOutBtn.addEventListener("click", () => setZoom(state.zoom - ZOOM_STEP));
  dom.zoomInBtn.addEventListener("click", () => setZoom(state.zoom + ZOOM_STEP));
  dom.zoomResetBtn.addEventListener("click", () => setZoom(1));
  dom.zoomFitBtn.addEventListener("click", fitCanvasToViewport);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Delete") {
      deleteSelected();
    }
    if ((event.ctrlKey || event.metaKey) && event.key === "0") {
      event.preventDefault();
      setZoom(1);
    }
  });
}

async function init() {
  loadPanelSizes();
  applyPanelSizes(false);
  bindEvents();
  await loadCatalog();
  await loadSavedWorkflowList();
  await loadDefaultWorkflow();
  applyZoom();
}

init().catch((error) => {
  console.error(error);
  if (dom.logPanel) {
    dom.logPanel.textContent = String(error);
  }
});

}
