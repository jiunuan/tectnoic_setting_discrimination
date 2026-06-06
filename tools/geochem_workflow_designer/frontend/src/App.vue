<template>
  <div class="app-shell" v-cloak>
    <header class="topbar topbar-shell">
      <div class="topbar-brand">
        <p class="eyebrow">BASALT PIPELINE WORKBENCH</p>
        <h1>构造判别工作流编排台</h1>
      </div>

      <nav class="topbar-nav" aria-label="Primary Views">
        <button type="button" class="nav-chip" :class="{ active: activePage === 'designer' }" @click="setPage('designer')">
          工作流编排
        </button>
        <button type="button" class="nav-chip" :class="{ active: activePage === 'tasks' }" @click="setPage('tasks')">
          任务管理
          <span v-if="runningCount" class="nav-chip-badge">{{ runningCount }}</span>
        </button>
      </nav>

      <div class="topbar-actions" v-show="activePage === 'designer'">
        <button id="loadPresetBtn" class="primary">加载默认模板</button>
        <button id="saveWorkflowBtn">保存到项目</button>
        <button id="exportWorkflowBtn">导出 JSON</button>
        <button id="deleteSelectedBtn">删除所选</button>
        <button id="stopWorkflowBtn" class="danger" disabled>停止执行</button>
        <button id="runWorkflowBtn" class="accent">运行工作流</button>
      </div>

      <div class="topbar-actions" v-show="activePage === 'tasks'">
        <button type="button" @click="refreshTaskPageData" :disabled="runsLoading">刷新任务</button>
        <button type="button" class="primary" @click="setPage('designer')">返回编排</button>
      </div>
    </header>

    <section v-show="activePage === 'designer'" class="designer-page">
      <section class="summary-strip">
        <div class="summary-card">
          <span class="summary-label">Runtime Context</span>
          <strong id="contextRepoRoot">-</strong>
        </div>
        <div class="summary-card">
          <span class="summary-label">Preset Focus</span>
          <strong>先切分，再只在 Train 拟合</strong>
        </div>
        <div class="summary-card">
          <span class="summary-label">Current Workflow</span>
          <strong id="workflowName">default</strong>
        </div>
      </section>

      <main id="workspace" class="workspace">
        <aside class="panel left-panel">
          <div class="panel-header">
            <h2>节点库</h2>
            <p>从这里拖到画布</p>
          </div>
          <div id="palette"></div>
          <div class="panel-header secondary">
            <h2>已保存工作流</h2>
          </div>
          <div class="workflow-loader">
            <select id="savedWorkflowSelect"></select>
            <button id="loadSavedWorkflowBtn">加载</button>
            <button id="refreshWorkflowsBtn">刷新</button>
          </div>
        </aside>

        <div id="leftResizeHandle" class="panel-resizer" data-resize-side="left" role="separator" aria-orientation="vertical" aria-label="调整左侧面板宽度" tabindex="0"></div>

        <section class="canvas-panel">
          <div class="canvas-toolbar">
            <span>画布提示：拖节点、拖圆点连线、点击边或节点后可删除。</span>
            <div class="canvas-toolbar-actions">
              <div class="zoom-controls">
                <button id="zoomOutBtn" type="button" class="tool-btn">-</button>
                <button id="zoomResetBtn" type="button" class="tool-btn zoom-label">100%</button>
                <button id="zoomInBtn" type="button" class="tool-btn">+</button>
                <button id="zoomFitBtn" type="button" class="tool-btn">适配</button>
              </div>
              <div id="currentStepIndicator" class="current-step idle">
                <span class="current-step-label">当前执行</span>
                <strong id="currentStepText">等待运行</strong>
                <span id="currentStepSubtext" class="current-step-subtext">启动后会在这里显示当前节点和脚本。</span>
              </div>
              <span id="runStatusBadge" class="status-badge idle">idle</span>
            </div>
          </div>
          <div id="canvasViewport">
            <div id="canvasScaler">
              <svg id="edgeLayer"></svg>
              <div id="canvas"></div>
            </div>
          </div>
        </section>

        <div id="rightResizeHandle" class="panel-resizer" data-resize-side="right" role="separator" aria-orientation="vertical" aria-label="调整右侧面板宽度" tabindex="0"></div>

        <aside class="panel right-panel">
          <div id="rightPanelSections" class="right-panel-sections">
            <section class="right-panel-section property-section">
              <div class="panel-header">
                <h2>属性面板</h2>
                <p>选中节点后编辑参数</p>
              </div>
              <div id="propertyPanel" class="property-panel empty">选择一个节点以编辑参数。</div>
            </section>

            <div id="bottomResizeHandle" class="panel-resizer panel-resizer-horizontal" data-resize-side="bottom" role="separator" aria-orientation="horizontal" aria-label="调整属性面板和运行日志高度" tabindex="0"></div>

            <section class="right-panel-section log-section">
              <div class="panel-header secondary log-panel-header">
                <h2>运行日志</h2>
                <p>后端执行状态与报错会显示在这里</p>
              </div>
              <pre id="logPanel" class="log-panel"></pre>
            </section>
          </div>
        </aside>
      </main>
    </section>

    <section v-show="activePage === 'tasks'" class="task-manager-page">
      <div class="task-summary-grid">
        <article class="task-summary-card running">
          <span class="summary-label">Running</span>
          <strong>{{ runningCount }}</strong>
          <p>当前正在执行的工作流数量</p>
        </article>
        <article class="task-summary-card queued">
          <span class="summary-label">Tracked Runs</span>
          <strong>{{ runs.length }}</strong>
          <p>当前会话内已记录的任务总数</p>
        </article>
        <article class="task-summary-card completed">
          <span class="summary-label">Completed</span>
          <strong>{{ completedCount }}</strong>
          <p>已成功完成的工作流</p>
        </article>
        <article class="task-summary-card failed">
          <span class="summary-label">Failed / Cancelled</span>
          <strong>{{ failedOrCancelledCount }}</strong>
          <p>需要关注的失败或取消任务</p>
        </article>
      </div>

      <div class="task-manager-layout">
        <section class="panel task-panel task-list-panel">
          <div class="panel-header">
            <h2>任务列表</h2>
            <p>支持并行运行，点击任务可查看实时状态与日志。</p>
          </div>
          <div class="task-list-toolbar">
            <span class="task-list-note">自动轮询：1.2 秒</span>
            <span class="task-list-note" v-if="selectedRunSummary">当前查看：{{ selectedRunSummary.workflow_name }}</span>
          </div>
          <div class="task-list" v-if="runs.length">
            <button
              v-for="run in runs"
              :key="run.id"
              type="button"
              class="task-card"
              :class="[`status-${run.status}`, { selected: selectedRunId === run.id }]"
              @click="selectRun(run.id)"
            >
              <span class="task-card-top">
                <span class="task-card-title">{{ run.workflow_name }}</span>
                <span class="task-status-pill" :class="run.status">{{ statusLabel(run.status) }}</span>
              </span>
              <span class="task-card-meta">Run ID {{ run.id }} ? {{ formatTime(run.created_at) }}</span>
              <span class="task-card-current">
                <strong>{{ run.current_node_title || '等待节点进入运行态' }}</strong>
                <span>{{ run.current_node_status || 'idle' }}</span>
              </span>
              <span class="task-card-progress">
                <span class="task-progress-bar"><span :style="{ width: `${run.progress_percent}%` }"></span></span>
                <span>{{ run.completed_nodes }}/{{ run.total_nodes || 0 }} 节点 ? {{ run.progress_percent }}%</span>
              </span>
            </button>
          </div>
          <div v-else class="task-empty-state">当前还没有启动过工作流。去“工作流编排”页运行一个任务后，这里会开始统计。</div>
        </section>

        <section class="panel task-panel task-detail-panel">
          <template v-if="selectedRunSummary">
            <div class="task-detail-header">
              <div>
                <p class="eyebrow">RUN DETAIL</p>
                <h2>{{ selectedRunSummary.workflow_name }}</h2>
                <p class="task-detail-subtitle">{{ statusLabel(selectedRunSummary.status) }} ? Run ID {{ selectedRunSummary.id }}</p>
              </div>
              <div class="task-detail-actions">
                <button type="button" @click="setPage('designer')">切换到编排页</button>
                <button
                  type="button"
                  class="danger"
                  @click="cancelRun(selectedRunSummary.id)"
                  :disabled="selectedRunSummary.status !== 'running'"
                >
                  停止任务
                </button>
              </div>
            </div>

            <div class="task-detail-grid">
              <article>
                <label>创建时间</label>
                <strong>{{ formatTime(selectedRunSummary.created_at) }}</strong>
              </article>
              <article>
                <label>结束时间</label>
                <strong>{{ selectedRunSummary.finished_at ? formatTime(selectedRunSummary.finished_at) : '进行中' }}</strong>
              </article>
              <article>
                <label>当前节点</label>
                <strong>{{ selectedRunSummary.current_node_title || '等待启动' }}</strong>
              </article>
              <article>
                <label>节点进度</label>
                <strong>{{ selectedRunSummary.completed_nodes }}/{{ selectedRunSummary.total_nodes || 0 }}</strong>
              </article>
            </div>

            <div class="task-detail-log-block">
              <div class="task-detail-log-head">
                <span>实时日志</span>
                <span>{{ displayedLogs.length }} / {{ selectedRun?.logs?.length || 0 }}</span>
              </div>
              <pre class="task-detail-log-view"><span v-for="(line, index) in displayedLogs" :key="`${selectedRunSummary.id}-${index}`" class="task-detail-log-line">{{ line }}</span></pre>
            </div>
          </template>
          <div v-else class="task-empty-state task-empty-state-detail">选择左侧一个任务后，这里会显示实时状态、节点进度和日志。</div>
        </section>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';

const PAGE_STORAGE_KEY = 'geochem-workflow-active-page';
const activePage = ref(window.localStorage.getItem(PAGE_STORAGE_KEY) || 'designer');
const runs = ref([]);
const runsLoading = ref(false);
const selectedRunId = ref('');
const selectedRun = ref(null);
let taskPollTimer = null;

function statusLabel(status) {
  const labels = {
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
    idle: '空闲',
  };
  return labels[status] || status || '未知';
}

function formatTime(value) {
  return value ? value.replace('T', ' ') : '-';
}

function pickDefaultRunId(list) {
  return list.find((run) => run.status === 'running')?.id || list[0]?.id || '';
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

async function refreshRuns() {
  runsLoading.value = true;
  try {
    const data = await api('/api/runs');
    runs.value = data.runs || [];
    if (!runs.value.some((run) => run.id === selectedRunId.value)) {
      selectedRunId.value = pickDefaultRunId(runs.value);
    }
  } finally {
    runsLoading.value = false;
  }
}

async function refreshSelectedRun(force = false) {
  if (!selectedRunId.value) {
    selectedRun.value = null;
    return;
  }
  const summary = runs.value.find((run) => run.id === selectedRunId.value);
  const shouldFetch = force
    || !selectedRun.value
    || selectedRun.value.id !== selectedRunId.value
    || summary?.status === 'running'
    || summary?.status !== selectedRun.value?.status;

  if (!shouldFetch) {
    return;
  }

  const data = await api(`/api/runs/${encodeURIComponent(selectedRunId.value)}`);
  selectedRun.value = data.run;
}

async function refreshTaskPageData(forceDetail = false) {
  await refreshRuns();
  await refreshSelectedRun(forceDetail);
}

function startTaskPolling() {
  if (taskPollTimer) {
    clearInterval(taskPollTimer);
  }
  taskPollTimer = window.setInterval(() => {
    refreshTaskPageData().catch((error) => {
      console.error('Failed to refresh run list', error);
    });
  }, 1200);
}

async function cancelRun(runId) {
  await api(`/api/runs/${encodeURIComponent(runId)}/cancel`, {
    method: 'POST',
    body: JSON.stringify({}),
  });
  await refreshTaskPageData(true);
}

function selectRun(runId) {
  selectedRunId.value = runId;
}

function setPage(page) {
  activePage.value = page;
}

const runningCount = computed(() => runs.value.filter((run) => run.status === 'running').length);
const completedCount = computed(() => runs.value.filter((run) => run.status === 'completed').length);
const failedOrCancelledCount = computed(() => runs.value.filter((run) => ['failed', 'cancelled'].includes(run.status)).length);
const selectedRunSummary = computed(() => runs.value.find((run) => run.id === selectedRunId.value) || null);
const displayedLogs = computed(() => (selectedRun.value?.logs || []).slice(-160));

watch(activePage, (page) => {
  window.localStorage.setItem(PAGE_STORAGE_KEY, page);
  if (page === 'tasks') {
    refreshTaskPageData(true).catch((error) => {
      console.error('Failed to refresh task page', error);
    });
    return;
  }
  requestAnimationFrame(() => {
    window.dispatchEvent(new Event('resize'));
  });
});

watch(selectedRunId, () => {
  refreshSelectedRun(true).catch((error) => {
    console.error('Failed to refresh selected run', error);
  });
});

onMounted(async () => {
  const module = await import('./legacy-app.js');
  module.mountWorkflowUi();
  await refreshTaskPageData(true);
  startTaskPolling();
});

onBeforeUnmount(() => {
  if (taskPollTimer) {
    clearInterval(taskPollTimer);
  }
});
</script>
