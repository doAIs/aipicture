<template>
  <div class="model-training-view">
    <div class="page-header">
      <h1>Model Training</h1>
      <p>Train custom AI models on your own datasets</p>
    </div>

    <el-tabs v-model="activeTab" class="training-tabs">
      <el-tab-pane label="New Training" name="new">
        <div class="content-grid">
          <div class="config-panel">
            <div class="panel-card">
              <h3>Model Type</h3>
              <el-select v-model="config.modelType" placeholder="Select model type" class="full-width">
                <el-option label="Image Classifier" value="image_classifier" />
                <el-option label="Object Detector (YOLO)" value="object_detector" />
                <el-option label="Text-to-Image (LoRA)" value="text_to_image" />
              </el-select>
            </div>

            <div class="panel-card">
              <h3>Base Model</h3>
              <el-select v-model="config.baseModel" placeholder="Select base model" class="full-width">
                <el-option 
                  v-for="model in baseModels" 
                  :key="model.id" 
                  :label="model.name" 
                  :value="model.id"
                />
              </el-select>
            </div>

            <div class="panel-card">
              <h3>Dataset</h3>
              <el-upload
                drag
                :auto-upload="false"
                :on-change="handleDatasetUpload"
                accept=".zip,.tar,.tar.gz"
                class="dataset-upload"
              >
                <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
                <div class="el-upload__text">
                  Drop dataset (ZIP/TAR) or <em>click to upload</em>
                </div>
              </el-upload>
              
              <div v-if="datasetInfo" class="dataset-info">
                <el-icon><Folder /></el-icon>
                <span>{{ datasetInfo.name }} ({{ datasetInfo.samples }} samples)</span>
              </div>
            </div>

            <div class="panel-card">
              <h3>Training Parameters</h3>
              
              <div class="param-row">
                <label>Epochs</label>
                <el-input-number v-model="config.epochs" :min="1" :max="100" />
              </div>
              
              <div class="param-row">
                <label>Batch Size</label>
                <el-input-number v-model="config.batchSize" :min="1" :max="64" />
              </div>
              
              <div class="param-row">
                <label>Learning Rate</label>
                <el-input-number v-model="config.learningRate" :min="0.00001" :max="0.1" :step="0.0001" :precision="5" />
              </div>
              
              <div class="param-row">
                <label>Save Steps</label>
                <el-input-number v-model="config.saveSteps" :min="100" :max="10000" :step="100" />
              </div>
            </div>

            <el-button 
              type="primary" 
              size="large"
              :loading="isStarting"
              :disabled="!canStart"
              class="start-btn"
              @click="startTraining"
            >
              <el-icon><VideoPlay /></el-icon>
              Start Training
            </el-button>
          </div>

          <div class="preview-panel">
            <div class="panel-card">
              <h3>Training Configuration Preview</h3>
              <pre class="config-preview">{{ configPreview }}</pre>
            </div>
          </div>
        </div>
      </el-tab-pane>

      <el-tab-pane label="Active Training" name="active">
        <div v-if="activeTasks.length > 0" class="active-training">
          <div 
            v-for="task in activeTasks" 
            :key="task.task_id"
            class="training-card"
          >
            <div class="training-header">
              <h3>{{ task.task_id }}</h3>
              <el-tag :type="getStatusType(task.status)">{{ task.status }}</el-tag>
            </div>
            
            <div class="training-progress">
              <ProgressBar 
                :percentage="task.progress" 
                :label="`Epoch ${task.current_epoch || 0}/${task.total_epochs || config.epochs}`"
                :show-stats="true"
                status="processing"
              />
            </div>
            
            <div class="training-metrics" v-if="task.metrics">
              <div class="metric">
                <span class="metric-label">Loss</span>
                <span class="metric-value">{{ task.loss?.toFixed(4) || 'N/A' }}</span>
              </div>
              <div class="metric">
                <span class="metric-label">Learning Rate</span>
                <span class="metric-value">{{ task.learning_rate?.toExponential(2) || 'N/A' }}</span>
              </div>
            </div>
            
            <div class="training-actions">
              <el-button type="danger" @click="stopTraining(task.task_id)">
                Stop Training
              </el-button>
            </div>
          </div>
        </div>
        
        <div v-else class="empty-state">
          <el-icon><DataAnalysis /></el-icon>
          <p>No active training tasks</p>
        </div>
      </el-tab-pane>

      <el-tab-pane label="History" name="history">
        <div class="history-table">
          <el-table :data="completedTasks" stripe>
            <el-table-column prop="task_id" label="Task ID" width="200" />
            <el-table-column prop="status" label="Status" width="120">
              <template #default="{ row }">
                <el-tag :type="getStatusType(row.status)">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="current_epoch" label="Epochs" width="100" />
            <el-table-column prop="loss" label="Final Loss" width="120">
              <template #default="{ row }">
                {{ row.loss?.toFixed(4) || 'N/A' }}
              </template>
            </el-table-column>
            <el-table-column prop="completed_at" label="Completed" />
            <el-table-column label="Actions" width="150">
              <template #default="{ row }">
                <el-button size="small" @click="downloadModel(row.task_id)">Download</el-button>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled, Folder, VideoPlay, DataAnalysis } from '@element-plus/icons-vue'
import { ProgressBar } from '@/components/common'
import { useTrainingStore } from '@/stores/training'
import { trainingApi, type TrainingConfig } from '@/api/endpoints/training'

const trainingStore = useTrainingStore()

const activeTab = ref('new')
const isStarting = ref(false)

const config = reactive<TrainingConfig>({
  model_type: 'image_classifier',
  base_model: '',
  epochs: 10,
  batch_size: 8,
  learning_rate: 0.0001,
  save_steps: 500
})

const datasetInfo = ref<{ name: string; samples: number } | null>(null)

const baseModels = ref([
  { id: 'resnet50', name: 'ResNet-50' },
  { id: 'efficientnet-b0', name: 'EfficientNet-B0' },
  { id: 'yolov8n', name: 'YOLOv8-Nano' },
  { id: 'yolov8s', name: 'YOLOv8-Small' },
  { id: 'sd-v1-5', name: 'Stable Diffusion v1.5' }
])

const activeTasks = computed(() => trainingStore.activeTasks)
const completedTasks = computed(() => trainingStore.completedTasks)

const canStart = computed(() => {
  return config.base_model && datasetInfo.value
})

const configPreview = computed(() => {
  return JSON.stringify({
    model_type: config.model_type,
    base_model: config.base_model,
    epochs: config.epochs,
    batch_size: config.batch_size,
    learning_rate: config.learning_rate,
    save_steps: config.save_steps,
    dataset: datasetInfo.value?.name || 'Not selected'
  }, null, 2)
})

const handleDatasetUpload = (file: any) => {
  // Simulate dataset info extraction
  datasetInfo.value = {
    name: file.name,
    samples: Math.floor(Math.random() * 10000) + 1000
  }
}

const startTraining = async () => {
  isStarting.value = true
  
  try {
    const taskId = await trainingStore.startModelTraining(config)
    if (taskId) {
      ElMessage.success('Training started!')
      activeTab.value = 'active'
    }
  } catch (error: any) {
    ElMessage.error(error.message || 'Failed to start training')
  } finally {
    isStarting.value = false
  }
}

const stopTraining = async (taskId: string) => {
  try {
    await trainingStore.stopTraining(taskId)
    ElMessage.success('Training stopped')
  } catch (error: any) {
    ElMessage.error(error.message || 'Failed to stop training')
  }
}

const downloadModel = async (taskId: string) => {
  try {
    await trainingApi.downloadModel(taskId)
    ElMessage.success('Download started')
  } catch (error: any) {
    ElMessage.error(error.message || 'Download failed')
  }
}

const getStatusType = (status: string) => {
  const types: Record<string, string> = {
    pending: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger',
    cancelled: 'info'
  }
  return types[status] || 'info'
}

onMounted(() => {
  trainingStore.fetchAllTasks()
})
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.model-training-view { padding: 24px; }

.page-header {
  margin-bottom: 32px;
  h1 { font-size: 2rem; background: $gradient-neon; -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  p { color: $text-secondary; }
}

.training-tabs {
  :deep(.el-tabs__header) { margin-bottom: 24px; }
}

.content-grid {
  display: grid;
  grid-template-columns: 1fr 400px;
  gap: 24px;
  @media (max-width: 1200px) { grid-template-columns: 1fr; }
}

.panel-card {
  @include glass-effect;
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 20px;
  h3 { font-size: 1rem; color: $text-primary; margin-bottom: 16px; }
}

.full-width { width: 100%; }

.param-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  label { font-size: 14px; color: $text-secondary; }
}

.dataset-upload { width: 100%; }

.dataset-info {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 12px;
  padding: 12px;
  background: rgba($neon-green, 0.1);
  border-radius: 8px;
  color: $neon-green;
}

.start-btn {
  width: 100%;
  height: 48px;
  font-size: 16px;
  .el-icon { margin-right: 8px; }
}

.config-preview {
  background: rgba(0, 0, 0, 0.3);
  padding: 16px;
  border-radius: 8px;
  font-family: monospace;
  font-size: 13px;
  color: $text-secondary;
  overflow-x: auto;
}

.training-card {
  @include glass-effect;
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 20px;
}

.training-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  h3 { margin: 0; font-size: 1rem; }
}

.training-progress { margin-bottom: 20px; }

.training-metrics {
  display: flex;
  gap: 24px;
  margin-bottom: 20px;
}

.metric {
  .metric-label { display: block; font-size: 12px; color: $text-muted; }
  .metric-value { font-size: 1.25rem; font-weight: 600; color: $neon-cyan; }
}

.empty-state {
  text-align: center;
  padding: 60px;
  color: $text-muted;
  .el-icon { font-size: 64px; margin-bottom: 16px; }
}

.history-table {
  @include glass-effect;
  border-radius: 16px;
  padding: 24px;
}
</style>
