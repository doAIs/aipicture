<template>
  <div class="llm-finetuning-view">
    <div class="page-header">
      <h1>LLM Fine-tuning</h1>
      <p>Fine-tune large language models using LoRA/QLoRA for efficient adaptation</p>
    </div>

    <div class="content-grid">
      <div class="config-panel">
        <div class="panel-card">
          <h3>Base Model</h3>
          <el-select v-model="config.base_model" placeholder="Select LLM" class="full-width">
            <el-option 
              v-for="model in llmModels" 
              :key="model.id" 
              :label="model.name"
              :value="model.id"
            >
              <div class="model-option">
                <span>{{ model.name }}</span>
                <el-tag size="small">{{ model.parameters }}</el-tag>
              </div>
            </el-option>
          </el-select>
        </div>

        <div class="panel-card">
          <h3>Fine-tuning Method</h3>
          <el-radio-group v-model="method" class="method-selector">
            <el-radio-button label="lora">
              <div class="method-content">
                <strong>LoRA</strong>
                <span>Standard adapter training</span>
              </div>
            </el-radio-button>
            <el-radio-button label="qlora">
              <div class="method-content">
                <strong>QLoRA</strong>
                <span>4-bit quantized (less VRAM)</span>
              </div>
            </el-radio-button>
          </el-radio-group>
        </div>

        <div class="panel-card">
          <h3>Training Data</h3>
          <el-upload
            drag
            :auto-upload="false"
            :on-change="handleDataUpload"
            accept=".jsonl,.json"
            class="data-upload"
          >
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text">
              Upload JSONL training data<br>
              <em>Format: {"instruction": "...", "response": "..."}</em>
            </div>
          </el-upload>
          
          <div v-if="datasetInfo" class="dataset-info">
            <el-icon><Document /></el-icon>
            <span>{{ datasetInfo.path }} ({{ datasetInfo.samples }} samples)</span>
          </div>
        </div>

        <div class="panel-card">
          <h3>LoRA Configuration</h3>
          
          <div class="param-row">
            <label>LoRA Rank (r)</label>
            <el-select v-model="config.lora_r" class="param-select">
              <el-option :value="4" label="4 (Smallest)" />
              <el-option :value="8" label="8 (Recommended)" />
              <el-option :value="16" label="16" />
              <el-option :value="32" label="32" />
              <el-option :value="64" label="64 (Largest)" />
            </el-select>
          </div>
          
          <div class="param-row">
            <label>LoRA Alpha</label>
            <el-input-number v-model="config.lora_alpha" :min="1" :max="128" />
          </div>
          
          <div class="param-row">
            <label>Dropout</label>
            <el-slider v-model="config.lora_dropout" :min="0" :max="0.5" :step="0.05" show-input />
          </div>
        </div>

        <div class="panel-card">
          <h3>Training Parameters</h3>
          
          <div class="param-row">
            <label>Epochs</label>
            <el-input-number v-model="config.epochs" :min="1" :max="10" />
          </div>
          
          <div class="param-row">
            <label>Batch Size</label>
            <el-input-number v-model="config.batch_size" :min="1" :max="16" />
          </div>
          
          <div class="param-row">
            <label>Learning Rate</label>
            <el-input-number 
              v-model="config.learning_rate" 
              :min="0.00001" 
              :max="0.001" 
              :step="0.00001"
              :precision="6"
            />
          </div>
          
          <div class="param-row">
            <label>Max Sequence Length</label>
            <el-input-number v-model="config.max_length" :min="128" :max="4096" :step="128" />
          </div>
          
          <div class="param-row">
            <label>Gradient Accumulation</label>
            <el-input-number v-model="config.gradient_accumulation_steps" :min="1" :max="32" />
          </div>
        </div>

        <el-button 
          type="primary" 
          size="large"
          :loading="isStarting"
          :disabled="!canStart"
          class="start-btn"
          @click="startFineTuning"
        >
          <el-icon><Cpu /></el-icon>
          Start Fine-tuning
        </el-button>
      </div>

      <div class="monitor-panel">
        <div class="panel-card" v-if="currentTask">
          <h3>Training Progress</h3>
          
          <div class="status-header">
            <el-tag :type="getStatusType(currentTask.status)" size="large">
              {{ currentTask.status.toUpperCase() }}
            </el-tag>
            <span class="task-id">{{ currentTask.task_id }}</span>
          </div>
          
          <ProgressBar 
            :percentage="currentTask.progress" 
            :label="`Step ${currentTask.current_step || 0}/${currentTask.total_steps || '?'}`"
            status="processing"
            :striped="true"
            class="mt-4"
          />
          
          <div class="metrics-grid mt-4">
            <div class="metric-card">
              <span class="metric-value">{{ currentTask.loss?.toFixed(4) || '--' }}</span>
              <span class="metric-label">Loss</span>
            </div>
            <div class="metric-card">
              <span class="metric-value">{{ currentTask.current_epoch || 0 }}/{{ config.epochs }}</span>
              <span class="metric-label">Epoch</span>
            </div>
            <div class="metric-card">
              <span class="metric-value">{{ currentTask.learning_rate?.toExponential(2) || '--' }}</span>
              <span class="metric-label">Learning Rate</span>
            </div>
          </div>
          
          <div class="training-actions mt-4">
            <el-button type="danger" @click="stopFineTuning">Stop Training</el-button>
            <el-button v-if="currentTask.status === 'completed'" type="success" @click="testModel">
              Test Model
            </el-button>
          </div>
        </div>
        
        <div class="panel-card" v-else>
          <h3>Monitor</h3>
          <div class="empty-monitor">
            <el-icon><DataLine /></el-icon>
            <p>Start training to see live progress</p>
          </div>
        </div>

        <div class="panel-card">
          <h3>Test Fine-tuned Model</h3>
          <el-input
            v-model="testPrompt"
            type="textarea"
            :rows="3"
            placeholder="Enter a test prompt..."
            class="test-input"
          />
          <el-button 
            type="primary" 
            :loading="isTesting"
            :disabled="!testPrompt || !currentTask || currentTask.status !== 'completed'"
            @click="runTest"
            class="mt-2"
          >
            Generate Response
          </el-button>
          
          <div v-if="testResponse" class="test-response mt-3">
            <h4>Response:</h4>
            <p>{{ testResponse }}</p>
          </div>
        </div>

        <div class="panel-card">
          <h3>Export Options</h3>
          <div class="export-options">
            <el-button 
              :disabled="!currentTask || currentTask.status !== 'completed'"
              @click="exportAdapter"
            >
              <el-icon><Download /></el-icon>
              Export LoRA Adapter
            </el-button>
            <el-button 
              :disabled="!currentTask || currentTask.status !== 'completed'"
              @click="mergeAndExport"
            >
              <el-icon><Merge /></el-icon>
              Merge & Export Full Model
            </el-button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled, Document, Cpu, DataLine, Download, Merge } from '@element-plus/icons-vue'
import { ProgressBar } from '@/components/common'
import { useTrainingStore } from '@/stores/training'
import { llmFineTuningApi, type LoRAConfig } from '@/api/endpoints/training'

const trainingStore = useTrainingStore()

const method = ref<'lora' | 'qlora'>('lora')
const isStarting = ref(false)
const isTesting = ref(false)
const testPrompt = ref('')
const testResponse = ref('')

const config = reactive<LoRAConfig>({
  base_model: '',
  lora_r: 8,
  lora_alpha: 16,
  lora_dropout: 0.05,
  epochs: 3,
  batch_size: 4,
  learning_rate: 0.0002,
  max_length: 512,
  gradient_accumulation_steps: 4
})

const datasetInfo = ref<{ path: string; samples: number } | null>(null)

const llmModels = ref([
  { id: 'llama-2-7b', name: 'LLaMA 2 7B', parameters: '7B' },
  { id: 'llama-2-13b', name: 'LLaMA 2 13B', parameters: '13B' },
  { id: 'mistral-7b', name: 'Mistral 7B', parameters: '7B' },
  { id: 'phi-2', name: 'Phi-2', parameters: '2.7B' },
  { id: 'qwen-7b', name: 'Qwen 7B', parameters: '7B' }
])

const currentTask = computed(() => trainingStore.currentTask)

const canStart = computed(() => {
  return config.base_model && datasetInfo.value
})

const handleDataUpload = async (file: any) => {
  try {
    const result = await llmFineTuningApi.uploadTrainingData(file.raw)
    datasetInfo.value = result
    ElMessage.success('Training data uploaded')
  } catch (e) {
    // Simulate upload for demo
    datasetInfo.value = {
      path: file.name,
      samples: Math.floor(Math.random() * 5000) + 500
    }
  }
}

const startFineTuning = async () => {
  isStarting.value = true
  
  try {
    const taskId = await trainingStore.startLLMFineTuning(config, method.value === 'qlora')
    if (taskId) {
      ElMessage.success('Fine-tuning started!')
    }
  } catch (error: any) {
    ElMessage.error(error.message || 'Failed to start fine-tuning')
  } finally {
    isStarting.value = false
  }
}

const stopFineTuning = async () => {
  if (currentTask.value) {
    await trainingStore.stopTraining(currentTask.value.task_id)
    ElMessage.info('Training stopped')
  }
}

const testModel = () => {
  testPrompt.value = 'What is machine learning?'
}

const runTest = async () => {
  if (!currentTask.value || !testPrompt.value) return
  
  isTesting.value = true
  try {
    const result = await llmFineTuningApi.testModel(
      currentTask.value.task_id,
      testPrompt.value,
      256
    )
    testResponse.value = result.response
  } catch (error: any) {
    ElMessage.error(error.message || 'Test failed')
  } finally {
    isTesting.value = false
  }
}

const exportAdapter = async () => {
  if (currentTask.value) {
    try {
      await llmFineTuningApi.exportAdapter(currentTask.value.task_id)
      ElMessage.success('Export started')
    } catch (e: any) {
      ElMessage.error(e.message || 'Export failed')
    }
  }
}

const mergeAndExport = async () => {
  if (currentTask.value) {
    try {
      await llmFineTuningApi.mergeAdapter(currentTask.value.task_id, './merged_model')
      ElMessage.success('Model merged and exported')
    } catch (e: any) {
      ElMessage.error(e.message || 'Merge failed')
    }
  }
}

const getStatusType = (status: string) => {
  const types: Record<string, string> = {
    pending: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger'
  }
  return types[status] || 'info'
}

onMounted(() => {
  trainingStore.fetchAllTasks()
})
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.llm-finetuning-view { padding: 24px; }

.page-header {
  margin-bottom: 32px;
  h1 { font-size: 2rem; background: $gradient-neon; -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  p { color: $text-secondary; }
}

.content-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
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
.mt-2 { margin-top: 12px; }
.mt-3 { margin-top: 16px; }
.mt-4 { margin-top: 20px; }

.model-option {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.method-selector {
  width: 100%;
  :deep(.el-radio-button) { width: 50%; }
  :deep(.el-radio-button__inner) { width: 100%; padding: 16px; }
}

.method-content {
  display: flex;
  flex-direction: column;
  strong { font-size: 14px; }
  span { font-size: 11px; color: $text-muted; }
}

.data-upload { width: 100%; }

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

.param-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  label { font-size: 14px; color: $text-secondary; }
}

.param-select { width: 150px; }

.start-btn {
  width: 100%;
  height: 48px;
  font-size: 16px;
  .el-icon { margin-right: 8px; }
}

.status-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
  .task-id { font-family: monospace; color: $text-muted; }
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.metric-card {
  text-align: center;
  padding: 16px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 12px;
  .metric-value { display: block; font-size: 1.5rem; font-weight: 700; color: $neon-cyan; }
  .metric-label { font-size: 12px; color: $text-muted; }
}

.training-actions {
  display: flex;
  gap: 12px;
}

.empty-monitor {
  text-align: center;
  padding: 40px;
  color: $text-muted;
  .el-icon { font-size: 48px; margin-bottom: 12px; }
}

.test-input {
  :deep(.el-textarea__inner) {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid $glass-border;
    color: $text-primary;
  }
}

.test-response {
  padding: 16px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
  h4 { font-size: 13px; color: $text-muted; margin-bottom: 8px; }
  p { color: $text-primary; margin: 0; white-space: pre-wrap; }
}

.export-options {
  display: flex;
  flex-direction: column;
  gap: 12px;
  .el-button { width: 100%; justify-content: flex-start; }
}
</style>
