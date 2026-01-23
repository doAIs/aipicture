<template>
  <div class="image-recognition-view">
    <div class="page-header">
      <h1>Image Recognition</h1>
      <p>Classify and detect objects in images using YOLO and classification models</p>
    </div>

    <div class="content-grid">
      <div class="input-panel">
        <div class="panel-card">
          <h3>Upload Image</h3>
          <ImageUpload v-model="sourceImage" @change="onImageChange" />
        </div>

        <div class="panel-card">
          <h3>Recognition Mode</h3>
          <el-radio-group v-model="recognitionMode" class="mode-selector">
            <el-radio-button label="detection">Object Detection</el-radio-button>
            <el-radio-button label="classification">Classification</el-radio-button>
          </el-radio-group>
        </div>

        <div class="panel-card">
          <h3>Model Selection</h3>
          <el-select v-model="selectedModel" placeholder="Select model" class="model-select">
            <el-option 
              v-for="model in models" 
              :key="model.id" 
              :label="model.name" 
              :value="model.id"
            />
          </el-select>
          
          <div class="setting-row" v-if="recognitionMode === 'detection'">
            <label>Confidence Threshold</label>
            <el-slider v-model="confidence" :min="0.1" :max="1" :step="0.05" show-input />
          </div>
        </div>

        <el-button 
          type="primary" 
          size="large" 
          :loading="isProcessing"
          :disabled="!sourceImage"
          class="analyze-btn"
          @click="analyzeImage"
        >
          {{ isProcessing ? 'Analyzing...' : 'Analyze Image' }}
        </el-button>
      </div>

      <div class="output-panel">
        <div class="panel-card result-card">
          <h3>Results</h3>
          
          <div v-if="isProcessing" class="processing-state">
            <el-icon class="is-loading"><Loading /></el-icon>
            <p>Processing image...</p>
          </div>
          
          <div v-else-if="results.length > 0" class="results-container">
            <div class="result-image">
              <canvas ref="canvasRef" />
            </div>
            
            <div class="detections-list">
              <h4>Detected Objects</h4>
              <div 
                v-for="(result, index) in results" 
                :key="index"
                class="detection-item"
              >
                <span class="detection-label">{{ result.label }}</span>
                <div class="confidence-bar">
                  <div 
                    class="confidence-fill" 
                    :style="{ width: (result.confidence * 100) + '%' }"
                  />
                </div>
                <span class="confidence-value">{{ (result.confidence * 100).toFixed(1) }}%</span>
              </div>
            </div>
          </div>
          
          <div v-else class="empty-result">
            <el-icon><Search /></el-icon>
            <p>Upload an image and click analyze to see results</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Loading, Search } from '@element-plus/icons-vue'
import { ImageUpload } from '@/components/common'
import { imageRecognitionApi, type Detection, type Classification } from '@/api/endpoints/recognition'

interface Result {
  label: string
  confidence: number
  bbox?: [number, number, number, number]
}

const sourceImage = ref<File | null>(null)
const sourceImagePreview = ref<string>('')
const recognitionMode = ref<'detection' | 'classification'>('detection')
const selectedModel = ref('yolov8')
const confidence = ref(0.5)
const isProcessing = ref(false)
const results = ref<Result[]>([])
const canvasRef = ref<HTMLCanvasElement>()

const models = ref([
  { id: 'yolov8', name: 'YOLOv8 (Object Detection)' },
  { id: 'yolov8-seg', name: 'YOLOv8-Seg (Segmentation)' },
  { id: 'resnet50', name: 'ResNet-50 (Classification)' },
  { id: 'vit', name: 'ViT (Vision Transformer)' }
])

const onImageChange = (file: File | null) => {
  if (file) {
    const reader = new FileReader()
    reader.onload = (e) => {
      sourceImagePreview.value = e.target?.result as string
    }
    reader.readAsDataURL(file)
    results.value = []
  }
}

const analyzeImage = async () => {
  if (!sourceImage.value) return

  isProcessing.value = true
  results.value = []

  try {
    const response = recognitionMode.value === 'detection'
      ? await imageRecognitionApi.detect(sourceImage.value, selectedModel.value)
      : await imageRecognitionApi.classify(sourceImage.value, selectedModel.value)

    if (response.detections) {
      results.value = response.detections.map(d => ({
        label: d.label,
        confidence: d.confidence,
        bbox: d.bbox
      }))
      drawDetections()
    } else if (response.classifications) {
      results.value = response.classifications.map(c => ({
        label: c.label,
        confidence: c.confidence
      }))
    }

    ElMessage.success('Analysis complete!')
  } catch (error: any) {
    ElMessage.error(error.message || 'Analysis failed')
  } finally {
    isProcessing.value = false
  }
}

const drawDetections = () => {
  if (!canvasRef.value || !sourceImagePreview.value) return

  const canvas = canvasRef.value
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const img = new Image()
  img.onload = () => {
    canvas.width = img.width
    canvas.height = img.height
    ctx.drawImage(img, 0, 0)

    results.value.forEach((result, index) => {
      if (result.bbox) {
        const [x, y, w, h] = result.bbox
        const colors = ['#00d4ff', '#ff00ff', '#00ff88', '#ff6b35', '#ffd93d']
        const color = colors[index % colors.length]
        
        ctx.strokeStyle = color
        ctx.lineWidth = 3
        ctx.strokeRect(x, y, w, h)
        
        ctx.fillStyle = color
        ctx.font = '16px Arial'
        ctx.fillText(`${result.label} ${(result.confidence * 100).toFixed(0)}%`, x, y - 5)
      }
    })
  }
  img.src = sourceImagePreview.value
}
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.image-recognition-view {
  padding: 24px;
}

.page-header {
  margin-bottom: 32px;
  h1 {
    font-size: 2rem;
    background: $gradient-neon;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  p { color: $text-secondary; }
}

.content-grid {
  display: grid;
  grid-template-columns: 400px 1fr;
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

.mode-selector {
  width: 100%;
  :deep(.el-radio-button__inner) {
    width: 100%;
  }
}

.model-select {
  width: 100%;
  margin-bottom: 16px;
}

.setting-row {
  margin-top: 16px;
  label { display: block; font-size: 13px; color: $text-secondary; margin-bottom: 8px; }
}

.analyze-btn {
  width: 100%;
  height: 48px;
  font-size: 16px;
}

.result-card {
  min-height: 400px;
}

.processing-state, .empty-result {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 300px;
  color: $text-muted;
  .el-icon { font-size: 64px; margin-bottom: 16px; }
}

.results-container {
  display: grid;
  grid-template-columns: 1fr 300px;
  gap: 24px;
  @media (max-width: 992px) { grid-template-columns: 1fr; }
}

.result-image {
  canvas {
    max-width: 100%;
    border-radius: 12px;
  }
}

.detections-list {
  h4 { font-size: 14px; color: $text-secondary; margin-bottom: 16px; }
}

.detection-item {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  padding: 12px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
}

.detection-label {
  font-weight: 500;
  color: $text-primary;
  min-width: 100px;
}

.confidence-bar {
  flex: 1;
  height: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  background: $gradient-neon;
  border-radius: 4px;
}

.confidence-value {
  font-size: 13px;
  color: $neon-cyan;
  min-width: 50px;
  text-align: right;
}
</style>
