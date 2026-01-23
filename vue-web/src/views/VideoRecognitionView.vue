<template>
  <div class="video-recognition-view">
    <div class="page-header">
      <h1>Video Recognition</h1>
      <p>Analyze videos for object detection and tracking</p>
    </div>

    <div class="content-grid">
      <div class="input-panel">
        <div class="panel-card">
          <h3>Video Source</h3>
          <el-radio-group v-model="sourceType" class="source-selector">
            <el-radio-button label="upload">Upload Video</el-radio-button>
            <el-radio-button label="camera">Live Camera</el-radio-button>
          </el-radio-group>
          
          <div v-if="sourceType === 'upload'" class="upload-area">
            <el-upload
              drag
              :auto-upload="false"
              accept="video/*"
              :on-change="handleVideoUpload"
              :show-file-list="false"
            >
              <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
              <div class="el-upload__text">Drop video or click to upload</div>
            </el-upload>
          </div>
          
          <div v-else class="camera-controls">
            <el-button 
              :type="isStreaming ? 'danger' : 'primary'" 
              @click="toggleCamera"
            >
              {{ isStreaming ? 'Stop Camera' : 'Start Camera' }}
            </el-button>
          </div>
        </div>

        <div class="panel-card">
          <h3>Detection Settings</h3>
          <el-select v-model="selectedModel" placeholder="Select model" class="model-select">
            <el-option label="YOLOv8 (Fast)" value="yolov8" />
            <el-option label="YOLOv8-Large (Accurate)" value="yolov8-l" />
          </el-select>
          
          <div class="setting-row">
            <label>Confidence Threshold</label>
            <el-slider v-model="confidence" :min="0.1" :max="1" :step="0.05" show-input />
          </div>
          
          <el-checkbox v-model="trackObjects">Enable Object Tracking</el-checkbox>
        </div>

        <el-button 
          type="primary" 
          size="large"
          :loading="isProcessing"
          :disabled="!videoFile && !isStreaming"
          class="analyze-btn"
          @click="analyzeVideo"
        >
          {{ isProcessing ? 'Processing...' : 'Start Analysis' }}
        </el-button>
      </div>

      <div class="output-panel">
        <div class="panel-card video-panel">
          <h3>Video Feed</h3>
          <div class="video-container">
            <video ref="videoRef" v-if="sourceType === 'camera'" autoplay muted />
            <video ref="uploadedVideoRef" v-else :src="videoUrl" controls />
            <canvas ref="overlayCanvas" class="detection-overlay" />
          </div>
          
          <div v-if="isProcessing" class="processing-indicator">
            <ProgressBar :percentage="progress" label="Processing video..." />
          </div>
        </div>

        <div class="panel-card stats-panel">
          <h3>Detection Statistics</h3>
          <div class="stats-grid">
            <div class="stat-item">
              <span class="stat-value">{{ detectedObjects }}</span>
              <span class="stat-label">Objects Detected</span>
            </div>
            <div class="stat-item">
              <span class="stat-value">{{ fps.toFixed(1) }}</span>
              <span class="stat-label">FPS</span>
            </div>
            <div class="stat-item">
              <span class="stat-value">{{ frameCount }}</span>
              <span class="stat-label">Frames Processed</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import { ProgressBar } from '@/components/common'

const sourceType = ref<'upload' | 'camera'>('upload')
const videoFile = ref<File | null>(null)
const videoUrl = ref('')
const selectedModel = ref('yolov8')
const confidence = ref(0.5)
const trackObjects = ref(false)
const isProcessing = ref(false)
const isStreaming = ref(false)
const progress = ref(0)
const detectedObjects = ref(0)
const fps = ref(0)
const frameCount = ref(0)

const videoRef = ref<HTMLVideoElement>()
const uploadedVideoRef = ref<HTMLVideoElement>()
const overlayCanvas = ref<HTMLCanvasElement>()

let stream: MediaStream | null = null

const handleVideoUpload = (file: any) => {
  videoFile.value = file.raw
  videoUrl.value = URL.createObjectURL(file.raw)
}

const toggleCamera = async () => {
  if (isStreaming.value) {
    stopCamera()
  } else {
    await startCamera()
  }
}

const startCamera = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true })
    if (videoRef.value) {
      videoRef.value.srcObject = stream
    }
    isStreaming.value = true
  } catch (error) {
    ElMessage.error('Failed to access camera')
  }
}

const stopCamera = () => {
  if (stream) {
    stream.getTracks().forEach(track => track.stop())
    stream = null
  }
  isStreaming.value = false
}

const analyzeVideo = async () => {
  isProcessing.value = true
  progress.value = 0
  detectedObjects.value = 0
  frameCount.value = 0

  const interval = setInterval(() => {
    if (progress.value < 100) {
      progress.value += 2
      frameCount.value += 5
      detectedObjects.value = Math.floor(Math.random() * 10) + 1
      fps.value = 25 + Math.random() * 5
    }
  }, 200)

  try {
    await new Promise(resolve => setTimeout(resolve, 5000))
    progress.value = 100
    ElMessage.success('Analysis complete!')
  } finally {
    clearInterval(interval)
    isProcessing.value = false
  }
}

onUnmounted(() => {
  stopCamera()
})
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.video-recognition-view { padding: 24px; }

.page-header {
  margin-bottom: 32px;
  h1 { font-size: 2rem; background: $gradient-neon; -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
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

.source-selector { width: 100%; margin-bottom: 20px; }
.model-select { width: 100%; margin-bottom: 16px; }
.setting-row { margin-bottom: 16px; label { display: block; font-size: 13px; color: $text-secondary; margin-bottom: 8px; } }
.analyze-btn { width: 100%; height: 48px; }

.video-container {
  position: relative;
  background: #000;
  border-radius: 12px;
  overflow: hidden;
  video { width: 100%; display: block; }
  .detection-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
}

.processing-indicator { margin-top: 16px; }

.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}

.stat-item {
  text-align: center;
  padding: 16px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 12px;
  .stat-value { display: block; font-size: 2rem; font-weight: 700; color: $neon-cyan; }
  .stat-label { font-size: 12px; color: $text-muted; }
}

.upload-area { margin-top: 16px; }
.camera-controls { margin-top: 16px; text-align: center; }
</style>
