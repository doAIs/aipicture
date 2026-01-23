<template>
  <div class="face-recognition-view">
    <div class="page-header">
      <h1>Face Recognition</h1>
      <p>Detect and recognize faces in images and live video streams</p>
    </div>

    <div class="content-grid">
      <div class="input-panel">
        <div class="panel-card">
          <h3>Input Source</h3>
          <el-radio-group v-model="sourceType" class="source-selector">
            <el-radio-button label="image">Upload Image</el-radio-button>
            <el-radio-button label="camera">Live Camera</el-radio-button>
          </el-radio-group>
          
          <template v-if="sourceType === 'image'">
            <ImageUpload v-model="sourceImage" @change="onImageChange" class="mt-3" />
          </template>
          
          <template v-else>
            <div class="camera-controls mt-3">
              <el-button 
                :type="isStreaming ? 'danger' : 'primary'"
                size="large"
                @click="toggleCamera"
              >
                <el-icon><VideoCamera /></el-icon>
                {{ isStreaming ? 'Stop Camera' : 'Start Camera' }}
              </el-button>
            </div>
          </template>
        </div>

        <div class="panel-card">
          <h3>Recognition Mode</h3>
          <el-radio-group v-model="recognitionMode">
            <el-radio label="detect">Face Detection Only</el-radio>
            <el-radio label="recognize">Face Recognition</el-radio>
          </el-radio-group>
        </div>

        <el-button 
          v-if="sourceType === 'image'"
          type="primary" 
          size="large"
          :loading="isProcessing"
          :disabled="!sourceImage"
          class="analyze-btn"
          @click="analyzeFaces"
        >
          {{ isProcessing ? 'Analyzing...' : 'Analyze Faces' }}
        </el-button>
      </div>

      <div class="main-panel">
        <div class="panel-card video-panel">
          <h3>{{ sourceType === 'camera' ? 'Live Feed' : 'Analysis Result' }}</h3>
          <div class="video-container">
            <video 
              ref="videoRef" 
              v-show="sourceType === 'camera'" 
              autoplay 
              muted 
              playsinline
            />
            <canvas ref="canvasRef" :class="{ hidden: sourceType === 'camera' && !isStreaming }" />
            
            <div v-if="!isStreaming && sourceType === 'camera'" class="camera-placeholder">
              <el-icon><VideoCamera /></el-icon>
              <p>Click "Start Camera" to begin</p>
            </div>
          </div>
          
          <div v-if="detectedFaces.length > 0" class="faces-info">
            <div class="face-count">
              <el-icon><Avatar /></el-icon>
              <span>{{ detectedFaces.length }} face(s) detected</span>
            </div>
          </div>
        </div>

        <div class="panel-card results-panel" v-if="detectedFaces.length > 0">
          <h3>Detected Faces</h3>
          <div class="faces-grid">
            <div 
              v-for="(face, index) in detectedFaces" 
              :key="index"
              class="face-card"
            >
              <div class="face-image">
                <img :src="face.thumbnail" :alt="face.name || 'Unknown'" />
              </div>
              <div class="face-info">
                <span class="face-name">{{ face.name || 'Unknown' }}</span>
                <span class="face-confidence">{{ (face.confidence * 100).toFixed(1) }}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="sidebar-panel">
        <div class="panel-card">
          <h3>Face Database</h3>
          <div class="registered-faces">
            <div 
              v-for="person in registeredFaces" 
              :key="person.id"
              class="registered-face"
            >
              <el-avatar :size="40" :src="person.thumbnail">
                {{ person.name.charAt(0) }}
              </el-avatar>
              <span class="person-name">{{ person.name }}</span>
              <el-button 
                :icon="Delete" 
                circle 
                size="small"
                @click="removeFace(person.id)"
              />
            </div>
            
            <div v-if="registeredFaces.length === 0" class="empty-faces">
              <p>No faces registered</p>
            </div>
          </div>
          
          <el-divider />
          
          <h4>Register New Face</h4>
          <el-input v-model="newFaceName" placeholder="Person's name" class="mb-2" />
          <el-button 
            type="primary" 
            :disabled="!newFaceName || !sourceImage"
            @click="registerFace"
          >
            Register Face
          </el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { VideoCamera, Avatar, Delete } from '@element-plus/icons-vue'
import { ImageUpload } from '@/components/common'
import { faceRecognitionApi } from '@/api/endpoints/recognition'

interface DetectedFace {
  id: string
  name?: string
  confidence: number
  bbox: [number, number, number, number]
  thumbnail: string
}

interface RegisteredPerson {
  id: string
  name: string
  thumbnail?: string
}

const sourceType = ref<'image' | 'camera'>('image')
const sourceImage = ref<File | null>(null)
const sourceImagePreview = ref<string>('')
const recognitionMode = ref<'detect' | 'recognize'>('recognize')
const isProcessing = ref(false)
const isStreaming = ref(false)
const detectedFaces = ref<DetectedFace[]>([])
const registeredFaces = ref<RegisteredPerson[]>([])
const newFaceName = ref('')

const videoRef = ref<HTMLVideoElement>()
const canvasRef = ref<HTMLCanvasElement>()
let stream: MediaStream | null = null
let animationId: number | null = null

const onImageChange = (file: File | null) => {
  if (file) {
    const reader = new FileReader()
    reader.onload = (e) => {
      sourceImagePreview.value = e.target?.result as string
      drawImageOnCanvas()
    }
    reader.readAsDataURL(file)
  }
  detectedFaces.value = []
}

const drawImageOnCanvas = () => {
  if (!canvasRef.value || !sourceImagePreview.value) return
  
  const canvas = canvasRef.value
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  
  const img = new Image()
  img.onload = () => {
    canvas.width = img.width
    canvas.height = img.height
    ctx.drawImage(img, 0, 0)
  }
  img.src = sourceImagePreview.value
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
    stream = await navigator.mediaDevices.getUserMedia({ 
      video: { width: 1280, height: 720 } 
    })
    
    if (videoRef.value) {
      videoRef.value.srcObject = stream
      videoRef.value.onloadedmetadata = () => {
        if (canvasRef.value && videoRef.value) {
          canvasRef.value.width = videoRef.value.videoWidth
          canvasRef.value.height = videoRef.value.videoHeight
        }
        startDetectionLoop()
      }
    }
    isStreaming.value = true
  } catch (error) {
    ElMessage.error('Failed to access camera. Please check permissions.')
  }
}

const stopCamera = () => {
  if (stream) {
    stream.getTracks().forEach(track => track.stop())
    stream = null
  }
  if (animationId) {
    cancelAnimationFrame(animationId)
    animationId = null
  }
  isStreaming.value = false
  detectedFaces.value = []
}

const startDetectionLoop = () => {
  const processFrame = async () => {
    if (!isStreaming.value || !videoRef.value || !canvasRef.value) return
    
    const canvas = canvasRef.value
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Draw video frame
    ctx.drawImage(videoRef.value, 0, 0)
    
    // Simulate face detection (replace with actual API call)
    // In production, you would send the frame to the backend
    
    animationId = requestAnimationFrame(processFrame)
  }
  
  processFrame()
}

const analyzeFaces = async () => {
  if (!sourceImage.value) return
  
  isProcessing.value = true
  detectedFaces.value = []
  
  try {
    const response = recognitionMode.value === 'detect'
      ? await faceRecognitionApi.detect(sourceImage.value)
      : await faceRecognitionApi.recognize(sourceImage.value)
    
    if (response.faces) {
      detectedFaces.value = response.faces.map(face => ({
        id: face.face_id,
        name: face.name,
        confidence: face.confidence,
        bbox: face.bbox,
        thumbnail: '' // Would be generated from bbox
      }))
      
      drawFaceBoxes()
    }
    
    ElMessage.success(`Found ${detectedFaces.value.length} face(s)`)
  } catch (error: any) {
    ElMessage.error(error.message || 'Face analysis failed')
  } finally {
    isProcessing.value = false
  }
}

const drawFaceBoxes = () => {
  if (!canvasRef.value) return
  
  const ctx = canvasRef.value.getContext('2d')
  if (!ctx) return
  
  // Redraw image first
  drawImageOnCanvas()
  
  // Then draw face boxes
  setTimeout(() => {
    detectedFaces.value.forEach((face, index) => {
      const [x, y, w, h] = face.bbox
      const colors = ['#00d4ff', '#ff00ff', '#00ff88']
      const color = colors[index % colors.length]
      
      ctx.strokeStyle = color
      ctx.lineWidth = 3
      ctx.strokeRect(x, y, w, h)
      
      ctx.fillStyle = color
      ctx.font = 'bold 16px Arial'
      const label = face.name || 'Unknown'
      ctx.fillText(label, x, y - 8)
    })
  }, 100)
}

const registerFace = async () => {
  if (!sourceImage.value || !newFaceName.value) return
  
  try {
    const response = await faceRecognitionApi.register({
      name: newFaceName.value,
      image: sourceImage.value
    })
    
    if (response.success) {
      registeredFaces.value.push({
        id: response.face_id,
        name: newFaceName.value
      })
      newFaceName.value = ''
      ElMessage.success('Face registered successfully!')
    }
  } catch (error: any) {
    ElMessage.error(error.message || 'Failed to register face')
  }
}

const removeFace = async (faceId: string) => {
  try {
    await faceRecognitionApi.deleteFace(faceId)
    registeredFaces.value = registeredFaces.value.filter(f => f.id !== faceId)
    ElMessage.success('Face removed')
  } catch (error: any) {
    ElMessage.error(error.message || 'Failed to remove face')
  }
}

const loadRegisteredFaces = async () => {
  try {
    const faces = await faceRecognitionApi.listFaces()
    registeredFaces.value = faces.map(f => ({
      id: f.id,
      name: f.name
    }))
  } catch (e) {
    // Ignore errors on initial load
  }
}

onMounted(() => {
  loadRegisteredFaces()
})

onUnmounted(() => {
  stopCamera()
})
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.face-recognition-view { padding: 24px; }

.page-header {
  margin-bottom: 32px;
  h1 { font-size: 2rem; background: $gradient-neon; -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  p { color: $text-secondary; }
}

.content-grid {
  display: grid;
  grid-template-columns: 320px 1fr 280px;
  gap: 24px;
  @media (max-width: 1400px) { grid-template-columns: 1fr; }
}

.panel-card {
  @include glass-effect;
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 20px;
  h3, h4 { font-size: 1rem; color: $text-primary; margin-bottom: 16px; }
}

.source-selector { width: 100%; }
.mt-3 { margin-top: 16px; }
.mb-2 { margin-bottom: 12px; }
.camera-controls { text-align: center; }
.analyze-btn { width: 100%; height: 48px; }

.video-container {
  position: relative;
  background: #000;
  border-radius: 12px;
  overflow: hidden;
  min-height: 400px;
  
  video, canvas {
    width: 100%;
    display: block;
    &.hidden { display: none; }
  }
}

.camera-placeholder {
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: $text-muted;
  .el-icon { font-size: 64px; margin-bottom: 16px; }
}

.faces-info {
  margin-top: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  color: $neon-green;
}

.face-count {
  display: flex;
  align-items: center;
  gap: 8px;
}

.faces-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 16px;
}

.face-card {
  text-align: center;
  padding: 12px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 12px;
  
  .face-image {
    width: 80px;
    height: 80px;
    margin: 0 auto 8px;
    border-radius: 50%;
    overflow: hidden;
    border: 2px solid $neon-cyan;
    img { width: 100%; height: 100%; object-fit: cover; }
  }
  
  .face-name { display: block; font-weight: 500; color: $text-primary; }
  .face-confidence { font-size: 12px; color: $neon-green; }
}

.registered-faces {
  max-height: 300px;
  overflow-y: auto;
}

.registered-face {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px;
  margin-bottom: 8px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
  
  .person-name { flex: 1; color: $text-primary; }
}

.empty-faces {
  text-align: center;
  padding: 20px;
  color: $text-muted;
}
</style>
