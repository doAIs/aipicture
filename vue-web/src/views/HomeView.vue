<template>
  <div class="dashboard">
    <!-- Welcome Section -->
    <section class="welcome-section">
      <div class="welcome-content">
        <h1 class="welcome-title">
          <span class="gradient-text">AI Multimedia Platform</span>
        </h1>
        <p class="welcome-subtitle">
          Professional-grade AI platform for image generation, video processing, 
          recognition, and model training
        </p>
      </div>
      <div class="system-stats">
        <div class="stat-card">
          <div class="stat-icon gpu">
            <el-icon><Monitor /></el-icon>
          </div>
          <div class="stat-info">
            <span class="stat-value">{{ systemStatus.gpuAvailable ? 'Available' : 'N/A' }}</span>
            <span class="stat-label">GPU Status</span>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon models">
            <el-icon><Cpu /></el-icon>
          </div>
          <div class="stat-info">
            <span class="stat-value">{{ loadedModelsCount }}</span>
            <span class="stat-label">Loaded Models</span>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon tasks">
            <el-icon><DataAnalysis /></el-icon>
          </div>
          <div class="stat-info">
            <span class="stat-value">{{ activeTasks }}</span>
            <span class="stat-label">Active Tasks</span>
          </div>
        </div>
      </div>
    </section>

    <!-- Features Grid -->
    <section class="features-section">
      <h2 class="section-title">Features</h2>
      <div class="features-grid">
        <!-- Generation Features -->
        <div class="feature-card" @click="navigateTo('/text-to-image')">
          <div class="feature-icon generation">
            <el-icon><Picture /></el-icon>
          </div>
          <div class="feature-content">
            <h3>Text to Image</h3>
            <p>Generate stunning images from text descriptions using Stable Diffusion</p>
          </div>
          <div class="feature-arrow">
            <el-icon><ArrowRight /></el-icon>
          </div>
        </div>

        <div class="feature-card" @click="navigateTo('/image-to-image')">
          <div class="feature-icon generation">
            <el-icon><PictureFilled /></el-icon>
          </div>
          <div class="feature-content">
            <h3>Image to Image</h3>
            <p>Transform and enhance images with AI-powered editing</p>
          </div>
          <div class="feature-arrow">
            <el-icon><ArrowRight /></el-icon>
          </div>
        </div>

        <div class="feature-card" @click="navigateTo('/text-to-video')">
          <div class="feature-icon video">
            <el-icon><VideoCamera /></el-icon>
          </div>
          <div class="feature-content">
            <h3>Text to Video</h3>
            <p>Create videos from text descriptions using AI video generation</p>
          </div>
          <div class="feature-arrow">
            <el-icon><ArrowRight /></el-icon>
          </div>
        </div>

        <div class="feature-card" @click="navigateTo('/image-recognition')">
          <div class="feature-icon recognition">
            <el-icon><Search /></el-icon>
          </div>
          <div class="feature-content">
            <h3>Image Recognition</h3>
            <p>Classify and detect objects in images using YOLO and other models</p>
          </div>
          <div class="feature-arrow">
            <el-icon><ArrowRight /></el-icon>
          </div>
        </div>

        <div class="feature-card" @click="navigateTo('/face-recognition')">
          <div class="feature-icon recognition">
            <el-icon><Avatar /></el-icon>
          </div>
          <div class="feature-content">
            <h3>Face Recognition</h3>
            <p>Detect and recognize faces in images and live video streams</p>
          </div>
          <div class="feature-arrow">
            <el-icon><ArrowRight /></el-icon>
          </div>
        </div>

        <div class="feature-card" @click="navigateTo('/audio-processing')">
          <div class="feature-icon audio">
            <el-icon><Microphone /></el-icon>
          </div>
          <div class="feature-content">
            <h3>Audio Processing</h3>
            <p>Speech-to-text transcription and text-to-speech synthesis</p>
          </div>
          <div class="feature-arrow">
            <el-icon><ArrowRight /></el-icon>
          </div>
        </div>

        <div class="feature-card" @click="navigateTo('/model-training')">
          <div class="feature-icon training">
            <el-icon><SetUp /></el-icon>
          </div>
          <div class="feature-content">
            <h3>Model Training</h3>
            <p>Train custom AI models on your own datasets</p>
          </div>
          <div class="feature-arrow">
            <el-icon><ArrowRight /></el-icon>
          </div>
        </div>

        <div class="feature-card" @click="navigateTo('/llm-finetuning')">
          <div class="feature-icon training">
            <el-icon><MagicStick /></el-icon>
          </div>
          <div class="feature-content">
            <h3>LLM Fine-tuning</h3>
            <p>Fine-tune large language models with LoRA/QLoRA</p>
          </div>
          <div class="feature-arrow">
            <el-icon><ArrowRight /></el-icon>
          </div>
        </div>
      </div>
    </section>

    <!-- Quick Actions -->
    <section class="quick-actions-section">
      <h2 class="section-title">Quick Actions</h2>
      <div class="quick-actions">
        <el-button type="primary" size="large" @click="navigateTo('/text-to-image')">
          <el-icon><Picture /></el-icon>
          Generate Image
        </el-button>
        <el-button size="large" @click="navigateTo('/face-recognition')">
          <el-icon><Avatar /></el-icon>
          Start Camera
        </el-button>
        <el-button size="large" @click="navigateTo('/model-training')">
          <el-icon><SetUp /></el-icon>
          Train Model
        </el-button>
        <el-button size="large" @click="navigateTo('/audio-processing')">
          <el-icon><Microphone /></el-icon>
          Transcribe Audio
        </el-button>
      </div>
    </section>

    <!-- Recent Activity (placeholder) -->
    <section class="recent-section">
      <h2 class="section-title">Recent Activity</h2>
      <div class="recent-list">
        <div class="recent-empty">
          <el-icon><DocumentCopy /></el-icon>
          <p>No recent activity yet. Start generating or training to see your history here.</p>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useModelStore } from '@/stores/model'
import { useTrainingStore } from '@/stores/training'
import {
  Monitor,
  Cpu,
  DataAnalysis,
  Picture,
  PictureFilled,
  VideoCamera,
  Search,
  Avatar,
  Microphone,
  SetUp,
  MagicStick,
  ArrowRight,
  DocumentCopy
} from '@element-plus/icons-vue'

const router = useRouter()
const modelStore = useModelStore()
const trainingStore = useTrainingStore()

const systemStatus = ref({
  gpuAvailable: true
})
const loadedModelsCount = ref(0)
const activeTasks = ref(0)

const navigateTo = (path: string) => {
  router.push(path)
}

onMounted(async () => {
  // Fetch system status
  await modelStore.fetchSystemStatus()
  systemStatus.value = modelStore.systemStatus
  loadedModelsCount.value = modelStore.loadedModels.length
  
  // Fetch training tasks
  await trainingStore.fetchAllTasks()
  activeTasks.value = trainingStore.activeTasks.length
})
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.dashboard {
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
}

// Welcome Section
.welcome-section {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 40px;
  flex-wrap: wrap;
  gap: 24px;
}

.welcome-content {
  max-width: 600px;
}

.welcome-title {
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 12px;
  
  .gradient-text {
    background: $gradient-neon;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
}

.welcome-subtitle {
  font-size: 1.1rem;
  color: $text-secondary;
  line-height: 1.6;
}

.system-stats {
  display: flex;
  gap: 16px;
}

.stat-card {
  @include glass-effect;
  border-radius: 12px;
  padding: 16px 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 160px;
}

.stat-icon {
  width: 44px;
  height: 44px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  
  .el-icon {
    font-size: 22px;
  }
  
  &.gpu {
    background: rgba($neon-green, 0.15);
    color: $neon-green;
  }
  
  &.models {
    background: rgba($neon-cyan, 0.15);
    color: $neon-cyan;
  }
  
  &.tasks {
    background: rgba($neon-purple, 0.15);
    color: $neon-purple;
  }
}

.stat-info {
  display: flex;
  flex-direction: column;
}

.stat-value {
  font-size: 1.25rem;
  font-weight: 700;
  color: $text-primary;
}

.stat-label {
  font-size: 0.8rem;
  color: $text-muted;
}

// Section Title
.section-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: $text-primary;
  margin-bottom: 20px;
  position: relative;
  display: inline-block;
  
  &::after {
    content: '';
    position: absolute;
    bottom: -6px;
    left: 0;
    width: 40px;
    height: 3px;
    background: $gradient-neon;
    border-radius: 2px;
  }
}

// Features Section
.features-section {
  margin-bottom: 40px;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 20px;
}

.feature-card {
  @include glass-effect;
  border-radius: 16px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-4px);
    border-color: rgba($neon-cyan, 0.3);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3),
                0 0 20px rgba($neon-cyan, 0.1);
    
    .feature-arrow {
      opacity: 1;
      transform: translateX(0);
    }
  }
}

.feature-icon {
  width: 52px;
  height: 52px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  
  .el-icon {
    font-size: 26px;
  }
  
  &.generation {
    background: linear-gradient(135deg, rgba($neon-cyan, 0.2), rgba($neon-magenta, 0.2));
    color: $neon-cyan;
  }
  
  &.video {
    background: linear-gradient(135deg, rgba($neon-purple, 0.2), rgba($neon-magenta, 0.2));
    color: $neon-purple;
  }
  
  &.recognition {
    background: linear-gradient(135deg, rgba($neon-green, 0.2), rgba($neon-cyan, 0.2));
    color: $neon-green;
  }
  
  &.audio {
    background: linear-gradient(135deg, rgba($neon-orange, 0.2), rgba($neon-yellow, 0.2));
    color: $neon-orange;
  }
  
  &.training {
    background: linear-gradient(135deg, rgba($neon-magenta, 0.2), rgba($neon-purple, 0.2));
    color: $neon-magenta;
  }
}

.feature-content {
  flex: 1;
  
  h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: $text-primary;
    margin-bottom: 4px;
  }
  
  p {
    font-size: 0.85rem;
    color: $text-muted;
    margin: 0;
    line-height: 1.4;
  }
}

.feature-arrow {
  color: $neon-cyan;
  opacity: 0;
  transform: translateX(-10px);
  transition: all 0.3s ease;
  
  .el-icon {
    font-size: 20px;
  }
}

// Quick Actions
.quick-actions-section {
  margin-bottom: 40px;
}

.quick-actions {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  
  .el-button {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 24px;
    font-size: 15px;
  }
}

// Recent Section
.recent-section {
  margin-bottom: 24px;
}

.recent-list {
  @include glass-effect;
  border-radius: 16px;
  padding: 24px;
  min-height: 150px;
}

.recent-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: $text-muted;
  text-align: center;
  padding: 40px;
  
  .el-icon {
    font-size: 48px;
    margin-bottom: 16px;
    opacity: 0.5;
  }
  
  p {
    margin: 0;
    max-width: 400px;
  }
}

// Responsive
@media (max-width: 992px) {
  .welcome-section {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .system-stats {
    width: 100%;
    justify-content: space-between;
  }
}

@media (max-width: 768px) {
  .welcome-title {
    font-size: 2rem;
  }
  
  .system-stats {
    flex-direction: column;
  }
  
  .stat-card {
    width: 100%;
  }
  
  .features-grid {
    grid-template-columns: 1fr;
  }
  
  .quick-actions {
    flex-direction: column;
    
    .el-button {
      width: 100%;
    }
  }
}
</style>
