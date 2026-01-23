<template>
  <div class="video-generation-view">
    <div class="page-header">
      <h1>{{ pageTitle }}</h1>
      <p>{{ pageDescription }}</p>
    </div>

    <div class="content-grid">
      <div class="input-panel">
        <div class="panel-card">
          <h3>{{ inputType === 'text' ? 'Prompt' : 'Source Media' }}</h3>
          
          <template v-if="inputType === 'text'">
            <el-input
              v-model="prompt"
              type="textarea"
              :rows="4"
              placeholder="Describe the video you want to generate..."
            />
          </template>
          
          <template v-else-if="inputType === 'image'">
            <ImageUpload v-model="sourceFile" />
          </template>
          
          <template v-else>
            <el-upload
              drag
              :auto-upload="false"
              accept="video/*"
              :on-change="handleVideoUpload"
            >
              <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
              <div class="el-upload__text">Drop video here or click to upload</div>
            </el-upload>
          </template>
        </div>

        <div class="panel-card">
          <h3>Video Settings</h3>
          <div class="setting-row">
            <label>Duration (frames)</label>
            <el-slider v-model="settings.numFrames" :min="8" :max="64" :step="8" show-input />
          </div>
          <div class="setting-row">
            <label>FPS</label>
            <el-slider v-model="settings.fps" :min="8" :max="30" :step="1" show-input />
          </div>
          <div class="setting-row">
            <label>Steps</label>
            <el-slider v-model="settings.steps" :min="10" :max="50" :step="5" show-input />
          </div>
        </div>

        <el-button 
          type="primary" 
          size="large" 
          :loading="isGenerating"
          class="generate-btn"
          @click="generateVideo"
        >
          {{ isGenerating ? 'Generating...' : 'Generate Video' }}
        </el-button>
      </div>

      <div class="output-panel">
        <div class="panel-card">
          <h3>Generated Video</h3>
          <div v-if="isGenerating" class="generating-state">
            <ProgressBar :percentage="progress" label="Generating video..." />
          </div>
          <div v-else-if="resultVideo" class="video-result">
            <VideoPlayer :src="resultVideo" />
          </div>
          <div v-else class="empty-result">
            <el-icon><VideoCamera /></el-icon>
            <p>Your generated video will appear here</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { VideoCamera, UploadFilled } from '@element-plus/icons-vue'
import { ImageUpload, VideoPlayer, ProgressBar } from '@/components/common'

const route = useRoute()

const pageConfig = computed(() => {
  const path = route.path
  if (path.includes('text-to-video')) {
    return { title: 'Text to Video', desc: 'Create videos from text descriptions', input: 'text' }
  } else if (path.includes('image-to-video')) {
    return { title: 'Image to Video', desc: 'Animate images into videos', input: 'image' }
  } else {
    return { title: 'Video to Video', desc: 'Transform videos with AI', input: 'video' }
  }
})

const pageTitle = computed(() => pageConfig.value.title)
const pageDescription = computed(() => pageConfig.value.desc)
const inputType = computed(() => pageConfig.value.input)

const prompt = ref('')
const sourceFile = ref<File | null>(null)
const isGenerating = ref(false)
const progress = ref(0)
const resultVideo = ref<string | null>(null)

const settings = reactive({
  numFrames: 24,
  fps: 8,
  steps: 25
})

const handleVideoUpload = (file: any) => {
  sourceFile.value = file.raw
}

const generateVideo = async () => {
  isGenerating.value = true
  progress.value = 0

  const interval = setInterval(() => {
    if (progress.value < 90) progress.value += 2
  }, 300)

  try {
    await new Promise(resolve => setTimeout(resolve, 5000))
    progress.value = 100
    resultVideo.value = '/demo-video.mp4'
    ElMessage.success('Video generated!')
  } catch (e: any) {
    ElMessage.error(e.message || 'Generation failed')
  } finally {
    clearInterval(interval)
    isGenerating.value = false
  }
}
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.video-generation-view {
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

.setting-row {
  margin-bottom: 20px;
  label { display: block; font-size: 13px; color: $text-secondary; margin-bottom: 8px; }
}

.generate-btn {
  width: 100%;
  height: 48px;
  font-size: 16px;
}

.generating-state, .empty-result {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 300px;
  color: $text-muted;
  .el-icon { font-size: 64px; margin-bottom: 16px; opacity: 0.5; }
}

.video-result {
  max-width: 800px;
  margin: 0 auto;
}
</style>
