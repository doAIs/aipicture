<template>
  <div class="image-to-image-view">
    <div class="page-header">
      <h1>Image to Image Transformation</h1>
      <p>Transform existing images with AI-powered editing using text guidance</p>
    </div>

    <div class="content-grid">
      <!-- Input Panel -->
      <div class="input-panel">
        <div class="panel-card">
          <h3>Source Image</h3>
          <ImageUpload
            v-model="sourceImage"
            @change="onImageChange"
          />
        </div>

        <div class="panel-card">
          <h3>Transformation Prompt</h3>
          <el-input
            v-model="prompt"
            type="textarea"
            :rows="3"
            placeholder="Describe how you want to transform the image..."
            class="prompt-input"
          />
          
          <h3>Negative Prompt</h3>
          <el-input
            v-model="negativePrompt"
            type="textarea"
            :rows="2"
            placeholder="Things to avoid (optional)..."
            class="prompt-input"
          />
        </div>

        <div class="panel-card">
          <h3>Transformation Settings</h3>
          
          <div class="setting-row">
            <label>Strength ({{ settings.strength }})</label>
            <el-slider
              v-model="settings.strength"
              :min="0.1"
              :max="1"
              :step="0.05"
            />
            <span class="setting-hint">Higher = more transformation</span>
          </div>
          
          <div class="setting-row">
            <label>Steps</label>
            <el-slider
              v-model="settings.steps"
              :min="10"
              :max="100"
              :step="5"
              show-input
            />
          </div>
          
          <div class="setting-row">
            <label>Guidance Scale</label>
            <el-slider
              v-model="settings.guidanceScale"
              :min="1"
              :max="20"
              :step="0.5"
              show-input
            />
          </div>
        </div>

        <el-button 
          type="primary" 
          size="large" 
          :loading="isProcessing"
          :disabled="!sourceImage || !prompt.trim()"
          class="generate-btn"
          @click="transformImage"
        >
          <el-icon v-if="!isProcessing"><MagicStick /></el-icon>
          {{ isProcessing ? 'Transforming...' : 'Transform Image' }}
        </el-button>
      </div>

      <!-- Output Panel -->
      <div class="output-panel">
        <div class="panel-card comparison-card">
          <h3>Comparison</h3>
          <div class="comparison-view">
            <div class="comparison-item">
              <span class="comparison-label">Original</span>
              <div class="comparison-image">
                <img v-if="sourceImagePreview" :src="sourceImagePreview" alt="Original" />
                <div v-else class="empty-placeholder">
                  <el-icon><Picture /></el-icon>
                  <span>Upload an image</span>
                </div>
              </div>
            </div>
            
            <div class="comparison-arrow">
              <el-icon><Right /></el-icon>
            </div>
            
            <div class="comparison-item">
              <span class="comparison-label">Transformed</span>
              <div class="comparison-image">
                <div v-if="isProcessing" class="processing-overlay">
                  <el-icon class="is-loading"><Loading /></el-icon>
                  <ProgressBar :percentage="progress" />
                </div>
                <img v-else-if="resultImage" :src="resultImage" alt="Transformed" />
                <div v-else class="empty-placeholder">
                  <el-icon><PictureFilled /></el-icon>
                  <span>Result will appear here</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="panel-card" v-if="resultImage">
          <h3>Actions</h3>
          <div class="result-actions">
            <el-button :icon="Download" @click="downloadResult">Download Result</el-button>
            <el-button :icon="Refresh" @click="regenerate">Regenerate</el-button>
            <el-button :icon="Switch" @click="useResultAsSource">Use as Source</el-button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { 
  MagicStick, 
  Loading, 
  Download, 
  Refresh, 
  Picture, 
  PictureFilled,
  Right,
  Switch
} from '@element-plus/icons-vue'
import { ImageUpload, ProgressBar } from '@/components/common'
import { imageToImageApi, type ImageToImageParams } from '@/api/endpoints/generation'

const sourceImage = ref<File | null>(null)
const sourceImagePreview = ref<string>('')
const prompt = ref('')
const negativePrompt = ref('')
const isProcessing = ref(false)
const progress = ref(0)
const resultImage = ref<string | null>(null)

const settings = reactive({
  strength: 0.75,
  steps: 30,
  guidanceScale: 7.5
})

const onImageChange = (file: File | null) => {
  if (file) {
    const reader = new FileReader()
    reader.onload = (e) => {
      sourceImagePreview.value = e.target?.result as string
    }
    reader.readAsDataURL(file)
  } else {
    sourceImagePreview.value = ''
  }
}

const transformImage = async () => {
  if (!sourceImage.value || !prompt.value.trim()) {
    ElMessage.warning('Please upload an image and enter a prompt')
    return
  }

  isProcessing.value = true
  progress.value = 0

  try {
    const params: ImageToImageParams = {
      image: sourceImage.value,
      prompt: prompt.value,
      negative_prompt: negativePrompt.value || undefined,
      strength: settings.strength,
      num_inference_steps: settings.steps,
      guidance_scale: settings.guidanceScale
    }

    // Simulate progress
    const progressInterval = setInterval(() => {
      if (progress.value < 90) {
        progress.value += 5
      }
    }, 200)

    const response = await imageToImageApi.generate(params)
    
    clearInterval(progressInterval)
    progress.value = 100

    if (response.image_url) {
      resultImage.value = response.image_url
      ElMessage.success('Image transformed successfully!')
    }
  } catch (error: any) {
    ElMessage.error(error.message || 'Failed to transform image')
  } finally {
    isProcessing.value = false
  }
}

const downloadResult = () => {
  if (!resultImage.value) return
  
  const link = document.createElement('a')
  link.href = resultImage.value
  link.download = `transformed_${Date.now()}.png`
  link.click()
}

const regenerate = () => {
  transformImage()
}

const useResultAsSource = () => {
  if (resultImage.value) {
    sourceImagePreview.value = resultImage.value
    resultImage.value = null
  }
}
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.image-to-image-view {
  padding: 24px;
}

.page-header {
  margin-bottom: 32px;
  
  h1 {
    font-size: 2rem;
    margin-bottom: 8px;
    background: $gradient-neon;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  
  p {
    color: $text-secondary;
    margin: 0;
  }
}

.content-grid {
  display: grid;
  grid-template-columns: 400px 1fr;
  gap: 24px;
  
  @media (max-width: 1200px) {
    grid-template-columns: 1fr;
  }
}

.panel-card {
  @include glass-effect;
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 20px;
  
  h3 {
    font-size: 1rem;
    color: $text-primary;
    margin-bottom: 16px;
    
    &:not(:first-child) {
      margin-top: 24px;
    }
  }
}

.prompt-input {
  :deep(.el-textarea__inner) {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid $glass-border;
    color: $text-primary;
    
    &:focus {
      border-color: $neon-cyan;
    }
  }
}

.setting-row {
  margin-bottom: 20px;
  
  label {
    display: block;
    font-size: 13px;
    color: $text-secondary;
    margin-bottom: 8px;
  }
  
  .setting-hint {
    display: block;
    font-size: 11px;
    color: $text-muted;
    margin-top: 4px;
  }
}

.generate-btn {
  width: 100%;
  height: 48px;
  font-size: 16px;
  
  .el-icon {
    margin-right: 8px;
  }
}

.comparison-card {
  min-height: 400px;
}

.comparison-view {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 24px;
  
  @media (max-width: 768px) {
    flex-direction: column;
  }
}

.comparison-item {
  flex: 1;
  text-align: center;
}

.comparison-label {
  display: block;
  font-size: 14px;
  color: $text-secondary;
  margin-bottom: 12px;
  font-weight: 500;
}

.comparison-image {
  aspect-ratio: 1;
  max-width: 400px;
  margin: 0 auto;
  border-radius: 12px;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid $glass-border;
  position: relative;
  
  img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }
}

.empty-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  min-height: 300px;
  color: $text-muted;
  
  .el-icon {
    font-size: 48px;
    margin-bottom: 12px;
    opacity: 0.5;
  }
}

.comparison-arrow {
  font-size: 32px;
  color: $neon-cyan;
  
  @media (max-width: 768px) {
    transform: rotate(90deg);
  }
}

.processing-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.7);
  padding: 24px;
  
  .el-icon {
    font-size: 48px;
    color: $neon-cyan;
    margin-bottom: 16px;
  }
}

.result-actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
</style>
