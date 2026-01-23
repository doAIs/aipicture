<template>
  <div class="text-to-image-view">
    <div class="page-header">
      <h1>Text to Image Generation</h1>
      <p>Generate stunning images from text descriptions using Stable Diffusion</p>
    </div>

    <div class="content-grid">
      <!-- Input Panel -->
      <div class="input-panel">
        <div class="panel-card">
          <h3>Prompt</h3>
          <el-input
            v-model="prompt"
            type="textarea"
            :rows="4"
            placeholder="Describe the image you want to generate..."
            class="prompt-input"
          />
          
          <h3>Negative Prompt</h3>
          <el-input
            v-model="negativePrompt"
            type="textarea"
            :rows="2"
            placeholder="Things to avoid in the image (optional)..."
            class="prompt-input"
          />
        </div>

        <div class="panel-card">
          <h3>Model Selection</h3>
          <ModelSelector
            v-model="selectedModel"
            :models="availableModels"
            label=""
            placeholder="Select a model"
          />
        </div>

        <div class="panel-card">
          <h3>Generation Settings</h3>
          
          <div class="setting-row">
            <label>Width</label>
            <el-slider
              v-model="settings.width"
              :min="256"
              :max="1024"
              :step="64"
              show-input
            />
          </div>
          
          <div class="setting-row">
            <label>Height</label>
            <el-slider
              v-model="settings.height"
              :min="256"
              :max="1024"
              :step="64"
              show-input
            />
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
          
          <div class="setting-row">
            <label>Seed</label>
            <div class="seed-input">
              <el-input-number
                v-model="settings.seed"
                :min="-1"
                :max="2147483647"
                controls-position="right"
              />
              <el-button :icon="Refresh" @click="randomizeSeed" />
            </div>
            <span class="setting-hint">-1 for random seed</span>
          </div>
        </div>

        <el-button 
          type="primary" 
          size="large" 
          :loading="isGenerating"
          :disabled="!prompt.trim()"
          class="generate-btn"
          @click="generateImage"
        >
          <el-icon v-if="!isGenerating"><MagicStick /></el-icon>
          {{ isGenerating ? 'Generating...' : 'Generate Image' }}
        </el-button>
      </div>

      <!-- Output Panel -->
      <div class="output-panel">
        <div class="panel-card result-card">
          <h3>Generated Image</h3>
          
          <div v-if="isGenerating" class="generating-state">
            <div class="loading-animation">
              <el-icon class="is-loading"><Loading /></el-icon>
            </div>
            <ProgressBar 
              :percentage="progress" 
              label="Generating..."
              :show-stats="true"
              :current-value="currentStep"
              :total-value="settings.steps"
            />
          </div>
          
          <div v-else-if="generatedImage" class="image-result">
            <img :src="generatedImage" alt="Generated image" />
            <div class="image-actions">
              <el-button :icon="Download" @click="downloadImage">Download</el-button>
              <el-button :icon="ZoomIn" @click="viewFullscreen">View Full</el-button>
              <el-button :icon="Refresh" @click="regenerate">Regenerate</el-button>
            </div>
          </div>
          
          <div v-else class="empty-result">
            <el-icon><Picture /></el-icon>
            <p>Your generated image will appear here</p>
          </div>
        </div>

        <!-- Generation History -->
        <div class="panel-card history-card">
          <h3>Recent Generations</h3>
          <div class="history-grid" v-if="history.length > 0">
            <div 
              v-for="item in history" 
              :key="item.id"
              class="history-item"
              @click="selectFromHistory(item)"
            >
              <img :src="item.imageUrl" :alt="item.prompt" />
            </div>
          </div>
          <div v-else class="empty-history">
            <p>No generation history yet</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { 
  MagicStick, 
  Loading, 
  Download, 
  ZoomIn, 
  Refresh, 
  Picture 
} from '@element-plus/icons-vue'
import { ModelSelector, ProgressBar } from '@/components/common'
import { textToImageApi, type GenerationParams } from '@/api/endpoints/generation'
import type { ModelOption } from '@/components/common/ModelSelector.vue'

interface HistoryItem {
  id: string
  prompt: string
  imageUrl: string
  timestamp: Date
}

const prompt = ref('')
const negativePrompt = ref('')
const selectedModel = ref('')
const isGenerating = ref(false)
const progress = ref(0)
const currentStep = ref(0)
const generatedImage = ref<string | null>(null)
const history = ref<HistoryItem[]>([])

const settings = reactive({
  width: 512,
  height: 512,
  steps: 30,
  guidanceScale: 7.5,
  seed: -1
})

const availableModels = ref<ModelOption[]>([
  {
    id: 'stable-diffusion-v1-5',
    name: 'Stable Diffusion v1.5',
    description: 'Classic SD model, good for general purposes',
    category: 'Stable Diffusion',
    size: '4.27 GB'
  },
  {
    id: 'stable-diffusion-xl',
    name: 'Stable Diffusion XL',
    description: 'High-quality 1024x1024 image generation',
    category: 'Stable Diffusion',
    size: '6.94 GB'
  },
  {
    id: 'dreamshaper-8',
    name: 'DreamShaper 8',
    description: 'Artistic and creative image generation',
    category: 'Community Models',
    size: '2.13 GB'
  }
])

const randomizeSeed = () => {
  settings.seed = Math.floor(Math.random() * 2147483647)
}

const generateImage = async () => {
  if (!prompt.value.trim()) {
    ElMessage.warning('Please enter a prompt')
    return
  }

  isGenerating.value = true
  progress.value = 0
  currentStep.value = 0

  try {
    const params: GenerationParams = {
      prompt: prompt.value,
      negative_prompt: negativePrompt.value || undefined,
      width: settings.width,
      height: settings.height,
      num_inference_steps: settings.steps,
      guidance_scale: settings.guidanceScale,
      seed: settings.seed === -1 ? undefined : settings.seed,
      model_id: selectedModel.value || undefined
    }

    // Simulate progress for demo
    const progressInterval = setInterval(() => {
      if (currentStep.value < settings.steps) {
        currentStep.value++
        progress.value = (currentStep.value / settings.steps) * 100
      }
    }, 100)

    const response = await textToImageApi.generate(params)
    
    clearInterval(progressInterval)
    progress.value = 100
    currentStep.value = settings.steps

    if (response.image_url) {
      generatedImage.value = response.image_url
      
      // Add to history
      history.value.unshift({
        id: response.task_id,
        prompt: prompt.value,
        imageUrl: response.image_url,
        timestamp: new Date()
      })
      
      // Keep only last 10 items
      if (history.value.length > 10) {
        history.value = history.value.slice(0, 10)
      }
    }

    ElMessage.success('Image generated successfully!')
  } catch (error: any) {
    ElMessage.error(error.message || 'Failed to generate image')
  } finally {
    isGenerating.value = false
  }
}

const downloadImage = () => {
  if (!generatedImage.value) return
  
  const link = document.createElement('a')
  link.href = generatedImage.value
  link.download = `generated_${Date.now()}.png`
  link.click()
}

const viewFullscreen = () => {
  if (!generatedImage.value) return
  window.open(generatedImage.value, '_blank')
}

const regenerate = () => {
  randomizeSeed()
  generateImage()
}

const selectFromHistory = (item: HistoryItem) => {
  generatedImage.value = item.imageUrl
  prompt.value = item.prompt
}

onMounted(async () => {
  // Fetch available models
  try {
    const models = await textToImageApi.getModels()
    if (models.length > 0) {
      availableModels.value = models.map(m => ({
        id: m.id,
        name: m.name,
        description: m.description,
        category: 'Available Models'
      }))
    }
  } catch (e) {
    // Use default models if API fails
  }
})
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.text-to-image-view {
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

.seed-input {
  display: flex;
  gap: 8px;
  
  .el-input-number {
    flex: 1;
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

.result-card {
  min-height: 400px;
}

.generating-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 40px;
}

.loading-animation {
  margin-bottom: 24px;
  
  .el-icon {
    font-size: 64px;
    color: $neon-cyan;
  }
}

.image-result {
  img {
    width: 100%;
    max-height: 600px;
    object-fit: contain;
    border-radius: 12px;
    margin-bottom: 16px;
  }
}

.image-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
}

.empty-result {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px;
  color: $text-muted;
  
  .el-icon {
    font-size: 64px;
    margin-bottom: 16px;
    opacity: 0.5;
  }
}

.history-card {
  h3 {
    margin-bottom: 16px;
  }
}

.history-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
  gap: 12px;
}

.history-item {
  aspect-ratio: 1;
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  border: 2px solid transparent;
  transition: all 0.3s ease;
  
  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  
  &:hover {
    border-color: $neon-cyan;
    transform: scale(1.05);
  }
}

.empty-history {
  text-align: center;
  padding: 24px;
  color: $text-muted;
}
</style>
