import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '@/api'

export interface ModelInfo {
  id: string
  name: string
  category: string
  description?: string
  size?: string
  parameters?: string
  status: 'available' | 'loaded' | 'downloading' | 'error'
  loadProgress?: number
}

export interface SystemStatus {
  gpuAvailable: boolean
  gpuName?: string
  gpuMemoryTotal?: number
  gpuMemoryUsed?: number
  cpuUsage?: number
  memoryUsage?: number
}

export const useModelStore = defineStore('model', () => {
  // State
  const models = ref<ModelInfo[]>([])
  const loadedModels = ref<string[]>([])
  const currentModel = ref<string | null>(null)
  const systemStatus = ref<SystemStatus>({
    gpuAvailable: false
  })
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  // Getters
  const availableModels = computed(() => 
    models.value.filter(m => m.status === 'available' || m.status === 'loaded')
  )

  const modelsByCategory = computed(() => {
    const grouped: Record<string, ModelInfo[]> = {}
    models.value.forEach(model => {
      if (!grouped[model.category]) {
        grouped[model.category] = []
      }
      grouped[model.category].push(model)
    })
    return grouped
  })

  const isModelLoaded = computed(() => (modelId: string) => 
    loadedModels.value.includes(modelId)
  )

  // Actions
  async function fetchModels() {
    isLoading.value = true
    error.value = null
    try {
      const response = await api.get<ModelInfo[]>('/models/list')
      models.value = response
    } catch (e: any) {
      error.value = e.message || 'Failed to fetch models'
    } finally {
      isLoading.value = false
    }
  }

  async function loadModel(modelId: string) {
    const model = models.value.find(m => m.id === modelId)
    if (!model) return

    model.status = 'downloading'
    model.loadProgress = 0

    try {
      await api.post(`/models/load/${modelId}`)
      model.status = 'loaded'
      model.loadProgress = 100
      loadedModels.value.push(modelId)
    } catch (e: any) {
      model.status = 'error'
      error.value = e.message || 'Failed to load model'
    }
  }

  async function unloadModel(modelId: string) {
    try {
      await api.post(`/models/unload/${modelId}`)
      const index = loadedModels.value.indexOf(modelId)
      if (index > -1) {
        loadedModels.value.splice(index, 1)
      }
      const model = models.value.find(m => m.id === modelId)
      if (model) {
        model.status = 'available'
      }
    } catch (e: any) {
      error.value = e.message || 'Failed to unload model'
    }
  }

  async function fetchSystemStatus() {
    try {
      const response = await api.get<SystemStatus>('/system/status')
      systemStatus.value = response
    } catch (e: any) {
      console.error('Failed to fetch system status:', e)
    }
  }

  function setCurrentModel(modelId: string | null) {
    currentModel.value = modelId
  }

  function updateModelProgress(modelId: string, progress: number) {
    const model = models.value.find(m => m.id === modelId)
    if (model) {
      model.loadProgress = progress
      if (progress >= 100) {
        model.status = 'loaded'
      }
    }
  }

  return {
    // State
    models,
    loadedModels,
    currentModel,
    systemStatus,
    isLoading,
    error,
    // Getters
    availableModels,
    modelsByCategory,
    isModelLoaded,
    // Actions
    fetchModels,
    loadModel,
    unloadModel,
    fetchSystemStatus,
    setCurrentModel,
    updateModelProgress
  }
})
