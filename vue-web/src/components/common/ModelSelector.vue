<template>
  <div class="model-selector">
    <div class="selector-header" v-if="label">
      <span class="selector-label">{{ label }}</span>
      <el-tooltip v-if="tooltip" :content="tooltip" placement="top">
        <el-icon class="info-icon"><InfoFilled /></el-icon>
      </el-tooltip>
    </div>
    
    <el-select
      v-model="selectedModel"
      :placeholder="placeholder"
      :disabled="disabled || loading"
      :loading="loading"
      filterable
      class="model-select"
      @change="handleChange"
    >
      <template #prefix>
        <el-icon><Cpu /></el-icon>
      </template>
      
      <el-option-group
        v-for="group in groupedModels"
        :key="group.label"
        :label="group.label"
      >
        <el-option
          v-for="model in group.models"
          :key="model.id"
          :label="model.name"
          :value="model.id"
          :disabled="model.disabled"
        >
          <div class="model-option">
            <div class="model-info">
              <span class="model-name">{{ model.name }}</span>
              <span class="model-desc" v-if="model.description">{{ model.description }}</span>
            </div>
            <div class="model-meta">
              <el-tag v-if="model.size" size="small" type="info">{{ model.size }}</el-tag>
              <el-tag v-if="model.status === 'loaded'" size="small" type="success">Loaded</el-tag>
              <el-tag v-else-if="model.status === 'downloading'" size="small" type="warning">Downloading</el-tag>
            </div>
          </div>
        </el-option>
      </el-option-group>
    </el-select>
    
    <!-- Selected Model Info Card -->
    <div v-if="showModelInfo && currentModel" class="model-info-card">
      <div class="info-row">
        <span class="info-label">Model:</span>
        <span class="info-value">{{ currentModel.name }}</span>
      </div>
      <div class="info-row" v-if="currentModel.version">
        <span class="info-label">Version:</span>
        <span class="info-value">{{ currentModel.version }}</span>
      </div>
      <div class="info-row" v-if="currentModel.size">
        <span class="info-label">Size:</span>
        <span class="info-value">{{ currentModel.size }}</span>
      </div>
      <div class="info-row" v-if="currentModel.parameters">
        <span class="info-label">Parameters:</span>
        <span class="info-value">{{ currentModel.parameters }}</span>
      </div>
      <div class="info-description" v-if="currentModel.description">
        {{ currentModel.description }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { Cpu, InfoFilled } from '@element-plus/icons-vue'

export interface ModelOption {
  id: string
  name: string
  description?: string
  category?: string
  size?: string
  version?: string
  parameters?: string
  status?: 'available' | 'loaded' | 'downloading'
  disabled?: boolean
}

export interface ModelGroup {
  label: string
  models: ModelOption[]
}

const props = withDefaults(defineProps<{
  modelValue?: string
  models: ModelOption[]
  label?: string
  placeholder?: string
  tooltip?: string
  disabled?: boolean
  loading?: boolean
  showModelInfo?: boolean
}>(), {
  placeholder: 'Select a model',
  disabled: false,
  loading: false,
  showModelInfo: true
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
  (e: 'change', model: ModelOption | undefined): void
}>()

const selectedModel = ref(props.modelValue || '')

// Group models by category
const groupedModels = computed<ModelGroup[]>(() => {
  const groups: Record<string, ModelOption[]> = {}
  
  props.models.forEach(model => {
    const category = model.category || 'Other'
    if (!groups[category]) {
      groups[category] = []
    }
    groups[category].push(model)
  })
  
  return Object.entries(groups).map(([label, models]) => ({
    label,
    models
  }))
})

const currentModel = computed(() => {
  return props.models.find(m => m.id === selectedModel.value)
})

const handleChange = (value: string) => {
  emit('update:modelValue', value)
  emit('change', props.models.find(m => m.id === value))
}

watch(() => props.modelValue, (newValue) => {
  if (newValue !== selectedModel.value) {
    selectedModel.value = newValue || ''
  }
})
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.model-selector {
  width: 100%;
}

.selector-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.selector-label {
  font-size: 14px;
  color: $text-secondary;
  font-weight: 500;
}

.info-icon {
  font-size: 14px;
  color: $text-muted;
  cursor: help;
  
  &:hover {
    color: $neon-cyan;
  }
}

.model-select {
  width: 100%;
  
  :deep(.el-input__wrapper) {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid $glass-border !important;
    border-radius: 8px;
    
    &:hover {
      border-color: rgba($neon-cyan, 0.3) !important;
    }
    
    &.is-focus {
      border-color: $neon-cyan !important;
      box-shadow: 0 0 10px rgba($neon-cyan, 0.2) !important;
    }
  }
  
  :deep(.el-input__inner) {
    color: $text-primary !important;
  }
  
  :deep(.el-input__prefix) {
    color: $neon-cyan;
  }
}

.model-option {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 0;
}

.model-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.model-name {
  font-size: 14px;
  color: $text-primary;
}

.model-desc {
  font-size: 12px;
  color: $text-muted;
  max-width: 250px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-meta {
  display: flex;
  gap: 6px;
}

.model-info-card {
  margin-top: 12px;
  padding: 16px;
  @include glass-effect;
  border-radius: 12px;
}

.info-row {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
  
  &:last-child {
    margin-bottom: 0;
  }
}

.info-label {
  font-size: 13px;
  color: $text-muted;
  width: 100px;
  flex-shrink: 0;
}

.info-value {
  font-size: 13px;
  color: $text-primary;
  font-weight: 500;
}

.info-description {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid $glass-border;
  font-size: 13px;
  color: $text-secondary;
  line-height: 1.6;
}
</style>
