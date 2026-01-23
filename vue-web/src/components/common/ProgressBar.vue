<template>
  <div class="progress-wrapper" :class="{ animated: animated }">
    <div class="progress-header" v-if="showLabel || showPercentage">
      <span class="progress-label" v-if="showLabel">{{ label }}</span>
      <span class="progress-percentage" v-if="showPercentage">{{ displayPercentage }}%</span>
    </div>
    
    <div class="progress-track" :style="{ height: height + 'px' }">
      <div 
        class="progress-fill" 
        :class="[statusClass, { striped: striped }]"
        :style="{ width: displayPercentage + '%' }"
      >
        <div v-if="showGlow" class="progress-glow"></div>
      </div>
    </div>
    
    <div class="progress-footer" v-if="showStats">
      <span class="stat-item" v-if="currentValue !== undefined">
        {{ formatValue(currentValue) }} / {{ formatValue(totalValue) }}
      </span>
      <span class="stat-item" v-if="speed">
        {{ speed }}
      </span>
      <span class="stat-item" v-if="eta">
        ETA: {{ eta }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  percentage: number
  label?: string
  status?: 'default' | 'success' | 'warning' | 'error' | 'processing'
  height?: number
  showLabel?: boolean
  showPercentage?: boolean
  showStats?: boolean
  showGlow?: boolean
  animated?: boolean
  striped?: boolean
  currentValue?: number
  totalValue?: number
  speed?: string
  eta?: string
  valueUnit?: string
}>(), {
  percentage: 0,
  status: 'default',
  height: 8,
  showLabel: true,
  showPercentage: true,
  showStats: false,
  showGlow: true,
  animated: true,
  striped: false,
  valueUnit: ''
})

const displayPercentage = computed(() => {
  return Math.min(100, Math.max(0, Math.round(props.percentage)))
})

const statusClass = computed(() => {
  return `status-${props.status}`
})

const formatValue = (value: number | undefined): string => {
  if (value === undefined) return ''
  if (props.valueUnit === 'bytes') {
    return formatBytes(value)
  }
  return value.toLocaleString() + (props.valueUnit ? ` ${props.valueUnit}` : '')
}

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.progress-wrapper {
  width: 100%;
  
  &.animated {
    .progress-fill {
      transition: width 0.3s ease;
    }
  }
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.progress-label {
  font-size: 14px;
  color: $text-secondary;
  font-weight: 500;
}

.progress-percentage {
  font-size: 14px;
  color: $neon-cyan;
  font-weight: 600;
  font-family: monospace;
}

.progress-track {
  width: 100%;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 10px;
  overflow: hidden;
  position: relative;
}

.progress-fill {
  height: 100%;
  border-radius: 10px;
  position: relative;
  overflow: hidden;
  
  // Default status (cyan gradient)
  &.status-default {
    background: $gradient-neon;
  }
  
  &.status-success {
    background: linear-gradient(90deg, $neon-green, darken($neon-green, 15%));
  }
  
  &.status-warning {
    background: linear-gradient(90deg, $neon-yellow, $neon-orange);
  }
  
  &.status-error {
    background: linear-gradient(90deg, #ff4757, #ff6b6b);
  }
  
  &.status-processing {
    background: linear-gradient(90deg, $neon-purple, $neon-magenta);
  }
  
  // Striped effect
  &.striped {
    background-size: 30px 30px;
    background-image: linear-gradient(
      135deg,
      rgba(255, 255, 255, 0.15) 25%,
      transparent 25%,
      transparent 50%,
      rgba(255, 255, 255, 0.15) 50%,
      rgba(255, 255, 255, 0.15) 75%,
      transparent 75%,
      transparent
    );
    animation: stripes 1s linear infinite;
  }
}

.progress-glow {
  position: absolute;
  top: 0;
  right: 0;
  width: 100px;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
  animation: shimmer 2s infinite;
}

.progress-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
  flex-wrap: wrap;
  gap: 12px;
}

.stat-item {
  font-size: 12px;
  color: $text-muted;
  font-family: monospace;
}

@keyframes stripes {
  0% {
    background-position: 0 0;
  }
  100% {
    background-position: 30px 0;
  }
}

@keyframes shimmer {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(200%);
  }
}
</style>
