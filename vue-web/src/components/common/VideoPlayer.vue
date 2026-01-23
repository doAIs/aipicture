<template>
  <div class="video-player" :class="{ fullscreen: isFullscreen }">
    <div class="video-container" ref="containerRef">
      <video
        ref="videoRef"
        :src="src"
        :poster="poster"
        :autoplay="autoplay"
        :loop="loop"
        :muted="isMuted"
        @timeupdate="onTimeUpdate"
        @loadedmetadata="onLoadedMetadata"
        @play="isPlaying = true"
        @pause="isPlaying = false"
        @ended="onEnded"
        @error="onError"
      />
      
      <!-- Play/Pause Overlay -->
      <div class="video-overlay" @click="togglePlay">
        <transition name="fade">
          <div v-if="!isPlaying && !isLoading" class="play-button">
            <el-icon><VideoPlay /></el-icon>
          </div>
        </transition>
        <div v-if="isLoading" class="loading-indicator">
          <el-icon class="is-loading"><Loading /></el-icon>
        </div>
      </div>
      
      <!-- Controls -->
      <div class="video-controls" v-show="showControls">
        <!-- Progress Bar -->
        <div class="progress-container" @click="seekTo">
          <div class="progress-bar">
            <div class="progress-buffered" :style="{ width: bufferedPercent + '%' }"></div>
            <div class="progress-current" :style="{ width: progressPercent + '%' }"></div>
          </div>
        </div>
        
        <div class="controls-row">
          <div class="controls-left">
            <!-- Play/Pause -->
            <el-button :icon="isPlaying ? VideoPause : VideoPlay" circle size="small" @click="togglePlay" />
            
            <!-- Volume -->
            <div class="volume-control">
              <el-button :icon="isMuted ? Mute : Microphone" circle size="small" @click="toggleMute" />
              <el-slider
                v-model="volume"
                :min="0"
                :max="100"
                :show-tooltip="false"
                class="volume-slider"
                @input="setVolume"
              />
            </div>
            
            <!-- Time Display -->
            <span class="time-display">
              {{ formatTime(currentTime) }} / {{ formatTime(duration) }}
            </span>
          </div>
          
          <div class="controls-right">
            <!-- Playback Speed -->
            <el-dropdown trigger="click" @command="setPlaybackRate">
              <el-button circle size="small">
                <span class="speed-text">{{ playbackRate }}x</span>
              </el-button>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item v-for="rate in playbackRates" :key="rate" :command="rate">
                    {{ rate }}x
                  </el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
            
            <!-- Fullscreen -->
            <el-button :icon="isFullscreen ? Close : FullScreen" circle size="small" @click="toggleFullscreen" />
            
            <!-- Download -->
            <el-button v-if="downloadable" :icon="Download" circle size="small" @click="downloadVideo" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { 
  VideoPlay, 
  VideoPause, 
  Microphone, 
  Mute, 
  FullScreen, 
  Close,
  Download,
  Loading
} from '@element-plus/icons-vue'

const props = withDefaults(defineProps<{
  src: string
  poster?: string
  autoplay?: boolean
  loop?: boolean
  downloadable?: boolean
}>(), {
  autoplay: false,
  loop: false,
  downloadable: true
})

const emit = defineEmits<{
  (e: 'play'): void
  (e: 'pause'): void
  (e: 'ended'): void
  (e: 'error', error: Event): void
}>()

const videoRef = ref<HTMLVideoElement>()
const containerRef = ref<HTMLDivElement>()

const isPlaying = ref(false)
const isLoading = ref(true)
const isMuted = ref(false)
const isFullscreen = ref(false)
const showControls = ref(true)
const currentTime = ref(0)
const duration = ref(0)
const bufferedPercent = ref(0)
const volume = ref(100)
const playbackRate = ref(1)
const playbackRates = [0.5, 0.75, 1, 1.25, 1.5, 2]

let controlsTimeout: ReturnType<typeof setTimeout> | null = null

const progressPercent = computed(() => {
  return duration.value ? (currentTime.value / duration.value) * 100 : 0
})

const togglePlay = () => {
  if (!videoRef.value) return
  
  if (isPlaying.value) {
    videoRef.value.pause()
    emit('pause')
  } else {
    videoRef.value.play()
    emit('play')
  }
}

const toggleMute = () => {
  if (!videoRef.value) return
  isMuted.value = !isMuted.value
  videoRef.value.muted = isMuted.value
}

const setVolume = (value: number) => {
  if (!videoRef.value) return
  videoRef.value.volume = value / 100
  isMuted.value = value === 0
}

const setPlaybackRate = (rate: number) => {
  if (!videoRef.value) return
  playbackRate.value = rate
  videoRef.value.playbackRate = rate
}

const seekTo = (event: MouseEvent) => {
  if (!videoRef.value) return
  
  const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
  const percent = (event.clientX - rect.left) / rect.width
  videoRef.value.currentTime = percent * duration.value
}

const toggleFullscreen = async () => {
  if (!containerRef.value) return
  
  if (isFullscreen.value) {
    await document.exitFullscreen()
  } else {
    await containerRef.value.requestFullscreen()
  }
  isFullscreen.value = !isFullscreen.value
}

const downloadVideo = () => {
  const link = document.createElement('a')
  link.href = props.src
  link.download = 'video.mp4'
  link.click()
}

const onTimeUpdate = () => {
  if (!videoRef.value) return
  currentTime.value = videoRef.value.currentTime
  
  // Update buffered
  if (videoRef.value.buffered.length > 0) {
    bufferedPercent.value = (videoRef.value.buffered.end(0) / duration.value) * 100
  }
}

const onLoadedMetadata = () => {
  if (!videoRef.value) return
  duration.value = videoRef.value.duration
  isLoading.value = false
}

const onEnded = () => {
  isPlaying.value = false
  emit('ended')
}

const onError = (event: Event) => {
  isLoading.value = false
  emit('error', event)
}

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

const showControlsTemporarily = () => {
  showControls.value = true
  if (controlsTimeout) clearTimeout(controlsTimeout)
  controlsTimeout = setTimeout(() => {
    if (isPlaying.value) showControls.value = false
  }, 3000)
}

onMounted(() => {
  document.addEventListener('fullscreenchange', () => {
    isFullscreen.value = !!document.fullscreenElement
  })
})

onUnmounted(() => {
  if (controlsTimeout) clearTimeout(controlsTimeout)
})

// Expose methods
defineExpose({
  play: () => videoRef.value?.play(),
  pause: () => videoRef.value?.pause(),
  seek: (time: number) => {
    if (videoRef.value) videoRef.value.currentTime = time
  }
})
</script>

<style lang="scss" scoped>
@import '@/styles/tech-theme.scss';

.video-player {
  width: 100%;
  border-radius: 16px;
  overflow: hidden;
  @include glass-effect;
  
  &.fullscreen {
    border-radius: 0;
  }
}

.video-container {
  position: relative;
  width: 100%;
  background: #000;
  
  video {
    width: 100%;
    display: block;
  }
}

.video-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}

.play-button {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: rgba($neon-cyan, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
  
  .el-icon {
    font-size: 40px;
    color: $bg-primary;
  }
  
  &:hover {
    transform: scale(1.1);
    @include neon-glow($neon-cyan);
  }
}

.loading-indicator {
  .el-icon {
    font-size: 48px;
    color: $neon-cyan;
  }
}

.video-controls {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: linear-gradient(transparent, rgba(0, 0, 0, 0.8));
  padding: 16px;
}

.progress-container {
  cursor: pointer;
  padding: 8px 0;
}

.progress-bar {
  height: 4px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 2px;
  position: relative;
  
  &:hover {
    height: 6px;
  }
}

.progress-buffered {
  position: absolute;
  height: 100%;
  background: rgba(255, 255, 255, 0.3);
  border-radius: 2px;
}

.progress-current {
  position: absolute;
  height: 100%;
  background: $gradient-neon;
  border-radius: 2px;
}

.controls-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 8px;
}

.controls-left,
.controls-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.volume-control {
  display: flex;
  align-items: center;
  gap: 8px;
  
  .volume-slider {
    width: 80px;
    
    :deep(.el-slider__runway) {
      background: rgba(255, 255, 255, 0.2);
    }
    
    :deep(.el-slider__bar) {
      background: $neon-cyan;
    }
    
    :deep(.el-slider__button) {
      width: 12px;
      height: 12px;
      border-color: $neon-cyan;
    }
  }
}

.time-display {
  font-size: 13px;
  color: $text-secondary;
  font-family: monospace;
}

.speed-text {
  font-size: 11px;
  font-weight: 600;
}

.el-button {
  background: rgba(255, 255, 255, 0.1) !important;
  border: none !important;
  color: $text-primary !important;
  
  &:hover {
    background: rgba($neon-cyan, 0.2) !important;
    color: $neon-cyan !important;
  }
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
